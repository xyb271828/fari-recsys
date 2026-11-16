"""Fari ANN export, learned ranking, and diversity experiments."""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import faiss
import numpy as np
import torch
from lightgbm import LGBMRanker

from discoverylib.metrics import ndcg, percentile
from recsys.benchmark import baseline_models, positive_matrix
from recsys.data import requests, synthetic
from recsys.towers import fit_tower, vectors


def recency_features(data):
    rows, items, days, _, positive = data.events.T
    training = (days <= data.train_end) & (positive == 1)
    category_ids = np.unique(data.catalog.categories, return_inverse=True)[1]
    activity_counts = []
    for window in (7, 30):
        selected = training & (days > data.train_end - window)
        activity_counts.append(np.bincount(items[selected], minlength=data.catalog.size))
    counts = np.zeros((data.users, int(category_ids.max()) + 1), dtype=np.int32)
    np.add.at(counts, (rows[training], category_ids[items[training]]), 1)
    return activity_counts, counts, category_ids


def rank_features(contexts, candidates, user_vectors, item_vectors, als, features):
    activity_counts, affinity, categories = features
    rows = []
    for request, ids, query in zip(contexts, candidates, user_vectors, strict=True):
        user, _, hour, history, _ = request
        rows.append(np.column_stack((
            item_vectors[ids] @ query,
            als(np.array([user]))[0, ids],
            affinity[user, categories[ids]] / max(int(affinity[user].sum()), 1),
            activity_counts[0][ids], activity_counts[1][ids],
            np.full(len(ids), hour), np.full(len(ids), len(history)),
        )))
    return np.vstack(rows)


def recent_candidates(data, contexts):
    by_user = defaultdict(list)
    for user, item, day, hour, _ in data.events:
        by_user[user].append((day * 4 + hour, item))
    return [
        {item for timestamp, item in by_user[user] if 0 < (day * 4 + hour) - timestamp <= 4}
        for user, day, hour, *_ in contexts
    ]


def mmr(ids, scores, similarity, weight, excluded, k=10):
    # Scale scores within the request to put relevance and cosine penalties on comparable ranges.
    scores = (scores - scores.min()) / max(float(np.ptp(scores)), 1e-8)
    eligible = ~np.isin(ids, list(excluded))
    penalty = np.zeros(len(ids))
    chosen = []
    for _ in range(min(k, int(eligible.sum()))):
        objective = (1 - weight) * scores - weight * penalty
        index = int(np.argmax(np.where(eligible, objective, -np.inf)))
        chosen.append(int(ids[index]))
        eligible[index] = False
        penalty = np.maximum(penalty, similarity[ids, ids[index]])
    return chosen


def build_feed(ids, scores, similarity, weight, excluded, pinned=(), k=10):
    """Put externally scheduled IDs first and exclude them from the MMR pool."""
    pinned_ids = list(dict.fromkeys(pinned))
    remaining = max(0, k - len(pinned_ids))
    ranked = mmr(ids, scores, similarity, weight, excluded | set(pinned_ids), remaining)
    return pinned_ids + ranked


def quality(rankings, contexts, category):
    ndcgs, diversity = [], []
    for ranking, request in zip(rankings, contexts, strict=True):
        ndcgs.append(ndcg(ranking, dict.fromkeys(request[-1], 1)))
        categories = category[ranking]
        i, j = np.triu_indices(len(ranking), 1)
        diversity.append(float(np.mean(categories[i] != categories[j])) if len(i) else 0.0)
    return {"ndcg@10": mean(ndcgs), "category_diversity": mean(diversity)}


def evaluate_cold_start(parts, scores, popularity):
    result = {}
    for count in (0, 1, 3, 5):
        subset = [(i, row) for i, row in enumerate(parts) if len(row[3]) == count]
        model_scores = [ndcg(scores[i], dict.fromkeys(row[-1], 1)) for i, row in subset]
        fallback_scores = [ndcg(popularity, dict.fromkeys(row[-1], 1)) for _, row in subset]
        result[count] = {
            "requests": len(subset),
            "ranker_ndcg@10": mean(model_scores) if subset else None,
            "popularity_ndcg@10": mean(fallback_scores) if subset else None,
        }
    return result


def run(output, seed=42, epochs=10):
    torch.set_num_threads(2)
    faiss.omp_set_num_threads(1)
    data = synthetic(seed)
    train, validation, test = requests(data)
    matrix, counts, recent = positive_matrix(data)
    als = baseline_models(matrix, counts, recent, seed)["als"]
    rank_train = [r for r in validation if r[1] <= 79]
    tune = [r for r in validation if r[1] > 79]
    model, _ = fit_tower(data, train, seed=seed, epochs=epochs, correction=True)
    _, items = vectors(model, train[:1])
    index = faiss.IndexHNSWFlat(items.shape[1], 24, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 100
    index.hnsw.efSearch = 256
    index.add(items)
    similarity = items @ items.T
    properties = recency_features(data)
    features, candidates, query_vectors = [], [], []
    for part in (rank_train, tune, test):
        queries, _ = vectors(model, part)
        _, ids = index.search(queries, 200)
        if np.any(ids < 0):
            raise RuntimeError("ANN returned fewer than 200 candidates")
        query_vectors.append(queries)
        candidates.append(ids)
        features.append(rank_features(part, ids, queries, items, als, properties))

    labels = [int(item in row[-1]) for row, ids in zip(rank_train, candidates[0], strict=True) for item in ids]
    ranker = LGBMRanker(
        n_estimators=120, num_leaves=15, learning_rate=0.04, random_state=seed, n_jobs=2, verbosity=-1
    )
    ranker.fit(features[0], labels, group=[len(ids) for ids in candidates[0]])
    analyses = []
    raw_rankings = []
    weights = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    for part, ids, x in zip((tune, test), candidates[1:], features[1:], strict=True):
        predictions = ranker.booster_.predict(x).reshape(ids.shape)
        blocked = recent_candidates(data, part)
        raw = np.take_along_axis(ids, np.argsort(-predictions, axis=1)[:, :10], axis=1).tolist()
        raw_rankings.append(raw)
        rows = {}
        for weight in weights:
            reranked = [
                build_feed(i, scores, similarity, weight, excluded)
                for i, scores, excluded in zip(ids, predictions, blocked, strict=True)
            ]
            rows[weight] = quality(reranked, part, properties[-1])
        analyses.append(rows)
    floor = analyses[0][0.0]["ndcg@10"] * 0.95
    operating = max(
        (w for w in weights if analyses[0][w]["ndcg@10"] >= floor),
        key=lambda w: analyses[0][w]["category_diversity"],
    )
    popular = np.argsort(-counts, kind="stable").tolist()
    cold_tune = evaluate_cold_start(tune, raw_rankings[0], popular)
    # A policy proposal selected only from validation and subject to sparse bucket support.
    switch = next((
        count for count in (1, 3, 5)
        if cold_tune[count]["requests"] >= 15
        and cold_tune[count]["ranker_ndcg@10"] >= cold_tune[count]["popularity_ndcg@10"]
    ), None)

    snapshots = {}
    for user, item, day, hour, positive in sorted(data.events.tolist(), key=lambda row: (row[2], row[3])):
        if day <= data.train_end and positive == 1:
            snapshots.setdefault(user, []).append(item)
    serve_rows = [
        (user, data.train_end + 1, bucket, snapshots.get(user, [])[-16:], ())
        for user in sorted(snapshots) for bucket in range(4)
    ]
    users, _ = vectors(model, serve_rows)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "users.npz", user_ids=sorted(snapshots), vectors=users.reshape(-1, 4, items.shape[1]))
    faiss.write_index(index, str(output / "activities.faiss"))
    (output / "popularity.json").write_text(json.dumps(popular))
    ranker.booster_.save_model(str(output / "ranker.txt"))

    exact = np.argsort(-(query_vectors[2] @ items.T), axis=1)[:, :200]
    ann_overlap = mean(len(set(a) & set(b)) / 200 for a, b in zip(exact, candidates[2], strict=True))
    latency = []
    for query in query_vectors[2]:
        start = time.perf_counter_ns()
        index.search(query.reshape(1, -1), 200)
        latency.append((time.perf_counter_ns() - start) / 1e6)
    result = {
        "settings": {"seed": seed, "epochs": epochs, "ann": "HNSW M=24 efConstruction=100 efSearch=256"},
        "ranking_train_requests": len(rank_train), "mmr_validation_requests": len(tune),
        "ann_exact_top200_overlap": ann_overlap,
        "ann_search_ms": {f"p{p}": percentile(latency, p) for p in (50, 95, 99)},
        "test_retrieval_top10": quality(candidates[2][:, :10], test, properties[-1]),
        "test_ranker_top10": quality(raw_rankings[1], test, properties[-1]),
        "mmr_validation": analyses[0], "mmr_test": analyses[1], "mmr_weight": operating,
        "cold_start_validation": cold_tune,
        "cold_start_test": evaluate_cold_start(test, raw_rankings[1], popular),
        "cold_start_switch_proposal": switch,
    }
    (output / "ranking.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
