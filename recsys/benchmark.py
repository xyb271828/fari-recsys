"""Fit and compare recommenders on identical temporal requests."""

import argparse
import json
import platform
import sys
from importlib.metadata import version
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from implicit.als import AlternatingLeastSquares
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix

from discoverylib.metrics import coverage, paired_interval, relevance
from recsys.data import movielens, requests, synthetic
from recsys.towers import fit_tower, vectors


def positive_matrix(data):
    user, item, day, _, positive = data.events.T
    include = (day <= data.train_end) & (positive == 1)
    axes = (user[include], item[include])
    matrix = csr_matrix(
        (np.ones(include.sum(), dtype=np.float32), axes), shape=(data.users, data.catalog.size)
    )
    counts = np.bincount(item[include], minlength=data.catalog.size)
    recent = np.bincount(
        item[include], np.exp2((day[include] - data.train_end) / 14), minlength=data.catalog.size
    )
    return matrix, counts, recent


def baseline_models(matrix, counts, recent, seed):
    gram = (matrix.T @ matrix).toarray().astype(np.float64)
    norms = np.sqrt(np.diag(gram))
    denominator = np.maximum(norms[:, None] * norms[None, :], 1)
    cosine = (gram / denominator).astype(np.float32)
    np.fill_diagonal(cosine, 0)
    gram.flat[::len(gram) + 1] += 100
    inverse = cho_solve(cho_factor(gram), np.eye(len(gram)), overwrite_b=True)
    ease = (-inverse / np.diag(inverse)).astype(np.float32)
    np.fill_diagonal(ease, 0)
    als = AlternatingLeastSquares(factors=32, regularization=0.1, iterations=15, random_state=seed, num_threads=2)
    als.fit(15 * matrix, show_progress=False)
    return {
        "popularity": lambda users: np.tile(counts, (len(users), 1)),
        "time_decayed_popularity": lambda users: np.tile(recent, (len(users), 1)),
        "item_item_cf": lambda users: matrix[users] @ cosine,
        "als": lambda users: als.user_factors[users] @ als.item_factors.T,
        "ease_r": lambda users: matrix[users] @ ease,
    }


def make_rankings(contexts, scorer, item_count, seen=None, k=100):
    results = np.empty((len(contexts), min(k, item_count)), dtype=np.int32)
    for start in range(0, len(contexts), 256):
        users = np.asarray([row[0] for row in contexts[start:start + 256]])
        scores = scorer(start, users).astype(np.float32)
        if seen is not None:
            rows, items = seen[users].nonzero()
            scores[rows, items] = -np.inf
        order = np.argsort(-scores, kind="stable", axis=1)[:, :k]
        results[start:start + len(order)] = np.where(
            np.take_along_axis(scores, order, axis=1) == -np.inf, -1, order
        )
    return results


def evaluate(rankings, contexts, counts):
    by_user = defaultdict(list)
    ndcgs = []
    hits = np.zeros(2, dtype=np.int64)
    targets = np.zeros(2, dtype=np.int64)
    head = set(np.argsort(-counts, kind="stable")[:max(1, len(counts) // 5)])
    for candidates, (user, _, _, _, truth) in zip(rankings, contexts, strict=True):
        ranked = candidates[:100].tolist()
        metrics = relevance(ranked, dict.fromkeys(truth, 1))
        ndcgs.append(metrics["ndcg@10"])
        by_user[user].append(metrics["recall@100"])
        shown = set(ranked)
        for item in truth:
            bucket = int(item not in head)
            targets[bucket] += 1
            hits[bucket] += item in shown
    recalls = {user: mean(scores) for user, scores in by_user.items()}
    return {
        "recall@100": mean(recalls.values()),
        "ndcg@10": mean(ndcgs),
        "coverage@100": coverage(rankings[:, :100], range(len(counts))),
        "head_recall@100": float(hits[0] / targets[0]) if targets[0] else None,
        "tail_recall@100": float(hits[1] / targets[1]) if targets[1] else None,
        "head_tail_target_counts": targets.tolist(),
    }, recalls


def run(dataset, movie_path=None, seed=42, epochs=10, user_id_ablation=False):
    torch.set_num_threads(2)
    data = synthetic(seed) if dataset == "fari" else movielens(movie_path)
    train, validation, test = requests(data)
    matrix, counts, recent = positive_matrix(data)
    seen = matrix if dataset == "movielens" else None
    val_scores, test_scores, test_recalls = {}, {}, {}
    settings = {"dataset": dataset, "seed": seed, "epochs": epochs, "torch": torch.__version__,
                "cpu": platform.machine(), "torch_threads": torch.get_num_threads(),
                "python": sys.version.split()[0], "train_end_day": data.train_end,
                "validation_end_day": data.val_end,
                "libraries": {package: version(package) for package in ("numpy", "scipy", "implicit", "scikit-learn")},
                "config": {"history_length": 16, "tower_width": 32, "batch_size": 256, "lr": 0.003,
                           "temperature": 0.1, "als_factors": 32, "als_regularization": 0.1,
                           "als_iterations": 15, "ease_l2": 100, "popularity_half_life_days": 14}}

    def measure(name, contexts, scorer, metrics):
        rankings = make_rankings(contexts, scorer, data.catalog.size, seen)
        metrics[name], recalls = evaluate(rankings, contexts, counts)
        if metrics is test_scores:
            test_recalls[name] = recalls

    for name, score in baseline_models(matrix, counts, recent, seed).items():
        def baseline(start, users, model=score):
            return model(users)
        measure(name, validation, baseline, val_scores)
        measure(name, test, baseline, test_scores)

    losses = {}
    variants = [("two_tower", False, True), ("two_tower_logq", True, True)]
    if user_id_ablation:
        variants.append(("two_tower_logq_without_user_id", True, False))
    for name, corrected, use_user_id in variants:
        model, losses[name] = fit_tower(
            data, train, seed=seed, epochs=epochs, correction=corrected, use_user_id=use_user_id
        )
        for split, metrics in ((validation, val_scores), (test, test_scores)):
            queries, items = vectors(model, split)
            def score(start, users):
                return queries[start:start + len(users)] @ items.T
            measure(name, split, score, metrics)

    baseline = test_recalls["popularity"]
    lift = {name: paired_interval(baseline, test_recalls[name], seed=seed)
            for name in ("two_tower", "two_tower_logq")}
    ablation_delta = None
    if user_id_ablation:
        ablation_delta = paired_interval(
            test_recalls["two_tower_logq"], test_recalls["two_tower_logq_without_user_id"], seed=seed
        )
    total_pairs = len(np.unique(data.events[:, 0] * data.catalog.size + data.events[:, 1]))
    return {
        "settings": settings,
        "data": {"events": len(data.events), "outcomes": data.outcomes, "users": data.users,
                 "items": data.catalog.size, "sparsity": 1 - total_pairs / (data.users * data.catalog.size),
                 "requests": dict(zip(("train", "validation", "test"), map(len, (train, validation, test))))},
        "validation": val_scores, "test": test_scores, "test_recall_lift_vs_popularity": lift,
        "user_id_ablation_recall_delta": ablation_delta,
        "training_loss_by_epoch": losses,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("fari", "movielens"), default="fari")
    parser.add_argument("--movielens-zip", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--user-id-ablation", action="store_true")
    args = parser.parse_args()
    if args.dataset == "movielens" and args.movielens_zip is None:
        parser.error("--dataset movielens requires --movielens-zip")
    payload = run(args.dataset, args.movielens_zip, args.seed, args.epochs, args.user_id_ablation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
