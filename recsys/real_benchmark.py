"""Validation-selected public-data retrieval with user-paired uncertainty.

Run with OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m recsys.real_benchmark.
"""
import argparse
import copy
import json
import platform
import time
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from implicit.als import AlternatingLeastSquares
from scipy.linalg import cho_factor, cho_solve
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from recsys.real_data import prepare
from recsys.towers import TwoTower, tensor_contexts, vectors


def stable_topk(scores, k):
    """Exact top-k with item-ID tie breaking, without sorting the whole catalog."""
    order = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    selected = np.take_along_axis(scores, order, axis=1)
    threshold = selected.min(axis=1)
    # Repair ties crossing the partition boundary; pick the smallest item IDs.
    for row in np.flatnonzero((scores == threshold[:, None]).sum(axis=1) > 1):
        above = np.flatnonzero(scores[row] > threshold[row])
        tied = np.flatnonzero(scores[row] == threshold[row])
        order[row] = np.concatenate((above, tied[:k-len(above)]))
    sort = np.lexsort((order, -np.take_along_axis(scores, order, axis=1)), axis=1)
    return np.take_along_axis(order, sort, axis=1)


def rank(contexts, scorer, active, seen, warm, counts):
    """Full eligible catalog scoring, stable ties, shared cold-user fallback."""
    rankings = np.empty((len(contexts), min(100, int(active.sum()))), dtype=np.int32)
    for start in range(0, len(contexts), 256):
        users = np.array([row[0] for row in contexts[start:start + 256]])
        scores = np.asarray(scorer(start, users), dtype=np.float32)
        scores[~warm[users]] = counts
        scores[:, ~active] = -np.inf
        if seen is not None:
            rows, items = seen[users].nonzero()
            scores[rows, items] = -np.inf
        order = stable_topk(scores, rankings.shape[1])
        rankings[start:start + len(users)] = np.where(
            np.isfinite(np.take_along_axis(scores, order, axis=1)), order, -1)
    return rankings


def metrics(rankings, contexts, counts, active, warm):
    """User macro averages; tail recall is a target-occurrence micro average."""
    per_user = defaultdict(list)
    head_ids = np.flatnonzero(active)[np.argsort(-counts[active], kind="stable")[:max(1, int(active.sum()) // 5)]]
    head = set(head_ids.tolist())
    hits, targets = np.zeros(2), np.zeros(2)
    discounts = 1 / np.log2(np.arange(10) + 2)
    for ranked, row in zip(rankings, contexts, strict=True):
        truth = set(row[4])
        found = set(ranked) & truth
        recall = len(found) / len(truth)
        ndcg = sum(discounts[j] for j, item in enumerate(ranked[:10]) if item in truth)
        ndcg /= discounts[:min(10, len(truth))].sum()
        per_user[row[0]].append((recall, ndcg))
        for item in truth:
            bucket = int(item not in head)
            hits[bucket] += item in found
            targets[bucket] += 1
    users = np.array(sorted(per_user))
    values = np.array([np.mean(per_user[u], axis=0) for u in users])

    def cohort(mask):
        return {"users": int(mask.sum()),
                "recall@100": float(values[mask, 0].mean()) if mask.any() else None,
                "ndcg@10": float(values[mask, 1].mean()) if mask.any() else None}

    summary = {**cohort(np.ones(len(users), dtype=bool)), "requests": len(contexts),
               "warm": cohort(warm[users]), "cold": cohort(~warm[users]),
               "head_recall@100": float(hits[0] / targets[0]) if targets[0] else None,
               "tail_recall@100": float(hits[1] / targets[1]) if targets[1] else None,
               "head_tail_target_counts": targets.astype(int).tolist(),
               "coverage@100": len(set(rankings.ravel()) & set(np.flatnonzero(active))) / int(active.sum())}
    return summary, (users, values)


def paired(before, after, seed=2026, rounds=2000):
    if not np.array_equal(before[0], after[0]):
        raise ValueError("Paired evaluation requires identical users")
    delta = after[1] - before[1]
    rng = np.random.default_rng(seed)
    draws = np.array([delta[rng.integers(len(delta), size=len(delta))].mean(axis=0)
                      for _ in range(rounds)])
    return {name: {"delta": float(delta[:, i].mean()),
                   "ci95": np.quantile(draws[:, i], [.025, .975]).tolist()}
            for i, name in enumerate(("recall@100", "ndcg@10"))}


def train_tower(data, train, seed, correction, checkpoints, evaluate_validation):
    torch.manual_seed(seed)
    expanded = [(*row[:4], target, i) for i, row in enumerate(train) for target in row[4]]
    target = torch.tensor([row[4] for row in expanded], dtype=torch.long)
    frequency = torch.bincount(target, minlength=data.catalog.size)
    q = (frequency.double() + 1) / (len(target) + data.catalog.size)
    model = TwoTower(data)
    known = np.zeros(data.users, dtype=bool)
    known[[row[0] for row in train]] = True
    with torch.no_grad():
        model.user_id.weight[1:][~known] = 0
        model.item_id.weight[1:][frequency == 0] = 0
    loader = DataLoader(TensorDataset(*tensor_contexts(expanded), target,
                                     torch.tensor([row[5] for row in expanded])),
                        batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    best, best_score, selected_epoch = None, -np.inf, None
    trace = []
    for epoch in range(1, max(checkpoints) + 1):
        model.train()
        total = 0.
        for user, hour, history, items, request_ids in loader:
            unique, labels = items.unique(return_inverse=True)
            scores = model.encode_users(user, hour, history) @ model.encode_items(unique).T / .1
            if correction:
                inclusion = -torch.expm1((len(items) - 1) * torch.log1p(-q[unique]))
                scores = scores - inclusion.clamp_min(1e-12).log().float()
            columns = {item: col for col, item in enumerate(unique.tolist())}
            co_positive = [(j, columns[item])
                           for j, (rid, target_item) in enumerate(zip(request_ids.tolist(), items.tolist(), strict=True))
                           for item in train[rid][4] if item != target_item and item in columns]
            if co_positive:
                rows, cols = zip(*co_positive)
                scores[rows, cols] = -torch.inf
            loss = F.cross_entropy(scores, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(items)
        entry = {"epoch": epoch, "loss": total / len(target)}
        if epoch in checkpoints:
            model.eval()
            score = evaluate_validation(model)
            entry["validation_recall@100"] = score
            if score > best_score:
                best, best_score, selected_epoch = copy.deepcopy(model.state_dict()), score, epoch
        trace.append(entry)
        print(json.dumps({"tower_seed": seed, "logq": correction, **entry}), flush=True)
    model.load_state_dict(best)
    return model.eval(), {"selected_epoch": selected_epoch, "validation_recall@100": best_score,
                          "trace": trace}


def run(args):
    started = time.time()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data, train, splits, matrix, counts, active, seen, audit = prepare(args.dataset, args.source)
    warm = np.asarray(matrix.getnnz(axis=1) > 0)
    validation = splits["validation"]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"data": audit}), flush=True)
    report = {"data": audit, "selection_metric": "validation user-macro Recall@100",
              "seeds": args.seeds, "models": {}, "environment": {
                  "python": platform.python_version(), "platform": platform.platform(),
                  "cpu": platform.processor(), "torch_threads": 2,
                  "packages": {p: version(p) for p in ("numpy", "scipy", "pandas", "torch", "implicit", "scikit-learn")}}}
    user_scores = defaultdict(dict)

    def score_contexts(contexts, scorer):
        return metrics(rank(contexts, scorer, active, seen, warm, counts), contexts, counts, active, warm)

    def save(name, scorer, selection):
        result = {"selection": selection, "evaluation": {}}
        for split, contexts in splits.items():
            result["evaluation"][split], user_scores[split][name] = score_contexts(contexts, scorer)
        report["models"][name] = result
        print(json.dumps({"model": name, **result}), flush=True)
        (out / "progress.json").write_text(json.dumps(report, indent=2) + "\n")

    save("popularity", lambda start, users: np.tile(counts, (len(users), 1)), {})
    training_positive = data.events[(data.events[:, 2] <= data.train_end) & (data.events[:, 4] == 1)]
    decay_options = []
    for half_life in (3, 14, 60):
        recent = np.bincount(training_positive[:, 1],
                             np.exp2((training_positive[:, 2] - data.train_end) / half_life), minlength=len(counts))
        def scorer(start, users, rec=recent):
            return np.tile(rec, (len(users), 1))
        value = score_contexts(validation, scorer)[0]["recall@100"]
        decay_options.append((value, half_life, recent))
    value, half_life, recent = max(decay_options, key=lambda row: row[0])
    save("time_decayed_popularity", lambda start, users: np.tile(recent, (len(users), 1)),
         {"half_life_days": half_life, "trials": [{"half_life_days": h, "validation_recall@100": v} for v,h,_ in decay_options]})

    active_ids = np.flatnonzero(active)
    compact = matrix[:, active_ids]
    gram = (compact.T @ compact).toarray().astype(np.float64)
    best, options = None, []
    for l2 in (10, 100, 1000):
        regularized = gram.copy()
        regularized.flat[::len(gram) + 1] += l2
        inverse = cho_solve(cho_factor(regularized), np.eye(len(gram)))
        weights = (-inverse / np.diag(inverse)).astype(np.float32)
        np.fill_diagonal(weights, 0)

        def ease_score(start, users, weights=weights):
            scores = np.zeros((len(users), len(counts)), dtype=np.float32)
            scores[:, active_ids] = compact[users] @ weights
            return scores

        value = score_contexts(validation, ease_score)[0]["recall@100"]
        options.append({"l2": l2, "validation_recall@100": value})
        if best is None or value > best[0]:
            best = (value, l2, weights)
    weights = best[2]
    save("ease_r", lambda start, users, w=weights: ease_score(start, users, w), {"l2": best[1], "trials": options})
    del gram, inverse, regularized, weights, best

    for seed in args.seeds:
        best, trials = None, []
        for regularization in (.01, .1, 1.):
            als = AlternatingLeastSquares(factors=64, regularization=regularization, iterations=20,
                                          random_state=seed, num_threads=2)
            als.fit(15 * matrix, show_progress=False)
            def score(start, users, model=als):
                return model.user_factors[users] @ model.item_factors.T
            value = score_contexts(validation, score)[0]["recall@100"]
            trials.append({"regularization": regularization, "validation_recall@100": value})
            if best is None or value > best[0]:
                best = (value, regularization, als)
        als = best[2]
        save(f"als_seed{seed}", lambda start, users: als.user_factors[users] @ als.item_factors.T,
             {"regularization": best[1], "factors": 64, "iterations": 20, "alpha": 15, "trials": trials})

        for corrected in (False, True):
            name = f"two_tower{'_logq' if corrected else ''}_seed{seed}"

            def tower_score(model, contexts):
                queries, items = vectors(model, contexts)
                return lambda start, users: queries[start:start + len(users)] @ items.T

            model, selection = train_tower(data, train, seed, corrected, args.checkpoints,
                lambda model: score_contexts(validation, tower_score(model, validation))[0]["recall@100"])
            # Scores use separate context vectors for each split, with one frozen model.
            result = {"selection": selection, "evaluation": {}}
            for split, contexts in splits.items():
                result["evaluation"][split], user_scores[split][name] = score_contexts(contexts, tower_score(model, contexts))
            report["models"][name] = result
            print(json.dumps({"model": name, **result}), flush=True)
            (out / "progress.json").write_text(json.dumps(report, indent=2) + "\n")

    # Average metrics across initialization seeds, not ensemble predictions.
    for family in ("als", "two_tower", "two_tower_logq"):
        report["models"][family] = {"evaluation": {}}
        for split in splits:
            samples = [user_scores[split][f"{family}_seed{seed}"] for seed in args.seeds]
            assert all(np.array_equal(samples[0][0], sample[0]) for sample in samples)
            averaged = np.mean([sample[1] for sample in samples], axis=0)
            user_scores[split][family] = (samples[0][0], averaged)
            summaries = [report["models"][f"{family}_seed{s}"]["evaluation"][split] for s in args.seeds]
            report["models"][family]["evaluation"][split] = {
                key: {"mean": float(np.mean([r[key] for r in summaries])),
                      "seed_std": float(np.std([r[key] for r in summaries], ddof=1)) if len(summaries)>1 else 0}
                for key in ("recall@100", "ndcg@10", "head_recall@100", "tail_recall@100", "coverage@100")
                if summaries[0][key] is not None}
    baseline_names = ("popularity", "time_decayed_popularity", "ease_r", "als")
    def validation_score(name):
        return float(user_scores["validation"][name][1][:,0].mean())
    strongest = max(baseline_names, key=validation_score)
    report["strongest_baseline_selected_on_validation"] = strongest
    report["paired_user_bootstrap"] = {
        split: {name: {"vs_popularity": paired(scores["popularity"], scores[name]),
                       "vs_validation_selected_baseline": paired(scores[strongest], scores[name])}
                for name in ("als", "ease_r", "two_tower", "two_tower_logq")}
        for split, scores in user_scores.items() if split != "validation"}
    report["logq_ablation_paired"] = {split: paired(scores["two_tower"], scores["two_tower_logq"])
                                      for split, scores in user_scores.items() if split != "validation"}
    for split, scores in user_scores.items():
        np.savez_compressed(out / f"{split}_per_user.npz",
                            **{key + "_users": value[0] for key,value in scores.items()},
                            **{key + "_metrics": value[1] for key,value in scores.items()})
    report["wall_seconds"] = time.time() - started
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Completed {args.dataset}: {out / 'results.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("movielens", "kuairand"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[1, 3, 5, 10])
    run(parser.parse_args())


if __name__ == "__main__":
    main()
