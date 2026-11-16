"""Query-level offline metrics and paired uncertainty."""
import math
import random
from statistics import mean


def percentile(values, p):
    return sorted(values)[max(0, math.ceil(len(values) * p / 100) - 1)]


def dcg(grades, k):
    return sum((2**grade - 1) / math.log2(i + 2) for i, grade in enumerate(grades[:k]))


def ndcg(ranked, judgments, k=10):
    ideal = dcg(sorted(judgments.values(), reverse=True), k)
    actual = dcg([judgments.get(did, 0) for did in ranked[:k]], k)
    return actual / ideal if ideal else 0.0


def relevance(ranked, judgments):
    positive = {did for did, grade in judgments.items() if grade > 0}
    return {
        "ndcg@10": ndcg(ranked, judgments),
        "recall@100": len(set(ranked[:100]) & positive) / len(positive) if positive else 0.0,
        "mrr@10": next((1 / i for i, did in enumerate(ranked[:10], 1) if did in positive), 0.0),
    }


def paired_interval(before, after, seed=42, rounds=2000):
    if not before or before.keys() != after.keys():
        raise ValueError("Paired scores require the same non-empty query set")
    deltas = [after[key] - before[key] for key in sorted(before)]
    rng = random.Random(seed)
    samples = [mean(rng.choices(deltas, k=len(deltas))) for _ in range(rounds)]
    return {
        "delta": mean(deltas),
        "ci95": [percentile(samples, 2.5), percentile(samples, 97.5)],
    }


def ctr_metrics(labels, probabilities, training_ctr, bins=10):
    """Binary classification metrics with training-prevalence normalization."""
    from sklearn.metrics import roc_auc_score

    labels, probabilities = list(labels), list(probabilities)
    if not labels or len(labels) != len(probabilities) or set(labels) - {0, 1}:
        raise ValueError("Supply equally sized, non-empty binary labels and probabilities")
    if not 0 < training_ctr < 1 or bins < 1:
        raise ValueError("training_ctr must be between zero and one and bins must be positive")
    if any(not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Probabilities must be in [0, 1]")
    clipped = [min(1 - 1e-15, max(1e-15, p)) for p in probabilities]
    log_loss = -mean(math.log(p) if label else math.log1p(-p) for label, p in zip(labels, clipped))
    baseline = -mean(
        math.log(training_ctr) if label else math.log1p(-training_ctr) for label in labels
    )
    bucket_rows = [[] for _ in range(bins)]
    for label, probability in zip(labels, probabilities):
        bucket_rows[min(int(probability * bins), bins - 1)].append((label, probability))
    calibrated = sum(abs(sum(label - p for label, p in bucket)) for bucket in bucket_rows) / len(labels)
    return {
        "auc": float(roc_auc_score(labels, probabilities)) if len(set(labels)) == 2 else None,
        "log_loss": log_loss,
        "normalized_entropy": log_loss / baseline,
        "ece": calibrated,
    }


def coverage(rankings, catalog_ids):
    eligible = set(catalog_ids)
    seen = {did for ranking in rankings for did in ranking}
    return len(seen & eligible) / len(eligible) if eligible else 0.0
