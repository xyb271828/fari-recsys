"""Summarize Locust totals and matched per-request timing rows."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

STAGES = ("lookup", "faiss", "ids", "json")


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def percentiles(values):
    return {f"p{p}": round(percentile(values, p / 100), 6) for p in (50, 95, 99)}


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(prefix, duration, deployment, image=None):
    totals = next(row for row in read_rows(Path(f"{prefix}_stats.csv")) if row["Name"] == "Aggregated")
    requests = int(totals["Request Count"])
    failures = int(totals["Failure Count"])
    result = {
        "deployment": deployment,
        "image_id": image,
        "duration_seconds": duration,
        "requests": requests,
        "failures": failures,
        "achieved_qps": round(requests / duration, 3),
        "locust_http_ms": {f"p{p}": float(totals[f"{p}%"]) for p in (50, 95, 99)},
    }
    timings = Path(f"{prefix}.timing.csv")
    if not timings.exists():
        return result
    rows = read_rows(timings)
    if len(rows) != requests - failures:
        raise ValueError("Timing sample count differs from Locust successful request count")
    result["timed_requests_by_source"] = dict(Counter(row["source"] for row in rows))
    result["timing_ms_by_source"] = {}
    for source in result["timed_requests_by_source"]:
        samples = []
        for row in rows:
            if row["source"] == source:
                sample = {name: float(value) for name, value in row.items() if name != "source" and value}
                sample["other_asgi"] = sample["asgi"] - sum(sample.get(name, 0) for name in STAGES)
                sample["outside_asgi"] = sample["http_ms"] - sample["asgi"]
                samples.append(sample)
        http_p99 = percentile([sample["http_ms"] for sample in samples], 0.99)
        slow = [sample for sample in samples if sample["http_ms"] >= http_p99]
        result["timing_ms_by_source"][source] = {
            "all_request_percentiles": {
                name: percentiles([sample[name] for sample in samples]) for name in samples[0]
            },
            "http_p99_tail_requests": len(slow),
            "http_p99_tail_median": {
                name: round(percentile([sample[name] for sample in slow], 0.5), 6) for name in slow[0]
            },
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--duration", type=int, default=45)
    parser.add_argument("--deployment", choices=("native", "docker"), required=True)
    parser.add_argument("--image-id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.deployment == "docker" and not args.image_id:
        parser.error("--image-id is required for Docker results")
    summary = summarize(args.prefix, args.duration, args.deployment, args.image_id)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
