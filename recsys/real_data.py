"""Adapters and audit manifests for real-data, frozen-snapshot retrieval."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from recsys.data import Catalog, Dataset, movielens, requests


def checksum(path):
    digest = sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kuairand(directory):
    directory = Path(directory)
    columns = ["user_id", "video_id", "date", "hourmin", "time_ms", "is_click"]
    names = ["log_standard_4_08_to_4_21_pure.csv",
             "log_standard_4_22_to_5_08_pure.csv", "log_random_4_22_to_5_08_pure.csv"]
    frames = [pd.read_csv(directory / name, usecols=columns) for name in names]
    # Only IDs and timestamps are used. Future aggregate user/video statistics are excluded.
    users = max(int(frame.user_id.max()) for frame in frames) + 1
    items = max(int(frame.video_id.max()) for frame in frames) + 1

    def events(frame):
        frame = frame.sort_values("time_ms", kind="stable")
        day = (pd.to_datetime(frame.date.astype(str)) - pd.Timestamp("2022-04-08")).dt.days + 1
        return np.column_stack((frame.user_id, frame.video_id, day,
                                frame.hourmin // 600, frame.is_click)).astype(np.int32)

    standard = pd.concat(frames[:2], ignore_index=True)
    data = Dataset(Catalog(["video" for _ in range(items)], ["unknown"] * items),
                   events(standard), users, 11, 14,
                   {"standard_impressions": len(standard), "random_impressions": len(frames[2])})
    random_events = events(frames[2])
    random_data = replace(data, events=np.vstack((data.events[data.events[:, 2] <= 11], random_events)))
    random_requests = requests(random_data)[2]
    manifest = {name: {"rows": len(frame), "sha256": checksum(directory / name)}
                for name, frame in zip(names, frames, strict=True)}
    return data, random_requests, manifest


def prepare(dataset, source):
    if dataset == "movielens":
        data = movielens(source)
        random_requests = []
        manifest = {"ml-1m.zip": {"sha256": checksum(source)}}
    else:
        data, random_requests, manifest = kuairand(source)
    train, validation, test = requests(data)
    training = data.events[data.events[:, 2] <= data.train_end]
    positive = training[training[:, 4] == 1]
    matrix = csr_matrix((np.ones(len(positive), dtype=np.float32),
                         (positive[:, 0], positive[:, 1])), shape=(data.users, data.catalog.size))
    matrix.data[:] = 1  # Binary preference for ALS/EASE; popularity still counts events.
    counts = np.bincount(positive[:, 1], minlength=data.catalog.size)
    active = counts > 0  # Same training-positive candidate catalog for all methods.
    seen = None
    if dataset == "movielens":
        seen = csr_matrix((np.ones(len(training)), (training[:, 0], training[:, 1])),
                          shape=matrix.shape)

    def eligible(rows):
        result, excluded = [], 0
        for user, day, hour, history, targets in rows:
            keep = tuple(item for item in targets
                         if active[item] and (seen is None or seen[user, item] == 0))
            excluded += len(targets) - len(keep)
            if keep:
                result.append((user, day, hour, history, keep))
        return result, {"raw_requests": len(rows), "eligible_requests": len(result),
                        "excluded_target_occurrences": excluded}

    splits, audit = {}, {}
    for name, rows in (("validation", validation), ("test", test), ("random_test", random_requests)):
        if rows:
            splits[name], audit[name] = eligible(rows)
    meta = {"dataset": dataset, "events": len(data.events), "outcomes": data.outcomes,
            "users": data.users, "catalog_items": data.catalog.size,
            "training_positive_items": int(active.sum()), "training_impressions": len(training),
            "training_positive_events": len(positive), "training_requests": len(train),
            "train_end_day": data.train_end, "validation_end_day": data.val_end,
            "split_audit": audit, "inputs": manifest,
            "repeat_policy": "exclude all training-rated items" if seen is not None else "allow repeat videos",
            "target": "rating >= 4" if dataset == "movielens" else "is_click (click or valid_play, UI dependent)"}
    return data, train, splits, matrix, counts, active, seen, meta
