"""Public benchmark and deterministic synthetic activity data."""

from collections import defaultdict, deque
from dataclasses import dataclass
from zipfile import ZipFile

import numpy as np


@dataclass
class Catalog:
    titles: list[str]
    categories: list[str]

    @property
    def size(self):
        return len(self.titles)


@dataclass
class Dataset:
    catalog: Catalog
    events: np.ndarray  # user ID, item ID, day (one-based), hour bucket, outcome code (0=start, 1=positive, 2=skip)
    users: int
    train_end: int
    val_end: int
    outcomes: dict


def synthetic(seed=42, users=2000, item_count=300):
    rng = np.random.default_rng(seed)
    themes = [
        ("movement", "seated stretch", "light movement", "balance practice"),
        ("music", "piano listening", "folk songs", "rhythm session"),
        ("creative", "watercolor exercise", "collage project", "paper craft"),
        ("puzzles", "word puzzle", "number puzzle", "picture matching"),
        ("social", "group trivia", "story circle", "conversation prompts"),
        ("relaxation", "guided breathing", "nature sounds", "quiet reading"),
    ]
    titles, categories, events = [], [], []
    variants = ["short", "leisurely", "paired", "guided", "independent"]
    for item in range(item_count):
        category, *activities = themes[item % len(themes)]
        index = item // len(themes)
        titles.append(f"{variants[index % len(variants)]} {activities[index % 3]}")
        categories.append(category)
    categories_idx = np.arange(item_count) % len(themes)
    popularity = rng.lognormal(0, 0.55, item_count)
    item_quality = rng.normal(0, 0.06, item_count)
    resident_preferences = rng.dirichlet(np.full(len(themes), 0.7), users)
    outcomes = dict.fromkeys(("start", "completion", "skip"), 0)
    for user, preferences in enumerate(resident_preferences):
        for _ in range(rng.poisson(16)):
            day = int(rng.integers(1, 91))
            hour = int(rng.choice(4, p=[0.32, 0.30, 0.27, 0.11]))
            boost = np.ones(len(themes))
            boost[[0, 2, 4, 5][hour]] = 2.4
            weights = (0.07 + preferences[categories_idx]) * boost[categories_idx] * popularity
            item = int(rng.choice(item_count, p=weights / weights.sum()))
            chance = np.clip(
                0.34 + 0.5 * preferences[categories_idx[item]] + item_quality[item], 0.10, 0.85
            )
            outcome = rng.choice(3, p=[(1 - chance) * 0.45, chance, (1 - chance) * 0.55])
            outcomes[("start", "completion", "skip")[outcome]] += 1
            events.append((user, item, day, hour, int(outcome)))
    return Dataset(Catalog(titles, categories), np.asarray(events, dtype=np.int32), users, 75, 82, outcomes)


def movielens(path):
    with ZipFile(path) as archive:
        movies = [
            row.split("::") for row in archive.read("ml-1m/movies.dat").decode("latin-1").splitlines()
        ]
        ratings = np.asarray([
            tuple(map(int, row.split(b"::"))) for row in archive.read("ml-1m/ratings.dat").splitlines()
        ])
    item_ids = {int(movie[0]): i for i, movie in enumerate(movies)}
    timestamps = ratings[:, 3]
    days = timestamps // 86400 - timestamps.min() // 86400 + 1
    events = np.column_stack((
        ratings[:, 0] - 1,
        [item_ids[item] for item in ratings[:, 1]],
        days,
        timestamps % 86400 // 21600,
        ratings[:, 2] >= 4,
    )).astype(np.int32)
    train_end, val_end = np.quantile(days, [0.80, 0.90], method="nearest")
    return Dataset(
        Catalog([movie[1] for movie in movies], [movie[2].split("|")[0] for movie in movies]),
        events, int(ratings[:, 0].max()), int(train_end), int(val_end),
        {"ratings": len(events), "rating_4_or_5": int(events[:, 4].sum())},
    )


def requests(data, history_size=16):
    """Freeze evaluation histories at train cutoff and group positives by user/day/bucket."""
    histories = defaultdict(lambda: deque(maxlen=history_size))
    contexts = [[], [], []]
    events = data.events[np.argsort(data.events[:, 2], kind="stable")]
    day_groups = np.split(events, np.flatnonzero(np.diff(events[:, 2])) + 1)
    for rows in day_groups:
        day = int(rows[0, 2])
        part = 0 if day <= data.train_end else 1 if day <= data.val_end else 2
        by_request = defaultdict(set)
        for user, item, _, hour, positive in rows:
            if positive == 1:
                by_request[int(user), int(hour)].add(int(item))
        for (user, hour), targets in sorted(by_request.items()):
            contexts[part].append((user, day, hour, tuple(histories[user]), tuple(sorted(targets))))
        if part == 0:
            for user, item, _, hour, positive in rows[np.argsort(rows[:, 3], kind="stable")]:
                if positive == 1:
                    histories[int(user)].append(int(item))
    return contexts
