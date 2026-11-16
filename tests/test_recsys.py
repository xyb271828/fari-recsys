from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from recsys.data import Catalog, Dataset, requests, synthetic

pytest.importorskip("torch")
pytest.importorskip("implicit")
pytest.importorskip("faiss")

from fastapi.testclient import TestClient  # noqa: E402

from recsys.production import build_feed, mmr  # noqa: E402
from services.candidate_service.app import create_app  # noqa: E402


def test_history_excludes_current_day_and_freezes_at_training_cutoff():
    data = Dataset(
        Catalog(["one", "two", "three"], ["a", "b", "c"]),
        np.array([[0, 0, 74, 0, 1], [0, 1, 75, 1, 1], [0, 2, 76, 1, 1], [0, 2, 83, 0, 1]]),
        1, 75, 82, {},
    )
    train, validation, test = requests(data)
    assert [r[3] for r in train] == [(), (0,)]
    assert validation[0][3] == test[0][3] == (0, 1)


def test_synthetic_scale_and_seed():
    first, second = synthetic(users=100), synthetic(users=100)
    assert np.array_equal(first.events, second.events)
    assert set(first.outcomes) == {"start", "completion", "skip"}
    assert first.events[:, 2].min() == 1
    assert first.events[:, 2].max() == 90


def test_mmr_applies_exclusion_then_diversity():
    ids = np.array([0, 1, 2, 3])
    scores = np.array([1.0, 0.9, 0.85, 0.8])
    similarity = np.eye(4)
    similarity[1, 2] = similarity[2, 1] = 1.0
    assert mmr(ids, scores, similarity, 0.5, {0}, 2) == [1, 3]


def test_missing_index_serves_popularity():
    with TemporaryDirectory() as tmp:
        (Path(tmp) / "popularity.json").write_text("[2, 1, 0]")
        client = TestClient(create_app(tmp))
        assert client.get("/health").json() == {"ready": True, "ann_available": False}
        assert client.get("/candidates?user_id=5&k=2").json() == {
            "source": "popularity", "item_ids": [2, 1],
        }


def test_pinned_item_is_shown_without_ranking_it():
    ids = np.array([0, 1, 2])
    scores = np.array([0.5, 0.9, 0.8])
    assert build_feed(ids, scores, np.eye(3), 0.0, {0}, pinned=[0], k=2) == [0, 1]


def test_index_serves_known_user_and_falls_back_for_unknown(tmp_path):
    import faiss

    index = faiss.IndexHNSWFlat(2, 4, faiss.METRIC_INNER_PRODUCT)
    index.add(np.eye(2, dtype=np.float32))
    faiss.write_index(index, str(tmp_path / "activities.faiss"))
    vectors = np.tile(np.array([[[1.0, 0.0]]], dtype=np.float32), (1, 4, 1))
    np.savez(tmp_path / "users.npz", user_ids=[42], vectors=vectors)
    (tmp_path / "popularity.json").write_text("[1, 0]")
    client = TestClient(create_app(tmp_path))
    assert client.get("/candidates?user_id=42&hour_bucket=1&k=1").json() == {
        "source": "hnsw", "item_ids": [0],
    }
    assert client.get("/candidates?user_id=43&k=1").json() == {
        "source": "popularity", "item_ids": [1],
    }


def test_candidate_timing_reports_fallback_and_app_time(tmp_path, monkeypatch):
    monkeypatch.setenv("FARI_SERVER_TIMING", "1")
    (tmp_path / "popularity.json").write_text("[1, 0]")
    client = TestClient(create_app(tmp_path))
    response = client.get("/candidates?user_id=42&k=1")
    assert response.json() == {"source": "popularity", "item_ids": [1]}
    metrics = {
        name: float(value)
        for name, value in (entry.split(";dur=") for entry in response.headers["server-timing"].split(", "))
    }
    assert metrics.keys() == {"lookup", "ids", "json", "asgi"}
    assert metrics["asgi"] >= sum(metrics[name] for name in ("lookup", "ids", "json"))
    assert "server-timing" not in client.get("/health").headers
    monkeypatch.delenv("FARI_SERVER_TIMING")
    assert "server-timing" not in TestClient(create_app(tmp_path)).get("/candidates?user_id=42").headers
