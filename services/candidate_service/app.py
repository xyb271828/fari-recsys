"""Serve a frozen Fari candidate-generation snapshot."""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class CandidateTiming:
    """Record application time up to the ASGI response-start event."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/candidates":
            await self.app(scope, receive, send)
            return
        started = perf_counter_ns()
        stages = {}
        scope.setdefault("state", {})["candidate_timings"] = stages

        async def timed_send(message):
            if message["type"] == "http.response.start":
                stages["asgi"] = perf_counter_ns() - started
                value = ", ".join(f"{name};dur={ns / 1e6:.6f}" for name, ns in stages.items())
                message = {**message, "headers": [*message["headers"], (b"server-timing", value.encode())]}
            await send(message)

        await self.app(scope, receive, timed_send)


def create_app(artifact_dir=None):
    directory = Path(artifact_dir or os.environ.get("FARI_ARTIFACT_DIR", "runs/fari"))
    popularity = json.loads((directory / "popularity.json").read_text())
    index, users = None, {}
    try:
        faiss.omp_set_num_threads(1)
        candidate_index = faiss.read_index(str(directory / "activities.faiss"))
        with np.load(directory / "users.npz", allow_pickle=False) as snapshot:
            vectors = snapshot["vectors"]
            if vectors.shape != (len(snapshot["user_ids"]), 4, candidate_index.d):
                raise ValueError("User snapshot does not match index dimensions")
            users = dict(zip(snapshot["user_ids"].tolist(), vectors, strict=True))
        index = candidate_index
    except (OSError, ValueError, KeyError, RuntimeError):
        logger.warning("ANN artifacts could not load. Serving popularity fallback.", exc_info=True)
    app = FastAPI()
    if os.getenv("FARI_SERVER_TIMING") == "1":
        app.add_middleware(CandidateTiming)

    @app.get("/health")
    def health():
        return {"ready": bool(popularity), "ann_available": index is not None}

    @app.get("/candidates")
    async def candidates(request: Request, user_id: int = Query(ge=0), k: int = Query(200, ge=1, le=300),
                   hour_bucket: int | None = Query(None, ge=0, le=3)):
        if not popularity:
            raise HTTPException(status_code=503, detail="Popularity fallback is empty")
        timings = getattr(request.state, "candidate_timings", None)
        started = perf_counter_ns() if timings is not None else 0
        user_buckets = users.get(user_id) if index is not None else None
        query = None
        if user_buckets is not None:
            bucket = datetime.now(timezone.utc).hour // 6 if hour_bucket is None else hour_bucket
            query = user_buckets[bucket:bucket + 1]
        if timings is not None:
            timings["lookup"] = perf_counter_ns() - started
        if query is None:
            source = "popularity"
            started = perf_counter_ns() if timings is not None else 0
            item_ids = popularity[:k]
        else:
            source = "hnsw"
            started = perf_counter_ns() if timings is not None else 0
            _, ids = index.search(query, k)
            if timings is not None:
                end = perf_counter_ns()
                timings["faiss"] = end - started
                started = end
            item_ids = ids[0][ids[0] >= 0].tolist()
        if timings is not None:
            timings["ids"] = perf_counter_ns() - started
            started = perf_counter_ns()
        response = JSONResponse({"source": source, "item_ids": item_ids})
        if timings is not None:
            timings["json"] = perf_counter_ns() - started
        return response

    return app
