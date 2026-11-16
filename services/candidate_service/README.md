# Candidate service

From the repository root, install the recommender extra and create artifacts.

```sh
python -m pip install -e '.[recsys,dev]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m recsys.production --output runs/fari
FARI_ARTIFACT_DIR=runs/fari uvicorn services.candidate_service.app:create_app --factory --port 8000
```

Request `GET /candidates?user_id=42&k=200&hour_bucket=1`. When the optional
bucket is absent the service uses the current UTC six-hour bucket. The response
identifies whether HNSW or popularity supplied the results.

```sh
locust -f services/candidate_service/locustfile.py --headless -u 50 -r 50 -t 45s \
  --host http://127.0.0.1:8000 --csv runs/fari/load

docker build -f services/candidate_service/Dockerfile -t activity-candidates .
docker run --rm -p 127.0.0.1:8000:8000 \
  -v "$PWD/runs/fari:/artifacts:ro" activity-candidates
```

## Compare Docker HTTP latency with and without stage timing

With Docker Engine running and the host Python environment activated, run:

```sh
bash services/candidate_service/profile_docker.sh runs/fari runs/fari-docker-profile
```

The script builds the candidate image, checks that HNSW is loaded, and runs
two 45-second Locust tests at a target of 50 QPS. It records the image ID,
HTTP percentiles, and per-request timings. Set `FARI_PROFILE_PORT` if port 8768
is occupied. The profiling run enables the `Server-Timing` header with
`FARI_SERVER_TIMING=1`. The control run leaves the header disabled.

The application timer starts at ASGI middleware entry and stops before
sending response headers. The per-request difference between Locust HTTP
time and that timer includes work before ASGI entry, response transmission,
and the client-side HTTP path. See [follow-up benchmark](load-test.md#follow-up-latency-breakdown).
