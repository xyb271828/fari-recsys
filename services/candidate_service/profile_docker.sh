#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_dir"
artifact_dir=$(cd "${1:-runs/fari}" && pwd)
mkdir -p "$repo_dir/runs"
output_dir=${2:-$(mktemp -d "$repo_dir/runs/fari-docker-profile.XXXXXX")}
port=${FARI_PROFILE_PORT:-8768}
image=${FARI_PROFILE_IMAGE:-fari-candidates:profile}

mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd)
for file in activities.faiss users.npz popularity.json; do
    test -r "$artifact_dir/$file" || { echo "Missing snapshot: $artifact_dir/$file" >&2; exit 1; }
done
docker info --format '{{.ServerVersion}}' > "$output_dir/docker-version.txt"
docker build -f services/candidate_service/Dockerfile -t "$image" .
image_id=$(docker image inspect "$image" --format '{{.Id}}')
container_id=
trap 'if [[ -n "$container_id" ]]; then docker rm -f "$container_id" >/dev/null 2>&1 || true; fi' EXIT
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
for mode in control profile; do
    timing=0
    if [[ "$mode" = profile ]]; then timing=1; fi
    prefix="$output_dir/$mode"
    container_id=$(docker run -d -p "127.0.0.1:$port:8000" -e "FARI_SERVER_TIMING=$timing" \
        --mount "type=bind,source=$artifact_dir,target=/artifacts,readonly" "$image_id")
    for attempt in {1..40}; do
        if curl -fsS "http://127.0.0.1:$port/health" > "$prefix.health.json" 2>/dev/null; then break; fi
        sleep 0.5
    done
    python - "$prefix.health.json" <<'PY_HEALTH'
import json
import sys
from pathlib import Path

health = json.loads(Path(sys.argv[1]).read_text())
if health != {"ready": True, "ann_available": True}:
    raise SystemExit(f"Snapshot readiness check failed: {health}")
PY_HEALTH
    unset FARI_TIMING_CSV
    rm -f "$prefix.timing.csv"
    if [[ "$timing" = 1 ]]; then export FARI_TIMING_CSV="$prefix.timing.csv"; fi
    locust -f services/candidate_service/locustfile.py --headless -u 50 -r 50 -t 45s \
        --host "http://127.0.0.1:$port" --csv "$prefix" --only-summary
    python -m services.candidate_service.profile_summary --prefix "$prefix" --deployment docker \
        --image-id "$image_id" --output "$prefix.json"
    docker logs "$container_id" > "$prefix.server.log" 2>&1
    docker rm -f "$container_id" >/dev/null
    container_id=
done
printf 'Results: %s\n' "$output_dir"
