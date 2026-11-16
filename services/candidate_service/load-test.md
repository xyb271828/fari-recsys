# Candidate HTTP load test

Locust 2.34.0 and one Uvicorn 0.34.2 worker shared a Linux x86‑64 host
with an AMD EPYC 9V74 CPU. The service loaded a 300-item index and a frozen
user snapshot. Locust used 50 simulated clients, one request per client
per second, staggered startup and a 45-second test duration. Each
request selected a resident and bucket and requested 200 candidates.

| Offered QPS | Achieved QPS | Requests | Failures | p50 | p95 | p99 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 50.0 | 2,250 | 0 | 26 ms | 43 ms | 59 ms |

Achieved QPS divides completed requests by the full configured test
window. Locust reports approximated response percentiles. Measured p99
exceeded the 20 ms objective. The isolated ANN search was below 1 ms p99.
This run does not isolate serialization, loopback transport, server
queueing or client scheduling costs. Instrument the full endpoint and
repeat with the load generator on a separate host before assigning the
latency gap to any single component.

The fallback test removes the index and verifies that the endpoint still
returns the requested number of globally popular candidates for the fixture.
This baseline used a native Python process. The Apple Silicon Docker retest appears below.

## Follow-up latency breakdown

The repeat used the same 300-item snapshot on an AMD EPYC 9V74 host with
Python 3.12.14, Uvicorn 0.34.2 and Locust 2.34.0. A single native Uvicorn
worker and Locust ran on the same host. The Docker 28.5.2 daemon stopped
while creating its Unix socket with `operation not permitted`. The rows below report native process measurements on that host.

| Run | QPS | Requests | Failures | HTTP p50 | HTTP p95 | HTTP p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original native HTTP | 50 | 2,250 | 0 | 26 ms | 43 ms | 59 ms |
| Native repeat with stage timing | 50 | 2,250 | 0 | 20 ms | 30 ms | 33 ms |
| Native repeat without stage timing | 50 | 2,250 | 0 | 20 ms | 33 ms | 46 ms |

All runs exceeded the 20 ms HTTP p99 target. The difference between the
two repeat runs includes run-to-run variability. It does not estimate
the overhead of instrumentation.

The stage breakdown below uses 2,247 HNSW responses. Three popularity
fallback responses are tracked separately in the JSON. Stage percentiles
are computed independently. The tail column gives the median stage time
within the same 23 HNSW requests at or above the HNSW HTTP p99.

| Measurement | p50 | p95 | p99 | Median for the slowest HTTP requests |
| --- | ---: | ---: | ---: | ---: |
| User vector lookup | 0.002 ms | 0.003 ms | 0.009 ms | 0.002 ms |
| FAISS HNSW search | 0.067 ms | 0.146 ms | 1.181 ms | 0.066 ms |
| Candidate ID conversion | 0.008 ms | 0.018 ms | 0.040 ms | 0.009 ms |
| JSON serialization | 0.018 ms | 0.040 ms | 0.085 ms | 0.018 ms |
| Entire ASGI application to response start | 0.179 ms | 0.442 ms | 2.729 ms | 0.173 ms |
| HTTP outside measured ASGI time | 19.856 ms | 29.479 ms | 32.434 ms | 33.770 ms |
| End-to-end HTTP, HNSW only | 20.188 ms | 29.705 ms | 32.612 ms | 33.941 ms |

The HTTP tail was mostly outside the instrumented application. The
outside-ASGI time is computed for each request, using its Locust duration
minus its ASGI duration. It can include time before middleware entry,
server and client scheduling, TCP/protocol work and response transmission.
This experiment cannot assign the residual to any one of those components.

As a separate diagnostic, the readiness endpoint also reached **46 ms p99**
at **50 QPS** with **zero failures**. Its response contains 35 bytes and
does not look up a user, search HNSW, or serialize a candidate list.
This is consistent with the shared HTTP path contributing to the observed
tail. It is a separate run and does not isolate transport from process scheduling.

## Docker Desktop retest on Apple Silicon

On 2026-11-15, Locust 2.34.0 ran on the Mac host and sent requests to the
candidate service in Docker Desktop (Linux ARM64, Docker Engine 29.9.3).
Both runs used the same 300-item Fari snapshot and image
`sha256:8f1488f69c462e1a632e2d2d5f5526c54ab6b9cbcd333965f607d735ce64201d`.
The script configured 50 users, a 50 QPS target, and 45 seconds per run.
Both readiness checks confirmed the HNSW index loaded.

| Run and source | Requests | Failures | HTTP p50 | HTTP p95 | HTTP p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Timing off, final Locust console | 2,200 | 0 | 27 ms | 45 ms | **62 ms** |
| Timing off, archived JSON and CSV | 2,172 | 0 | 27 ms | 42 ms | **59 ms** |
| Timing on, console and archive | 2,250 | 0 | 37 ms | 55 ms | **91 ms** |

The uninstrumented **62 ms** is the final console measurement against
the **20 ms** HTTP p99 target. The control server log also contains 2,200
successful candidate responses. The archived control JSON agrees with the
archived CSV at 2,172 requests and 59 ms p99. The archive contains a
smaller client sample than the final console summary. The evidence does
not establish why. QPS normalized by the configured window is 48.267 in
the archived control summary and 50.0 in the instrumented summary.
The difference between the two runs does not isolate instrumentation
overhead. Host and runtime differences also preclude attributing changes
from the earlier Linux x86-64 benchmark solely to Docker.

### Stage timing

Of the 2,250 profiled requests, 2,245 used HNSW and five used popularity
fallback. The last column reports the median for each measurement within
the same 23 HNSW requests at or above the HNSW HTTP p99.

| Measurement | p50 | p95 | p99 | Median in HTTP p99 tail |
| --- | ---: | ---: | ---: | ---: |
| End-to-end HTTP | 36.599 ms | 55.309 ms | 90.681 ms | 94.280 ms |
| Outside measured ASGI time | 36.181 ms | 54.947 ms | 90.404 ms | 93.796 ms |
| ASGI entry to response start | 0.338 ms | 0.686 ms | 1.321 ms | 0.301 ms |
| User vector lookup | 0.002 ms | 0.010 ms | 0.025 ms | 0.003 ms |
| FAISS search | 0.176 ms | 0.315 ms | 0.482 ms | 0.151 ms |
| Candidate ID conversion | 0.007 ms | 0.033 ms | 0.092 ms | 0.008 ms |
| JSON serialization | 0.024 ms | 0.065 ms | 0.131 ms | 0.023 ms |

The residual outside ASGI dominates the slowest requests. It is computed
per request as client HTTP duration minus ASGI duration. It combines time
before application entry, response transmission, and client-side timing
and scheduling. The measurements leave the largest individual contributor
unresolved. A follow-up could hold the request sequence fixed and compare
the current Locust setup with a paced client on the same host.

[Archived Docker results and per-request timing](../../reports/latency/docker-mac-arm64/)
