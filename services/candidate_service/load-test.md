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
The load test used a local Python process. The Docker build and runtime
still need verification on a Docker-enabled host.

## Follow-up latency breakdown

The repeat used the same 300-item snapshot on an AMD EPYC 9V74 host with
Python 3.12.14, Uvicorn 0.34.2 and Locust 2.34.0. A single native Uvicorn
worker and Locust ran on the same host. The Docker 28.5.2 daemon stopped
while creating its Unix socket with `operation not permitted`. The rows
below are local process measurements. Docker image and workload results
are pending a Docker-enabled host.

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
