import csv
import os
import random
from pathlib import Path

from gevent import sleep
from locust import HttpUser, constant_throughput, events, task

timing_path = os.getenv("FARI_TIMING_CSV")
timing_samples = []
timing_fields = ("lookup", "faiss", "ids", "json", "asgi")


@events.test_start.add_listener
def clear_timings(environment, **kwargs):
    timing_samples.clear()


@events.test_stop.add_listener
def write_timings(environment, **kwargs):
    if timing_path:
        output = Path(timing_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as stream:
            rows = csv.writer(stream)
            rows.writerow(("source", "http_ms", *timing_fields))
            rows.writerows(timing_samples)


class CandidateUser(HttpUser):
    wait_time = constant_throughput(1)

    def on_start(self):
        sleep(random.random())

    @task
    def candidates(self):
        with self.client.get(
            "/candidates",
            params={"user_id": random.randrange(2000), "k": 200, "hour_bucket": random.randrange(4)},
            name="/candidates",
            catch_response=True,
        ) as response:
            if not response.ok:
                return
            payload = response.json()
            if len(payload["item_ids"]) != 200:
                response.failure("Expected 200 candidates")
                return
            if timing_path:
                header = response.headers.get("Server-Timing", "")
                stages = dict(entry.split(";dur=", 1) for entry in header.split(", ") if ";dur=" in entry)
                if "asgi" not in stages:
                    response.failure("Missing Server-Timing application duration")
                    return
                timing_samples.append((
                    payload["source"], response.request_meta["response_time"],
                    *(float(stages[name]) if name in stages else None for name in timing_fields),
                ))
