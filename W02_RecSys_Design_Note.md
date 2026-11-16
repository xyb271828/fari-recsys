# Fari activity recommendations

## Request path

When a resident opens the activity feed, the request provides a synthetic resident ID and
the current six-hour time bucket. A PyTorch user tower combines the resident ID, the mean
embedding of up to 16 prior completions, and the time bucket. The item tower combines
activity ID, category, and a projected title TF–IDF vector. Each tower outputs a normalized
32-dimensional vector. Training uses the dot product and in-batch negatives, correcting
sample inclusion from training-item frequencies in the logQ variant. Inference uses the raw
dot product. Title vocabulary is fit to the catalog metadata assumed available at launch.

Batch export builds a FAISS HNSW index from item vectors and precomputes the four
resident vectors used by the candidate service. FastAPI returns up to 200 item IDs.
Popularity serves unknown users and requests made while the index or user vectors are
unavailable. The index is a single-process CPU snapshot. On the 300-item catalog, compare
HNSW with exact scan before choosing an index for a deployment.

The offline LightGBM ranker takes the ANN candidates and uses tower and ALS scores,
user–category affinity, 7/30-day completion counts, time of day and history length.
The final list applies MMR to balance the learned score and item similarity. The simulator
blocks activities from earlier six-hour buckets within 24 hours. Scheduled reminders have
their own input and appear first without a model score. The synthetic benchmark
contains no reminder events. In production, the rule would use timestamped
impressions and separate reminder IDs supplied by the scheduling system.

## Data and evaluation

| Input | Training | Validation | Test |
| --- | --- | --- | --- |
| Synthetic activities | Days 1–75 | Days 76–82 | Days 83–90 |
| MovieLens‑1M | First ~80% by whole UTC day | Next ~10% | Final ~10% |

Synthetic requests treat completion as positive. A recorded skip or incomplete start
is an observed outcome. Missing observations are unknown. The current tower uses sampled
items for negatives and masks other positives known for the same request. MovieLens
uses ratings of 4 or 5 as positives and masks training positives at retrieval. Its
time bucket corresponds to when a rating was recorded.
All methods get the same evaluation requests within each dataset. Interaction-derived
tower inputs, affinity, popularity and ALS factors are frozen at the training cutoff.
Within training, every history contains only prior days. LightGBM fits on days 76–79,
and days 80–82 select the MMR weight. Random leave-one-out can put future events
from a resident or the catalog into learned features, yielding an optimistic offline
estimate. This experiment groups labels by resident, day and time bucket and uses
a paired resident-level bootstrap for Recall@100 deltas. NDCG@10 is a completion
proxy. Online engagement would need a separate experiment with exposure logs.

## Operations

- Candidate target: 50 requests/second with CPU p99 at or below 20 ms. Track
  achieved QPS, error rate, p50/p95/p99, fallback rate, and snapshot age.
- Release proposal: refresh counts and snapshot daily and retrain weekly once
  real logs exist. Version the model, index, user snapshot, and popularity list
  together. Roll back by restoring the last valid bundle. Test index
  build, exact-overlap parity, response contract and fallback before a rollout.
- Cold start: use popularity for unknown residents. The evaluation compares the
  offline ranker with popularity at 0, 1, 3, and 5 training completions.
  The validation cohorts did not establish a supported switch through five
  completions. Keep the cold-start fallback and gather more data before
  setting a production switch. The candidate route currently switches on snapshot
  availability, so it can be tested independently of an eventual feed policy.
- Catalog items with metadata can be embedded before an interaction is available.
  A launch-time item holdout with the item-ID feature masked would test this for
  Origami. The current results cover user cold start.

In a video feed, replace activities with videos and define a viewing target. In a
delivery feed, encode restaurant/menu metadata and use eligibility filters for
the delivery region before ranking by signals such as ETA and order propensity.
