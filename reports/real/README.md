# Real-data recommendation benchmarks

This directory separates real MovieLens and KuaiRand measurements from the earlier
synthetic activity experiment. Results are offline retrieval measurements, not
online engagement or revenue lift.

## Evaluation contract

All models score the full shared candidate catalog: items with at least one positive
training interaction. No sampled evaluation negatives are used. Binary targets are
grouped by user, calendar day and six-hour bucket. Metrics first average requests
within each user, then average users equally. NDCG uses binary relevance at 10;
Recall uses the top 100. Head/tail recall instead weights positive target occurrences;
head is the top 20% of eligible items by training positive frequency. Coverage is the
fraction of eligible items appearing in any top-100 result.

User history uses at most 16 earlier-day positive events during training and freezes
at the training cutoff for both validation and testing. Models are not refit on
validation. Popularity, sample frequencies, candidate membership and collaborative
features use training only. Training item IDs determine the retrieval catalog;
held-out positives outside it are excluded and counted explicitly in `split_audit`.
Users without a positive training interaction get the identical training-popularity
fallback for every method. Each model reports warm and cold user metrics separately.

Validation user-macro Recall@100 selects parameters and tower checkpoints. Search
spaces are fixed in the runner: time decay 3/14/60 days; EASE L2 10/100/1000;
ALS regularization .01/.1/1 with 64 factors, alpha 15 and 20 iterations; tower
checkpoints 1/3/5/10 epochs with 32-dimensional normalized vectors, temperature .1,
batch size 256 and AdamW learning rate .003. This is a modest, unequal-cost search,
not exhaustive or equal-compute tuning. Both tower variants use the same seed,
architecture, minibatch shuffle and training budget. Deduplicated in-batch negatives
use an inclusion-probability logQ correction; other positives in the same request
are masked out of the negative columns. Unobserved interactions remain unknown.

ALS and both tower variants use initialization seeds 42/43/44; deterministic
baselines run once. Aggregate metrics average seed-specific measurements, not
ensemble predictions. Paired confidence intervals resample users 2,000 times and
apply to the average per-user metric across these three seeds. They do not capture
temporal split uncertainty or the full distribution of training randomness. The
strongest baseline is chosen by validation Recall, independently of test results.
Per-user NPZ files permit rechecking these intervals; their integer user keys are
dataset-local pseudonyms. Raw datasets remain excluded from Git.

## MovieLens-1M

Source: [GroupLens MovieLens-1M](https://grouplens.org/datasets/movielens/1m/).
The official archive has 1,000,209 ratings, 6,040 users and metadata for 3,883 movies.
Ratings of 4 or 5 define positives. Whole-day 80%/90% timestamp quantiles define
train/validation/test, matching the existing adapter. Exclude every training-rated
movie (including low ratings) from recommendations. Movie metadata is treated as
available catalog information. A rating is not a click or an exposure, and missing
ratings are not observed dislikes. This is a frozen global temporal snapshot, with
substantial cold-user and temporal drift effects. It is not comparable numerically
with random leave-one-out benchmark scores.

The older `reports/results/w02_movielens_retrieval.json` used three epochs, masked
only positive training ratings, allowed a different candidate universe and did not
share the same cold-user fallback or NDCG aggregation. Preserve it as a historical
artifact; use this directory for the current protocol.

## KuaiRand-Pure

Source: [KuaiRand project](https://kuairand.com/),
[official archive](https://zenodo.org/records/10439422).
Training is standard-policy April 8–18, 2022; validation is standard-policy April
19–21; test is April 22–May 8. Random-exposure logs from the same test dates are
evaluated separately and never used in training or model selection. Repeat videos
are allowed. The candidate universe remains the training-positive subset of Pure's
restricted video pool for all models and both test policies.

Target is the published `is_click` field: it means click in the two-column UI and
valid playback in the single-column UI. Do not describe it as uniform click CTR.
Only IDs, timestamps and prior positives are used. No user-feature aggregates,
video-feature statistics, future engagement or watch-time labels enter the model.
The unchanged tower gets a constant `video` text token and an `unknown` category
instead of inventing metadata. Its history is only the history retained in Pure;
Pure has incomplete sequences and a retrospectively restricted candidate pool.

The random test measures retrieval of held-out positive interactions within this
pool. It conditions on positive requests and excludes ineligible targets. It is
not an unbiased estimate of CTR, counterfactual policy value, or causal lift.
This first validation targets generalization of retrieval, not off-policy evaluation.

## Reproduction

Use Python 3.12 and the pinned `requirements-real-benchmark.txt` environment. Install
the CPU PyTorch wheel first if GPU wheels are not needed:

```sh
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-real-benchmark.txt
python -m pytest tests/test_real_benchmark.py -q
python -m recsys.download_real --output data
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m recsys.real_benchmark \
  --dataset movielens --source data/ml-1m.zip --output runs/real/movielens
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m recsys.real_benchmark \
  --dataset kuairand --source data/KuaiRand-Pure/data --output runs/real/kuairand
```

The downloader checks the recorded MovieLens SHA-256 and published KuaiRand MD5,
and extracts only the three
required CSV files. The runner records actual input SHA-256 hashes, split counts,
parameter trials, selected checkpoints, epoch losses and package versions. The
GitHub Actions workflow can reproduce both datasets and upload raw result artifacts.
