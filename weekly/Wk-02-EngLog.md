# Week 2 engineering log

I built a recommendation pipeline around synthetic activity interactions. The
generator produced 31,941 outcomes across 2,000 residents and 300 activities,
with 94.94% of resident–activity pairs unobserved. I treated completions as the
training target and grouped evaluation requests by resident, day, and time
bucket. I froze history inputs at the training cutoff for both validation and
testing. A global temporal split keeps later interactions out of the training
features.

I compared global popularity, time-decayed popularity, item–item collaborative
filtering, ALS, EASE-R, and two PyTorch towers on the same test requests. The
logQ variant reached 54.88% Recall@100 against 56.61% for popularity. The paired
95% interval for its delta included zero. Tail recall was 42.46% for the
corrected tower and 26.88% for popularity. Correcting the sampling distribution
increased head recall relative to the uncorrected version and slightly reduced
tail recall. That pattern is consistent with the simulator giving many
residents exposure to a common set of popular activities. An exploratory
ablation removed the user ID and lifted aggregate recall to 56.51% while
reducing tail recall to 38.94%. The paired interval for the ablation
included zero.

I exported the item vectors to FAISS HNSW and served the candidate snapshot
through FastAPI with a popularity fallback. I fitted LightGBM on an early
validation slice, then used the later slice to select an MMR weight of 0.2.
On the held-out test set that setting increased mean category diversity
from 0.525 to 0.595 while reducing NDCG@10 from 0.0549 to 0.0532 compared
with filtering alone. The cold-start evaluation supported keeping the
popularity fallback while collecting more observations.

**Resume draft**

Built and evaluated a PyTorch two-tower recommender with FAISS HNSW candidate
serving and LightGBM ranking on 31.9K synthetic events. Measured 42.5% tail
Recall@100 versus 26.9% for global popularity on a temporal holdout.
