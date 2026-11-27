# Results comment for resume integration

The following measured results replace the earlier synthetic-data performance claims.

- **Real-data evaluation:** Compared popularity, time-decayed popularity, ALS, EASE and two-tower models with/without logQ correction on **1,000,209 real MovieLens-1M ratings** and **2,622,668 KuaiRand-Pure standard/random-exposure interactions**. Used chronological splits, full training-eligible candidate catalogs, validation-only model selection and three initialization seeds for stochastic models.
- **MovieLens-1M:** EASE increased Recall@100 from **25.73% to 30.68% (19.2% relative)** versus popularity, and NDCG@10 from **0.0743 to 0.0856**. Recall gain: **4.95 percentage points**, paired user-bootstrap 95% interval **[3.92, 5.99]**.
- **KuaiRand-Pure standard-policy test:** EASE increased Recall@100 from **17.76% to 30.77% (73.2% relative)** versus popularity, and NDCG@10 from **0.0212 to 0.0304**. Recall gain: **13.01 percentage points**, paired user-bootstrap 95% interval **[12.68, 13.37]**.
- **logQ ablation:** Across three seeds, correcting the two-tower sampling distribution increased Recall@100 from **13.12% to 24.56%** on MovieLens-1M and **11.38% to 28.83%** on the KuaiRand standard-policy test. The corresponding paired 95% intervals for the gains were **[10.31, 12.61]** and **[17.14, 17.76] percentage points**. Under these tested settings, both corrected towers remained below EASE on their respective standard tests.
- **Random-exposure validation:** On the separate KuaiRand random-exposure test, the corrected tower reached **2.58% Recall@100** versus **2.19%** for the raw tower and **2.47%** for EASE. Its gain versus EASE was **0.11 percentage points**, with a 95% interval of **[-0.01, 0.22]**, so superiority over EASE was not established. NDCG improvement over the raw tower was also not established.

These are offline retrieval measurements. The random-exposure retrieval metric conditions on positive requests within the retained candidate pool; it is not click-through rate, online engagement lift or causal business impact. Existing service-latency results belong to the earlier synthetic candidate snapshot.

The three-seed results are averages of independently trained models, not ensemble predictions. Confidence intervals resample users conditional on the fixed split and these fitted seeds. See [full results](SUMMARY.md) and [protocol](README.md) for details.
