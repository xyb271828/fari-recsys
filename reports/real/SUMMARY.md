# Real-data recommendation results

All results below use real public interactions. See [protocol](README.md) for
split rules, eligibility, user aggregation, tuning budgets and limitations.

Stochastic model entries are means across initialization seeds 42/43/44;
they are not ensemble results. Recall and NDCG use equal user weights.

## movielens

Validation-selected strongest baseline: **time_decayed_popularity**.

Training: 803,166 observed interactions; 464,697 positive events; 3,469 eligible items.

### test

| Model | Recall@100 | NDCG@10 | Tail Recall@100 | Coverage@100 |
| --- | ---: | ---: | ---: | ---: |
| popularity | 0.2573 | 0.0743 | 0.0000 | 0.1588 |
| time_decayed_popularity | 0.2919 | 0.0746 | 0.0006 | 0.1513 |
| ease_r | 0.3068 | 0.0856 | 0.0278 | 0.3964 |
| als | 0.2725 | 0.0682 | 0.0659 | 0.5799 |
| two_tower | 0.1312 | 0.0351 | 0.0627 | 1.0000 |
| two_tower_logq | 0.2456 | 0.0694 | 0.0039 | 0.3165 |

| Comparison | Recall delta (percentage points) | User-bootstrap 95% interval |
| --- | ---: | ---: |
| EASE vs popularity | +4.95 | [+3.92, +5.99] |
| logQ tower vs raw tower | +11.45 | [+10.31, +12.61] |
| two_tower vs validation-selected baseline | -16.07 | [-17.45, -14.78] |
| two_tower_logq vs validation-selected baseline | -4.62 | [-5.45, -3.80] |

| Model | Warm-user Recall@100 | Cold-user Recall@100 |
| --- | ---: | ---: |
| popularity | 0.2608 | 0.2388 |
| ease_r | 0.3200 | 0.2388 |
| als | 0.2791 | 0.2388 |
| two_tower | 0.1103 | 0.2388 |
| two_tower_logq | 0.2470 | 0.2388 |

Evaluation: 1,186 users (993 warm / 193 cold), 8,159 eligible positive requests.

### Selected checkpoints

| Seed | Raw tower epoch | logQ tower epoch |
| --- | ---: | ---: |
| 42 | 10 | 1 |
| 43 | 10 | 1 |
| 44 | 10 | 1 |

Recorded wall time: 1449.1 seconds on the local CPU run.
See JSON for every seed, tuning trial, split audit and package version.

## kuairand

Validation-selected strongest baseline: **ease_r**.

Training: 1,079,797 observed interactions; 501,655 positive events; 7,103 eligible items.

### test

| Model | Recall@100 | NDCG@10 | Tail Recall@100 | Coverage@100 |
| --- | ---: | ---: | ---: | ---: |
| popularity | 0.1776 | 0.0212 | 0.0000 | 0.0141 |
| time_decayed_popularity | 0.1855 | 0.0251 | 0.0000 | 0.0165 |
| ease_r | 0.3077 | 0.0304 | 0.0208 | 0.7159 |
| als | 0.3025 | 0.0179 | 0.0472 | 0.3987 |
| two_tower | 0.1138 | 0.0095 | 0.1265 | 0.9998 |
| two_tower_logq | 0.2883 | 0.0287 | 0.0358 | 0.5661 |

| Comparison | Recall delta (percentage points) | User-bootstrap 95% interval |
| --- | ---: | ---: |
| EASE vs popularity | +13.01 | [+12.68, +13.37] |
| logQ tower vs raw tower | +17.44 | [+17.14, +17.76] |
| two_tower vs validation-selected baseline | -19.39 | [-19.79, -19.00] |
| two_tower_logq vs validation-selected baseline | -1.94 | [-2.24, -1.63] |

| Model | Warm-user Recall@100 | Cold-user Recall@100 |
| --- | ---: | ---: |
| popularity | 0.1726 | 0.2582 |
| ease_r | 0.3108 | 0.2582 |
| als | 0.3053 | 0.2582 |
| two_tower | 0.1048 | 0.2582 |
| two_tower_logq | 0.2902 | 0.2582 |

Evaluation: 23,259 users (21,887 warm / 1,372 cold), 106,700 eligible positive requests.

### random_test

| Model | Recall@100 | NDCG@10 | Tail Recall@100 | Coverage@100 |
| --- | ---: | ---: | ---: | ---: |
| popularity | 0.0219 | 0.0014 | 0.0000 | 0.0141 |
| time_decayed_popularity | 0.0227 | 0.0014 | 0.0000 | 0.0165 |
| ease_r | 0.0247 | 0.0012 | 0.0007 | 0.7288 |
| als | 0.0236 | 0.0006 | 0.0015 | 0.4015 |
| two_tower | 0.0219 | 0.0012 | 0.0190 | 0.9998 |
| two_tower_logq | 0.0258 | 0.0012 | 0.0018 | 0.5652 |

| Comparison | Recall delta (percentage points) | User-bootstrap 95% interval |
| --- | ---: | ---: |
| EASE vs popularity | +0.28 | [+0.15, +0.41] |
| logQ tower vs raw tower | +0.39 | [+0.27, +0.51] |
| two_tower vs validation-selected baseline | -0.28 | [-0.43, -0.14] |
| two_tower_logq vs validation-selected baseline | +0.11 | [-0.01, +0.22] |

| Model | Warm-user Recall@100 | Cold-user Recall@100 |
| --- | ---: | ---: |
| popularity | 0.0221 | 0.0200 |
| ease_r | 0.0251 | 0.0200 |
| als | 0.0238 | 0.0200 |
| two_tower | 0.0221 | 0.0200 |
| two_tower_logq | 0.0263 | 0.0200 |

Evaluation: 22,647 users (20,959 warm / 1,688 cold), 127,072 eligible positive requests.

### Selected checkpoints

| Seed | Raw tower epoch | logQ tower epoch |
| --- | ---: | ---: |
| 42 | 10 | 10 |
| 43 | 10 | 10 |
| 44 | 10 | 10 |

Recorded wall time: 1742.2 seconds on the local CPU run.
See JSON for every seed, tuning trial, split audit and package version.

## Interpretation boundaries

- Only claim superiority for the named model, baseline, dataset, metric and split.
- A logQ improvement over the raw tower is not evidence of improvement over ALS/EASE or popularity.
- KuaiRand standard-policy and random-exposure retrieval are separate evaluations.
- Random-test retrieval conditions on positive requests in the retained candidate pool; it is not CTR or causal lift.
- Temporal changes, cold-user rates and candidate restrictions affect absolute values.
- One split per dataset and three initialization seeds do not establish production generalization.
