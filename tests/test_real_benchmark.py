import numpy as np
import pandas as pd
import pytest

from recsys.data import Catalog, Dataset
from recsys.real_benchmark import metrics, paired, rank, stable_topk
from recsys.real_data import prepare
from scipy.sparse import csr_matrix


def test_partial_topk_matches_full_stable_sort_with_boundary_ties():
    rng = np.random.default_rng(12)
    scores = rng.integers(0, 20, (100, 200)).astype(float)
    scores[:, :30] = -np.inf
    scores[0] = -np.inf
    for k in (1, 10, 100, 200):
        np.testing.assert_array_equal(stable_topk(scores, k),
                                      np.argsort(-scores, axis=1, kind='stable')[:, :k])


def test_rank_applies_shared_candidates_seen_filter_and_cold_fallback():
    contexts = [(0, 2, 0, (), (2,)), (1, 2, 0, (), (1,))]
    seen = csr_matrix(([1], ([0], [1])), shape=(2, 4))
    ranked = rank(contexts, lambda start, users: np.tile([100, 9, 8, 7], (len(users), 1)),
                  np.array([False, True, True, True]), seen,
                  np.array([True, False]), np.array([0, 2, 1, 3]))
    assert ranked[0].tolist() == [2, 3, -1]
    assert ranked[1].tolist() == [3, 1, 2]


def test_metrics_equal_weights_users_instead_of_requests():
    contexts = [(0, 2, 0, (), (0,)), (0, 3, 0, (), (0,)), (1, 2, 0, (), (0,))]
    result, per_user = metrics(np.array([[0], [0], [1]]), contexts,
                               np.array([1, 1]), np.ones(2, dtype=bool), np.array([True, False]))
    assert result['recall@100'] == .5
    assert result['ndcg@10'] == .5
    assert result['warm']['recall@100'] == 1
    assert result['cold']['recall@100'] == 0
    assert per_user[0].tolist() == [0, 1]


def test_paired_rejects_different_users_and_recovers_known_difference():
    before = (np.array([1, 2]), np.zeros((2, 2)))
    after = (np.array([1, 2]), np.ones((2, 2)) * .25)
    assert paired(before, after, rounds=20)['recall@100']['ci95'] == [.25, .25]
    with pytest.raises(ValueError):
        paired(before, (np.array([2, 3]), after[1]))


def test_prepare_drops_unretrievable_targets_and_masks_all_training_ratings(monkeypatch):
    data = Dataset(Catalog(['movie a', 'movie b', 'movie c'], ['a']*3),
                   np.array([[0,0,1,0,1], [0,1,1,0,0], [1,1,1,0,1],
                             [0,1,2,0,1], [0,2,2,0,1], [1,0,2,0,1],
                             [1,0,3,0,1]]), 2, 1, 2, {})
    monkeypatch.setattr('recsys.real_data.movielens', lambda path: data)
    monkeypatch.setattr('recsys.real_data.checksum', lambda path: 'test-checksum')
    _,_,splits,_,_,active,_,meta = prepare('movielens', 'unused')
    assert active.tolist() == [True, True, False]
    assert len(splits['validation']) == 1
    assert splits['validation'][0][0] == 1
    assert meta['split_audit']['validation']['excluded_target_occurrences'] == 2


def test_kuairand_keeps_random_test_out_of_training_and_validation(tmp_path):
    columns = ['user_id','video_id','date','hourmin','time_ms','is_click']
    early = [[0,0,20220418,100,1,1],[0,1,20220419,100,2,1]]
    late = [[0,0,20220422,100,3,1]]
    random = [[0,0,20220423,100,4,1],[0,1,20220423,100,5,1]]
    for name, rows in [('log_standard_4_08_to_4_21_pure.csv',early),
                       ('log_standard_4_22_to_5_08_pure.csv',late),
                       ('log_random_4_22_to_5_08_pure.csv',random)]:
        pd.DataFrame(rows,columns=columns).to_csv(tmp_path/name,index=False)
    data,train,splits,_,_,_,seen,meta = prepare('kuairand',tmp_path)
    assert len(train) == 1
    assert seen is None
    assert splits['test'][0][3] == splits['random_test'][0][3] == (0,)
    assert meta['training_positive_events'] == 1
    assert meta['split_audit']['random_test']['excluded_target_occurrences'] == 1
    assert data.outcomes['random_impressions'] == 2
