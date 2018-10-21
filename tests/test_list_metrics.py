"""Module for testing accuracy evaluation measures (RMSE, MAE...)"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from math import sqrt

import pytest

from surprise.accuracy import mae, prec10t4_prec5t4_rec10t4_rec5t4_ndcg10_ndcg5_ndcgfull_hits
import numpy as np


def pred(true_r, est, u0=None):
    """Just a small tool to build a prediction with appropriate format."""
    return (u0, None, true_r, est, None)


def test_mae():
    """Tests for the MAE function."""

    predictions = [pred(0, 0), pred(1, 1), pred(2, 2), pred(100, 100)]
    assert mae(predictions) == 0

    predictions = [pred(0, 0), pred(0, 2)]
    assert mae(predictions) == abs(0 - 2) / 2

    predictions = [pred(2, 0), pred(3, 4)]
    assert mae(predictions) == (abs(2 - 0) + abs(3 - 4)) / 2

    with pytest.raises(ValueError):
        mae([])


def test_prec():
    """Tests for the MAE function."""

    predictions = [pred(1, 1)]
    res = prec10t4_prec5t4_rec10t4_rec5t4_ndcg10_ndcg5_ndcgfull_hits(predictions)
    prec10, prec5, rec10, rec5, ndgc10, ndcg5, ndcgfull, hits = res

    # there are no hits so prec is nan
    assert np.isnan(prec10[0])
    assert np.isnan(rec5[0])
    assert hits[0] == 0

    predictions = [pred(1, 1), pred(5, 5), pred(5, 5)]
    res = prec10t4_prec5t4_rec10t4_rec5t4_ndcg10_ndcg5_ndcgfull_hits(predictions)
    prec10, prec5, rec10, rec5, ndgc10, ndcg5, ndcgfull, hits = res

    # precison is 1 and recall is 1
    assert prec10[0] == prec5[0] == rec10[0] == rec5[0] == 1
    assert hits[0] == 2


    # true, est
    predictions = [pred(1, 1), pred(5, 3), pred(5, 5)]
    res = prec10t4_prec5t4_rec10t4_rec5t4_ndcg10_ndcg5_ndcgfull_hits(predictions)
    prec10, prec5, rec10, rec5, ndgc10, ndcg5, ndcgfull, hits = res

    assert prec10[0] == prec5[0] == 1
    assert rec10[0] == rec5[0] == 1

    # make it 12 total
    predictions = [
        # correct negatives
        pred(1, 2), pred(2, 2), pred(3, 2),
        # correct pos
        pred(4, 4), pred(5, 4), pred(5, 4),
        # wrong negative
        pred(1, 5), pred(2, 5), pred(2, 5),
        # wrong positives
        pred(5, 1), pred(4, 1), pred(5, 1),
        
    ]
    res = prec10t4_prec5t4_rec10t4_rec5t4_ndcg10_ndcg5_ndcgfull_hits(predictions)
    prec10, prec5, rec10, rec5, ndgc10, ndcg5, ndcgfull, hits = res
    print(res)

    # there are 6 positives and 3 are correct. But only top 10 are considered!
    assert prec10[0] == 1/2
    # in the top 5 estimated values, there is some randomness
    assert prec5[0] == 2/5
    # in the top 10 estimated values, there should be 6 positives. We only get 4
    assert rec10[0] == 4/6
    # in the top 5, there should be 5 positives. we only get 4
    assert rec5[0] == 2/6
    assert hits[0] == 3

    """
    start new
    """
    predictions = [
        pred(5, 1), pred(5, 1), pred(5, 1),
        pred(5, 1), pred(5, 1), pred(5, 1),
        pred(5, 1), pred(5, 1), pred(5, 1),
        pred(5, 1),
        pred(1,5), pred(1,5), pred(1,5),
        pred(1,5), pred(1,5), pred(1,5),
        pred(1,5), pred(1,5), pred(1,5),
        pred(1,5),
    ]
    res = prec10t4_prec5t4_rec10t4_rec5t4_ndcg10_ndcg5_ndcgfull_hits(predictions)
    prec10, prec5, rec10, rec5, ndgc10, ndcg5, ndcgfull, hits = res
    print(res)
    assert hits[0] == 0

    

if __name__ == '__main__':
    test_prec()
