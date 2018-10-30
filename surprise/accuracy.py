"""
The :mod:`surprise.accuracy` module provides with tools for computing accuracy
metrics on a set of predictions.

Available accuracy metrics:

.. autosummary::
    :nosignatures:

    rmse
    mae
    fcp
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict
import numpy as np
from six import iteritems


def rmse(predictions, verbose=True):
    """Compute RMSE (Root Mean Squared Error).

    .. math::
        \\text{RMSE} = \\sqrt{\\frac{1}{|\\hat{R}|} \\sum_{\\hat{r}_{ui} \in
        \\hat{R}}(r_{ui} - \\hat{r}_{ui})^2}.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Root Mean Squared Error of predictions.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    mse = np.mean([float((true_r - est)**2)
                   for (_, _, true_r, est, _) in predictions])
    rmse_ = np.sqrt(mse)

    if verbose:
        print('RMSE: {0:1.4f}'.format(rmse_))

    return rmse_


def mae(predictions, verbose=True):
    """Compute MAE (Mean Absolute Error).

    .. math::
        \\text{MAE} = \\frac{1}{|\\hat{R}|} \\sum_{\\hat{r}_{ui} \in
        \\hat{R}}|r_{ui} - \\hat{r}_{ui}|

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Mean Absolute Error of predictions.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    mae_ = np.mean([float(abs(true_r - est))
                    for (_, _, true_r, est, _) in predictions])

    if verbose:
        print('MAE:  {0:1.4f}'.format(mae_))

    return mae_


def fcp(predictions, verbose=True):
    """Compute FCP (Fraction of Concordant Pairs).

    Computed as described in paper `Collaborative Filtering on Ordinal User
    Feedback <http://www.ijcai.org/Proceedings/13/Papers/449.pdf>`_ by Koren
    and Sill, section 5.2.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Fraction of Concordant Pairs.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    predictions_u = defaultdict(list)
    nc_u = defaultdict(int)
    nd_u = defaultdict(int)

    for u0, _, r0, est, _ in predictions:
        predictions_u[u0].append((r0, est))

    for u0, preds in iteritems(predictions_u):
        for r0i, esti in preds:
            for r0j, estj in preds:
                if esti > estj and r0i > r0j:
                    nc_u[u0] += 1
                if esti >= estj and r0i < r0j:
                    nd_u[u0] += 1

    nc = np.mean(list(nc_u.values())) if nc_u else 0
    nd = np.mean(list(nd_u.values())) if nd_u else 0

    try:
        fcp_ = nc / (nc + nd)
    except ZeroDivisionError:
        raise ValueError('cannot compute fcp on this list of prediction. ' +
                         'Does every user have at least two predictions?')

    if verbose:
        print('FCP:  {0:1.4f}'.format(fcp_))

    return fcp_


def dcg_at_k(ratings):
    """
    Discounted cumulative gain at k
    https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    Using formula from this MSR IR paper:
    https://dl.acm.org/citation.cfm?doid=1102351.1102363

    k is assumed to be the length of the input list
    args:
        ratings: a list of relevance scores, e.g. explicit ratings 1-5
    returns:
        a dcg_at_k value
    """
    k = len(ratings)

    return sum([
        (2 ** rating - 1) / 
        (np.math.log(i + 1, 2))
        for rating, i in zip(ratings, range(1, k+1))
    ])



# this pattern is getting quite unwieldy
# originally used it so I could match the existing API for accuracy functions
# and I didn't want to to do the list sorting more than one time.
# but this has gotten a bit crazy
#def prec10t4_prec5t4_rec10t4_rec5t4_ndcg10_ndcg5_ndcgfull_hits_normhits(predictions, verbose=True, head_items=None):
def list_metrics(predictions, verbose=True, head_items=None):
    """
    Return precision and recall at k metrics for each user.
    Also returns ndcg_at_k.
    https://en.wikipedia.org/wiki/Discounted_cumulative_gain

    """
    threshold = 4
    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        if head_items:
            if iid in head_items:
                continue
        user_est_true[uid].append((est, true_r))

    prec10, prec5, rec10, rec5 = {}, {}, {}, {}
    ndcg10, ndcg5, ndcgfull = {}, {}, {}
    n_hits, normhits = {}, {}
    avg_rating, avg_est, n_false_pos = {}, {}, {}
    total_hits = {}
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        np.random.seed(0)
        np.random.shuffle(user_ratings)
        user_ratings_sorted_by_est = sorted(user_ratings, key=lambda x: x[0], reverse=True)
        
        # also need to sort by true value for Ideal DCG
        user_ratings_sorted_by_true = sorted(user_ratings, key=lambda x: x[1], reverse=True)

        num_ratings = len(user_ratings)
        for k_for_ndcg, outdict in (
            (10, ndcg10),
            (5, ndcg5),
            (num_ratings, ndcgfull),
        ):
            if num_ratings >= k_for_ndcg:
                true_ratings_of_first_k_true = [x[1] for x in user_ratings_sorted_by_true[:k_for_ndcg]]
                true_ratings_of_first_k_est = [x[1] for x in  user_ratings_sorted_by_est[:k_for_ndcg]]

                ideal_dcg = dcg_at_k(true_ratings_of_first_k_true)
                pred_dcg = dcg_at_k(true_ratings_of_first_k_est)
                norm_dcg = pred_dcg / ideal_dcg
                outdict[uid] = norm_dcg
        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        for k_for_precrec, precdic, recdic in (
            (10, prec10, rec10),
            (5, prec5, rec5)
        ):
            # Number of recommended items in top k
            top_k = user_ratings_sorted_by_est[:k_for_precrec]
            n_rel_k = sum((true_r >= threshold) for (_, true_r) in top_k)

            prec_threshold = min(k_for_precrec, n_rel)
            top_threshold = user_ratings_sorted_by_est[:prec_threshold]
            n_rel_threshold = sum((true_r >= threshold) for (_, true_r) in top_threshold)

            if n_rel:
                precdic[uid] = n_rel_threshold / prec_threshold
            if n_rel:
                recdic[uid] = n_rel_k / n_rel

        # calculate the full_hits
        items_with_true_over_t = [(est, true_r) for (est, true_r) in user_ratings_sorted_by_est if true_r >= threshold]
        #possible_hits = sum((true_r >= threshold) for (_, true_r) in user_ratings_sorted_by_est)
        possible_hits = len(items_with_true_over_t)
        top_k = user_ratings_sorted_by_est[:possible_hits]

        n_hits[uid] = sum((true_r >= threshold) for (_, true_r) in top_k)
        n_false_pos[uid] = sum((true_r < threshold) for (_, true_r) in top_k)
        avg_rating[uid] = np.mean([true_r for (_, true_r) in user_ratings_sorted_by_est])
        avg_est[uid] = np.mean([est for (est, _) in user_ratings_sorted_by_est])
        total_hits[uid] = n_hits[uid]
        if possible_hits:
            normhits[uid] = n_hits[uid] / possible_hits

    if verbose:
        pass
    
    n_users = len(user_est_true)
    dicts_and_names = [
        (prec10, 'prec10t{}'.format(threshold)),
        (prec5, 'prec5t{}'.format(threshold)),
        (rec10, 'rec10t{}'.format(threshold)),
        (rec5, 'rec5t{}'.format(threshold)),
        (ndcg10, 'ndcg10'),
        (ndcg5, 'ndcg5'),
        (ndcgfull, 'ndcgfull'),
        (n_hits, 'hits'),
        (normhits, 'normhits'),
        (avg_rating, 'avgrating'),
        (avg_est, 'avgest'),
        (n_false_pos, 'falsepos'),
        (total_hits, 'totalhits')
    ]
    if n_users:
        ret = (
            (dic.values(), len(dic.values()) / n_users, name) for dic, name in dicts_and_names
        )

        # ret = (
        #     (prec10.values(), len(prec10.values()) / n_users, 'prec10t{}'.format(threshold)),
        #     (prec5.values(), len(prec5.values()) / n_users, 'prec5t{}'.format(threshold)),
        #     (rec10.values(), len(rec10.values()) / n_users, 'rec10t{}'.format(threshold)),
        #     (rec5.values(), len(rec5.values()) / n_users, 'rec5t{}'.format(threshold)),
        #     (ndcg10.values(), len(ndcg10.values()) / n_users, 'ndcg10'),
        #     (ndcg5.values(), len(ndcg5.values()) / n_users, 'ndcg5'),
        #     (ndcgfull.values(), len(ndcgfull.values()) / n_users, 'ndcgfull'),
        #     (n_hits.values(), len(n_hits.values()) / n_users, 'hits'),
        #     (normhits.values(), len(normhits.values()) / n_users, 'normhits'),
        # )
    else:
        ret = (
            ([], float('nan'), name) for _, name in dicts_and_names
        )

    prepped = {}
    for (vals, frac, name) in ret:
        if 'total' not in name:
            prepped[name] =  (np.mean(list(vals)), frac)
        else:
            prepped[name] = (np.sum(list(vals)), frac)
    # ret = {
    #     name: (np.mean(list(vals)), frac) for (vals, frac, name) in ret
    # }
    return prepped

