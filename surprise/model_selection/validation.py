'''The validation module contains the cross_validate function, inspired from
the mighty scikit learn.'''

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import time

import numpy as np
from joblib import Parallel
from joblib import delayed
from six import iteritems

from .split import get_cv
from .. import accuracy


def cross_validate(algo, data, measures=['rmse', 'mae'], cv=None,
                   return_train_measures=False, n_jobs=-1,
                   pre_dispatch='2*n_jobs', verbose=False):
    '''
    Run a cross validation procedure for a given algorithm, reporting accuracy
    measures and computation times.

    See an example in the :ref:`User Guide <cross_validate_example>`.

    Args: 
        algo(:obj:`AlgoBase \
            <surprise.prediction_algorithms.algo_base.AlgoBase>`):
            The algorithm to evaluate.
        data(:obj:`Dataset <surprise.dataset.Dataset>`): The dataset on which
            to evaluate the algorithm.
        measures(list of string): The performance measures to compute. Allowed
            names are function names as defined in the :mod:`accuracy
            <surprise.accuracy>` module. Default is ``['rmse', 'mae']``.
        cv(cross-validation iterator, int or ``None``): Determines how the
            ``data`` parameter will be split (i.e. how trainsets and testsets
            will be defined). If an int is passed, :class:`KFold
            <surprise.model_selection.split.KFold>` is used with the
            appropriate ``n_splits`` parameter. If ``None``, :class:`KFold
            <surprise.model_selection.split.KFold>` is used with
            ``n_splits=5``.
        return_train_measures(bool): Whether to compute performance measures on
            the trainsets. Default is ``False``.
        n_jobs(int): The maximum number of folds evaluated in parallel.

            - If ``-1``, all CPUs are used.
            - If ``1`` is given, no parallel computing code is used at all,\
                which is useful for debugging.
            - For ``n_jobs`` below ``-1``, ``(n_cpus + n_jobs + 1)`` are\
                used.  For example, with ``n_jobs = -2`` all CPUs but one are\
                used.

            Default is ``-1``.
        pre_dispatch(int or string): Controls the number of jobs that get
            dispatched during parallel execution. Reducing this number can be
            useful to avoid an explosion of memory consumption when more jobs
            get dispatched than CPUs can process. This parameter can be:

            - ``None``, in which case all the jobs are immediately created\
                and spawned. Use this for lightweight and fast-running\
                jobs, to avoid delays due to on-demand spawning of the\
                jobs.
            - An int, giving the exact number of total jobs that are\
                spawned.
            - A string, giving an expression as a function of ``n_jobs``,\
                as in ``'2*n_jobs'``.

            Default is ``'2*n_jobs'``.
        verbose(int): If ``True`` accuracy measures for each split are printed,
            as well as train and test times. Averages and standard deviations
            over all splits are also reported. Default is ``False``: nothing is
            printed.

    Returns:
        dict: A dict with the following keys:

            - ``'test_*'`` where ``*`` corresponds to a lower-case accuracy
              measure, e.g. ``'test_rmse'``: numpy array with accuracy values
              for each testset.

            - ``'train_*'`` where ``*`` corresponds to a lower-case accuracy
              measure, e.g. ``'train_rmse'``: numpy array with accuracy values
              for each trainset. Only available if ``return_train_measures`` is
              ``True``.

            - ``'fit_time'``: numpy array with the training time in seconds for
              each split.

            - ``'test_time'``: numpy array with the testing time in seconds for
              each split.

    '''
    measures = [m.lower() for m in measures]

    cv = get_cv(cv)

    delayed_list = (delayed(fit_and_score)(algo, trainset, testset, measures,
                                           return_train_measures)
                    for (trainset, testset) in cv.split(data))
    out = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch)(delayed_list)

    (test_measures_dicts,
     train_measures_dicts,
     fit_times,
     test_times,
     num_tested) = zip(*out)

    test_measures = dict()
    train_measures = dict()
    ret = dict()

    for m in test_measures_dicts[0]:
        print(m)
        # transform list of dicts into dict of lists
        # Same as in GridSearchCV.fit()
        test_measures[m] = np.asarray([d[m] for d in test_measures_dicts])
        ret['test_' + m] = test_measures[m]
        if return_train_measures:
            train_measures[m] = np.asarray([d[m] for d in
                                            train_measures_dicts])
            ret['train_' + m] = test_measures[m]

    ret['fit_time'] = fit_times
    ret['test_time'] = test_times
    ret['num_tested'] = [num_tested for _ in fit_times]

    if verbose:
        print_summary(algo, measures, test_measures, train_measures, fit_times,
                      test_times, cv.n_splits)

    return ret



def merge_fit_and_score(fit_and_score_outputs):
    """
    Takes in a list of results.
    Each item in the list of results (fit_and_score_outputs) has a 
    test_measure_dict, train_measure_dict, fit_time, test_times, num_tested

    Specifically for use with cross_validate_users. Re-use with care
    """
    merged_test = {}
    merged_train = {}
    concatenated_fit_times = []
    concatenated_test_times = []
    concatenated_num_tested = []

    for output in fit_and_score_outputs:
        merged_test.update(output[0])
        if output[1]:
            merged_train = merged_train.update(output[1])
        concatenated_fit_times.append(output[2])
        concatenated_test_times.append(output[3])
        concatenated_num_tested.append(output[4])
    return merged_test, merged_train, concatenated_fit_times, concatenated_test_times, concatenated_num_tested

def cross_validate_users(algo, data, all_uids, out_uids, measures=None, cv=5,
                   return_train_measures=False, n_jobs=-1,
                   pre_dispatch='2*n_jobs', verbose=False):
    '''
    Run a cross validation procedure for a given algorithm, reporting accuracy
    measures and computation times.

    See an example in the :ref:`User Guide <cross_validate_example>`.

    Args: 
        algo(:obj:`AlgoBase \
            <surprise.prediction_algorithms.algo_base.AlgoBase>`):
            The algorithm to evaluate.
        data(:obj:`Dataset <surprise.dataset.Dataset>`): The dataset on which
            to evaluate the algorithm.
        measures(list of string): The performance measures to compute. Allowed
            names are function names as defined in the :mod:`accuracy
            <surprise.accuracy>` module. Default is ``['rmse', 'mae']``.
        cv(cross-validation iterator, int or ``None``): Determines how the
            ``data`` parameter will be split (i.e. how trainsets and testsets
            will be defined). If an int is passed, :class:`KFold
            <surprise.model_selection.split.KFold>` is used with the
            appropriate ``n_splits`` parameter. If ``None``, :class:`KFold
            <surprise.model_selection.split.KFold>` is used with
            ``n_splits=5``.
        return_train_measures(bool): Whether to compute performance measures on
            the trainsets. Default is ``False``.
        n_jobs(int): The maximum number of folds evaluated in parallel.

            - If ``-1``, all CPUs are used.
            - If ``1`` is given, no parallel computing code is used at all,\
                which is useful for debugging.
            - For ``n_jobs`` below ``-1``, ``(n_cpus + n_jobs + 1)`` are\
                used.  For example, with ``n_jobs = -2`` all CPUs but one are\
                used.

            Default is ``-1``.
        pre_dispatch(int or string): Controls the number of jobs that get
            dispatched during parallel execution. Reducing this number can be
            useful to avoid an explosion of memory consumption when more jobs
            get dispatched than CPUs can process. This parameter can be:

            - ``None``, in which case all the jobs are immediately created\
                and spawned. Use this for lightweight and fast-running\
                jobs, to avoid delays due to on-demand spawning of the\
                jobs.
            - An int, giving the exact number of total jobs that are\
                spawned.
            - A string, giving an expression as a function of ``n_jobs``,\
                as in ``'2*n_jobs'``.

            Default is ``'2*n_jobs'``.
        verbose(int): If ``True`` accuracy measures for each split are printed,
            as well as train and test times. Averages and standard deviations
            over all splits are also reported. Default is ``False``: nothing is
            printed.

    Returns:
        dict: A dict with the following keys:

            - ``'test_*'`` where ``*`` corresponds to a lower-case accuracy
              measure, e.g. ``'test_rmse'``: numpy array with accuracy values
              for each testset.

            - ``'train_*'`` where ``*`` corresponds to a lower-case accuracy
              measure, e.g. ``'train_rmse'``: numpy array with accuracy values
              for each trainset. Only available if ``return_train_measures`` is
              ``True``.

            - ``'fit_time'``: numpy array with the training time in seconds for
              each split.

            - ``'test_time'``: numpy array with the testing time in seconds for
              each split.

    '''
    measures = [m.lower() for m in measures]

    cv = get_cv(cv)

    delayed_list = []
    for (trainset, testset, in_testset, out_testset) in cv.custom_user_split(data, all_uids, out_uids):
        delayed_list += [
            delayed(merge_fit_and_score)(fit_and_score(
                algo, trainset, some_testset, measures,
                return_train_measures, testset_name
            ) for (some_testset, testset_name) in (
                (testset, 'all'),
                (in_testset, 'in-group'),
                (out_testset, 'out-group'),
            ))
        ]
    out = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch)(tuple(delayed_list))

    (test_measures_dicts,
     train_measures_dicts,
     fit_times,
     test_times,
     num_tested) = zip(*out)

    test_measures = dict()
    train_measures = dict()
    ret = dict()

    for m in test_measures_dicts[0]:
        # transform list of dicts into dict of lists
        # Same as in GridSearchCV.fit()
        test_measures[m] = np.asarray([d[m] for d in test_measures_dicts])
        ret['test_' + m] = test_measures[m]
        if return_train_measures:
            train_measures[m] = np.asarray([d[m] for d in
                                            train_measures_dicts])
            ret['train_' + m] = test_measures[m]

    ret['fit_time'] = fit_times
    ret['test_time'] = test_times
    ret['num_tested'] = [num_tested for _ in fit_times]

    if verbose:
        print_summary(algo, measures, test_measures, train_measures, fit_times,
                      test_times, cv.n_splits)

    return ret


def fit_and_score(algo, trainset, testset, measures,
                  return_train_measures=False, testset_name='standard'):
    '''Helper method that trains an algorithm and compute accuracy measures on
    a testset. Also report train and test times.

    Args:
        algo(:obj:`AlgoBase \
            <surprise.prediction_algorithms.algo_base.AlgoBase>`):
            The algorithm to use.
        trainset(:obj:`Trainset <surprise.trainset.Trainset>`): The trainset.
        trainset(:obj:`testset`): The testset.
        measures(list of string): The performance measures to compute. Allowed
            names are function names as defined in the :mod:`accuracy
            <surprise.accuracy>` module.
        return_train_measures(bool): Whether to compute performance measures on
            the trainset. Default is ``False``.

    Returns:
        tuple: A tuple containing:

            - A dictionary mapping each accuracy metric to its value on the
            testset (keys are lower case).

            - A dictionary mapping each accuracy metric to its value on the
            trainset (keys are lower case). This dict is empty if
            return_train_measures is False.

            - The fit time in seconds.

            - The testing time in seconds.
    '''

    start_fit = time.time()
    algo.fit(trainset)
    fit_time = time.time() - start_fit
    start_test = time.time()
    predictions = algo.test(testset)
    test_time = time.time() - start_test

    if return_train_measures:
        train_predictions = algo.test(trainset.build_testset())

    test_measures = dict()
    train_measures = dict()
    for m in measures:
        f = getattr(accuracy, m.lower())
        result = f(predictions, verbose=0)
        if isinstance(result, tuple):
            sub_measures = m.split('_')
            for i_sm, sub_measure in enumerate(sub_measures):
                test_measures[sub_measure + '_' + testset_name] = result[i_sm]
        else:
            test_measures[m  + '_' + testset_name] = result
        # TODO: support return train measures (Copy or abstract the above code...)
        if return_train_measures:
            train_measures[m] = f(train_predictions, verbose=0)

    return test_measures, train_measures, fit_time, test_time, len(testset)


def print_summary(algo, measures, test_measures, train_measures, fit_times,
                  test_times, n_splits):
    '''Helper for printing the result of cross_validate.'''

    print('Evaluating {0} of algorithm {1} on {2} split(s).'.format(
          ', '.join((m.upper() for m in measures)),
          algo.__class__.__name__, n_splits))
    print()

    row_format = '{:<18}' + '{:<8}' * (n_splits + 2)
    s = row_format.format(
        '',
        *['Fold {0}'.format(i + 1) for i in range(n_splits)] + ['Mean'] +
        ['Std'])
    s += '\n'
    s += '\n'.join(row_format.format(
        key.upper() + ' (testset)',
        *['{:1.4f}'.format(v) for v in vals] +
        ['{:1.4f}'.format(np.mean(vals))] +
        ['{:1.4f}'.format(np.std(vals))])
        for (key, vals) in iteritems(test_measures))
    if train_measures:
        s += '\n'
        s += '\n'.join(row_format.format(
            key.upper() + ' (trainset)',
            *['{:1.4f}'.format(v) for v in vals] +
            ['{:1.4f}'.format(np.mean(vals))] +
            ['{:1.4f}'.format(np.std(vals))])
            for (key, vals) in iteritems(train_measures))
    s += '\n'
    s += row_format.format('Fit time',
                           *['{:.2f}'.format(t) for t in fit_times] +
                           ['{:.2f}'.format(np.mean(fit_times))] +
                           ['{:.2f}'.format(np.std(fit_times))])
    s += '\n'
    s += row_format.format('Test time',
                           *['{:.2f}'.format(t) for t in test_times] +
                           ['{:.2f}'.format(np.mean(test_times))] +
                           ['{:.2f}'.format(np.std(test_times))])
    print(s)
