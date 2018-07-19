'''The validation module contains the cross_validate function, inspired from
the mighty scikit learn.'''

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import time
from pprint import pprint
from collections import defaultdict
from pprint import pprint
import ast

import numpy as np
from joblib import Parallel
from joblib import delayed
from six import iteritems

from .split import get_cv, KFold

from .. import accuracy

from surprise.prediction_algorithms.predictions import Prediction


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
     num_tested, _) = zip(*out)

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



def merge_scores(fit_and_score_outputs):
    """
    Takes in a list of results.
    Each item in the list of results (fit_and_score_outputs) has a 
    test_measure_dict, train_measure_dict, fit_time, test_times, num_tested

    Specifically for use with cross_validate_custom. Re-use with care


    Return values:
    ret should look like this
    {
        'all': {
            test_rmse: [1, 0.8, ...],
            test_ndcg10: [0.6, 0.7, ...],

            'test_times': [1.2, 1.3, ...],
            'fit_time': 5.4
            ...
        },
        'in-group': {
            test_rmse: [1.1, 1.2, ...],
            ...
        },

        ...f
    }
    """
    # indices = {
    #     'test': 0, # this is where test stuff lives in the return tuple
    #     'train': 1,
    #     'fit_times': 2,
    #     'test_times': 3,
    #     'num_tested': 4,
    #     'cv_index': 5
    # }
    merged_ret = defaultdict(dict)
    for output in fit_and_score_outputs: # we don't know what order these are in
        test_metrics, _, fit_time, test_times, num_tested, cv_index = output
        cv_index = int(cv_index)
        key_template = '{metric_name}_{testset_name}'
        for testset_name, metric_name_to_vals in test_metrics.items():
            for metric_name, vals in metric_name_to_vals.items():
                metric_key = key_template.format(**{
                    'metric_name': metric_name,
                    'testset_name': testset_name
                })
                merged_ret[metric_key][cv_index] = vals
            for metric_name, source in (
                ('test_times', test_times),
                ('num_tested', num_tested)
            ):
                key = key_template.format(**{
                    'metric_name': metric_name,
                    'testset_name': testset_name
                })
                merged_ret[key][cv_index] = source[testset_name]
            merged_ret[key_template.format(**{
                'metric_name': 'fit_time', 'testset_name': testset_name
            })][cv_index] = fit_time


    for metric_name, dict_with_int_keys in merged_ret.items():
        as_list = []
        for i in range(len(fit_and_score_outputs)):
            # why might a key be missing?
            # if there were no like-boycott ratings, those metrics are not calculated
            # Here we set it to nan, and we can deal w/ this later
            as_list.append(dict_with_int_keys.get(i, float('nan')))
        merged_ret[metric_name] = as_list

    return dict(merged_ret)


def cross_validate_many(
        algo, data, empty_boycott_data, boycott_uid_sets, like_boycott_uid_sets, measures=None, cv=5,
        n_jobs=-1, pre_dispatch='2*n_jobs', verbose=False, head_items=None,
        load_path=None
    ):
    """
    see cross_validate

    Does not return train measures currently.
    Does not support saving results  - why not?

    TODO: what's the difference here? Why would somebody use this one?
    """
    crossfold_index_to_args = {}
    
    measures = [m.lower() for m in measures]
    for i in range(cv):
        crossfold_index_to_args[i] = []
    cv = KFold(cv, random_state=0)
    for (
        crossfold_index, row
    ) in enumerate(cv.custom_rating_split(data, empty_boycott_data, boycott_uid_sets, like_boycott_uid_sets)):
        for identifier in boycott_uid_sets.keys():
            (
                trainset, nonboycott_testset, boycott_testset,
                like_boycott_but_testset, 
                all_like_boycott_testset,
                all_testset
            ) = row[identifier]
            specific_testsets = {
                'all' + '__' + identifier: all_testset, 
                'non-boycott' + '__' + identifier: nonboycott_testset,
                'boycott' + '__' + identifier: boycott_testset,
                'like-boycott' + '__' + identifier: like_boycott_but_testset,
                'all-like-boycott' + '__' + identifier: all_like_boycott_testset
            }
            if crossfold_index_to_args[crossfold_index]:
                crossfold_index_to_args[crossfold_index][2].update(specific_testsets)
            else:
                crossfold_index_to_args[crossfold_index] = [
                    algo, trainset, specific_testsets, measures, False, crossfold_index
                ]

    outputs = []
    for i in range(len(crossfold_index_to_args)):
        algo, trainset, specific_testsets, measures, return_train_measures, crossfold_index = crossfold_index_to_args[i]
        # specific_testsets - what does this like right here?
        # one key per sourcefile/id and evaluation group
        # e.g. all__SOMEFILE_0001
        output = fit_and_score_many(
            algo, trainset, specific_testsets, measures, return_train_measures, crossfold_index, head_items, load_path
        )
        outputs.append(output)

    ret = merge_scores(outputs)
    if verbose:
        print(ret)

    return ret


def cross_validate_custom(
        algo, nonboycott, boycott, boycott_uid_set, like_boycott_uid_set, measures=None, cv=5,
        return_train_measures=False, n_jobs=-1,
        pre_dispatch='2*n_jobs', verbose=False, head_items=None, save_path=None, load_path=None):
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

    cv = KFold(cv, random_state=0)
    args_list = []


    # IF we're going to parallelize to the folds do this
    # number the crossfolds so we can keep track of them when/if they go out to threads
    # note that if you're threading at the experiment level this code doesn't do much b/c n_jobs will be set to 1.
    # build all the args that will be sent out in parallel
    if n_jobs != 1:
        for (
            crossfold_index, row
        ) in enumerate(cv.custom_rating_split(nonboycott, boycott, {'only': boycott_uid_set}, {'only': like_boycott_uid_set})):
            (
                trainset, nonboycott_testset, boycott_testset,
                like_boycott_but_testset, all_like_boycott_testset,
                all_testset
            ) = row['only']
            args_list += [[
                algo, trainset, {
                    'all': all_testset, 'non-boycott': nonboycott_testset,
                    'boycott': boycott_testset, 'like-boycott': like_boycott_but_testset,
                    'all-like-boycott': all_like_boycott_testset
                },
                measures,
                return_train_measures, crossfold_index
            ]]
        delayed_gen = (
            delayed(fit_and_score)(
                algo, trainset, testsets, measures, return_train_measures, crossfold_index, head_items, save_path
            ) for algo, trainset, testsets, measures, return_train_measures, crossfold_index in args_list
        )
        out = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch)(delayed_gen)
    # But if we're not parallelizing folds (probably because we parallelized at the experiment level)
    # just do everthing in series so we don't use 5x memory for no reason
    else:
        out = []
        for (
            crossfold_index, row
        ) in enumerate(cv.custom_rating_split(nonboycott, boycott, {'only': boycott_uid_set}, {'only': like_boycott_uid_set})):
            (
                trainset, nonboycott_testset, boycott_testset,
                like_boycott_but_testset, all_like_boycott_testset,
                all_testset
            ) = row['only']
            testsets =  {
                    'all': all_testset, 'non-boycott': nonboycott_testset,
                    'boycott': boycott_testset, 'like-boycott': like_boycott_but_testset,
                    'all-like-boycott': all_like_boycott_testset
            }
            out += [fit_and_score(
                algo, trainset, testsets, measures, return_train_measures, crossfold_index, head_items, save_path
            )]

    ret = merge_scores(out)
    if verbose:
        print(ret)

    return ret


def batch(iterable, batch_size=1):
    """make batches for an iterable"""
    num_items = len(iterable)
    for ndx in range(0, num_items, batch_size):
        yield iterable[ndx:min(ndx + batch_size, num_items)]


def eval_task(algo, specific_testsets, measures, head_items, crossfold_index, save_path=None, load_path=None, test_mode=False):
    """
    Evaluate on specific testsets.
    This function exists to make testset evaluation easier to parallelize.
    """
    ret = []
    if load_path:
        load_from = '{}_seed0_fold{}_predictions.txt'.format(load_path, crossfold_index)
        print('load_from', load_from)
        with open(load_from, 'r') as file_handler:
            content = ['[' + x.strip('\n') + ']' for x in file_handler.readlines()]
            assert(content[0] == '[uid,iid,r_ui,est,details,crossfold_index]')
            all_predictions = [Prediction(*ast.literal_eval(line)[:-1]) for line in content[1:]]
            uid_plus_iid_to_row = {}
            for prediction in all_predictions:
                uid_plus_iid = str(prediction[0]) + '_' + str(prediction[1])
                uid_plus_iid_to_row[uid_plus_iid] = prediction
    for key, specific_testset in specific_testsets.items():
        tic = time.time()
        if load_path:
            predictions = []
            for prediction in specific_testset:
                uid_plus_iid = str(prediction[0]) + '_' + str(prediction[1])
                predictions.append(uid_plus_iid_to_row[uid_plus_iid])
            
            if test_mode:
                print('Testing')
                computed = algo.test(specific_testset)
                print('loaded\n', predictions[:3])
                print('computed\n', computed[:3])
                assert predictions == computed
        else:
            predictions = algo.test(specific_testset)

        if save_path and load_path is None: # if you just loaded the predictions, don't save them again, waste of time...
            with open('{}_seed0_fold{}_{}_predictions.txt'.format(save_path, crossfold_index, key), 'w') as file_handler:
                file_handler.write('uid,iid,r_ui,est,details,crossfold_index\n')
                for prediction in predictions:
                    file_handler.write(','.join([str(x) for x in prediction] + [str(crossfold_index)]) + '\n')

        if not predictions:
            ret.append([key, {}, 0, 0])
            continue
        test_time = time.time() - tic
        test_measures = {}
        for m in measures:
            eval_func = getattr(accuracy, m.lower())
            result = eval_func(predictions, verbose=0)
            if 'ndcg' in m:
                tail_result = eval_func(predictions, verbose=0, head_items=head_items)
                sub_measures = m.split('_')
                for i_sm, sub_measure in enumerate(sub_measures):
                    mean_val, frac_of_users = result[i_sm]
                    tail_mean_val, _ = tail_result[i_sm]
                    test_measures[sub_measure] = mean_val
                    test_measures[sub_measure + '_frac'] = frac_of_users
                    test_measures['tail' + sub_measure] = tail_mean_val
            else:
                test_measures[m] = result
        ret.append([key, test_measures, test_time, len(specific_testset)])
    return ret


def fit_and_score_many(
        algo, trainset, testset, measures,
        return_train_measures=False, crossfold_index=None, head_items=None,
        load_path=None, test_mode=False
    ):
    """
    see fit and score
    
    How is this different?
    Does not currently support saving...

    There is some repeated code from fit_and_score(...)
    Consider a refactor.
    """
    start_fit = time.time()
    if load_path is None or test_mode:
        algo.fit(trainset)
    fit_time = time.time() - start_fit

    test_times = {}
    num_tested = {}

    train_measures = {}
    ret_measures = {}
    if not isinstance(testset, dict):
        raise ValueError()
    # key is the testgroup (non-boycott, boycott, etc)
    # val is the list of ratings
    keys = list(testset.keys())
    delayed_list = []

    batchsize = 100 # TODO: why batch this many?
    for _, key_batch in enumerate(batch(keys, batchsize)):
        specific_testsets = {}
        for key in key_batch:
            print('key in fit_and_score_many', key)
            specific_testsets[key] = testset[key]
        delayed_list += [delayed(eval_task)(
            algo, specific_testsets, measures, head_items, crossfold_index, load_path=load_path, test_mode=test_mode
        )]


    out = Parallel(n_jobs=-1)((x for x in delayed_list))
    
    # flatten
    rows = []
    for chunk in out:
        for row in chunk:
            rows.append(row)
    
    for key, test_measures, specific_test_time, specific_num_tested in rows:
        if test_measures is None:
            continue
        ret_measures[key] = test_measures
        test_times[key] = specific_test_time
        num_tested[key] = specific_num_tested

    return ret_measures, train_measures, fit_time, test_times, num_tested, crossfold_index




def fit_and_score(
        algo, trainset, testset, measures,
        return_train_measures=False, crossfold_index=None, head_items=None, save_path=None, load_path=None
    ):
    '''Helper method that trains an algorithm and compute accuracy measures on
    a testset. Also report train and test times.

    Args:
        algo(:obj:`AlgoBase \
            <surprise.prediction_algorithms.algo_base.AlgoBase>`):
            The algorithm to use.
        trainset(:obj:`Trainset <surprise.trainset.Trainset>`): The trainset.
        trainset(:obj:`testset`): The testset.
        Could also be a dictionary of testsets. Key is the testset name and val is the testset.
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
    if load_path is None: # then we can't load predictions!
        algo.fit(trainset)
    fit_time = time.time() - start_fit
    start_test = time.time()

    test_times = {}
    num_tested = {}

    # make a `train_measures` dict no matter what, for backwards compatability
    train_measures = {}
    ret_measures = {}
    if isinstance(testset, dict):
        # key is the testgroup (non-boycott, boycott, etc)
        # val is the list of ratings
        results = eval_task(algo, testset, measures, head_items, crossfold_index, save_path=save_path, load_path=load_path)
        for (key, test_measures, test_time, num_tested_) in results:
            ret_measures[key] = test_measures
            test_times[key] = test_time
            num_tested[key] = num_tested_

    # backward compatability - in the original version testset is a list not a dict.
    # this is because typically there's just one testset, not multiple.
    else:
        predictions = algo.test(testset)
        test_time = time.time() - start_test                
        if not predictions:
            return {}, {}, 0, 0, 0, 0
        if return_train_measures:
            train_predictions = algo.test(trainset.build_testset())
        test_measures = dict()
        for m in measures:
            f = getattr(accuracy, m.lower())
            result = f(predictions, verbose=0)
            if isinstance(result, tuple):
                sub_measures = m.split('_')
                for i_sm, sub_measure in enumerate(sub_measures):
                    mean_val, frac_of_users = result[i_sm]
                    test_measures[sub_measure] = mean_val
                    test_measures[sub_measure + '_frac'] = frac_of_users
            else:
                test_measures[m] = result
            # TODO: support return train measures (Copy or abstract the above code...)
            if return_train_measures:
                train_measures[m] = f(train_predictions, verbose=0)

    return ret_measures if ret_measures else test_measures, train_measures, fit_time, test_times if test_times else test_time, num_tested, crossfold_index


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
