'''
The :mod:`model_selection.split<surprise.model_selection.split>` module
contains various cross-validation iterators. Design and tools are inspired from
the mighty scikit learn.

The available iterators are:

.. autosummary::
    :nosignatures:

    KFold
    RepeatedKFold
    ShuffleSplit
    LeaveOneOut
    PredefinedKFold

This module also contains a function for splitting datasets into trainset and
testset:

.. autosummary::
    :nosignatures:

    train_test_split

'''

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from itertools import chain
from math import ceil, floor
from collections import defaultdict
import time

import numbers

from six import iteritems
from six import string_types

import numpy as np

from ..utils import get_rng


def get_cv(cv):
    '''Return a 'validated' CV iterator.'''

    if cv is None:
        return KFold(n_splits=5)
    if isinstance(cv, numbers.Integral):
        return KFold(n_splits=cv)
    if hasattr(cv, 'split') and not isinstance(cv, string_types):
        return cv  # str have split

    raise ValueError('Wrong CV object. Expecting None, an int or CV iterator, '
                     'got a {}'.format(type(cv)))


class KFold():
    '''A basic cross-validation iterator.

    Each fold is used once as a testset while the k - 1 remaining folds are
    used for training.

    See an example in the :ref:`User Guide <use_cross_validation_iterators>`.

    Args:
        n_splits(int): The number of folds.
        random_state(int, RandomState instance from numpy, or ``None``):
            Determines the RNG that will be used for determining the folds. If
            int, ``random_state`` will be used as a seed for a new RNG. This is
            useful to get the same splits over multiple calls to ``split()``.
            If RandomState instance, this same instance is used as RNG. If
            ``None``, the current RNG from numpy is used. ``random_state`` is
            only used if ``shuffle`` is ``True``.  Default is ``None``.
        shuffle(bool): Whether to shuffle the ratings in the ``data`` parameter
            of the ``split()`` method. Shuffling is not done in-place. Default
            is ``True``.
    '''

    def __init__(self, n_splits=5, random_state=None, shuffle=True):

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, data):
        '''Generator function to iterate over trainsets and testsets.

        Args:
            data(:obj:`Dataset<surprise.dataset.Dataset>`): The data containing
                ratings that will be devided into trainsets and testsets.

        Yields:
            tuple of (trainset, testset)
        '''

        if self.n_splits > len(data.raw_ratings) or self.n_splits < 2:
            raise ValueError('Incorrect value for n_splits={0}. '
                             'Must be >=2 and less than the number '
                             'of ratings'.format(len(data.raw_ratings)))

        # We use indices to avoid shuffling the original data.raw_ratings list.
        indices = np.arange(len(data.raw_ratings))

        if self.shuffle:
            get_rng(self.random_state).shuffle(indices)

        start, stop = 0, 0
        for i_fold in range(self.n_splits):
            start = stop
            stop += len(indices) // self.n_splits
            if i_fold < len(indices) % self.n_splits:
                stop += 1

            raw_trainset = [data.raw_ratings[i] for i in chain(indices[:start],
                                                               indices[stop:])]
            raw_testset = [data.raw_ratings[i] for i in indices[start:stop]]

            trainset = data.construct_trainset(raw_trainset)
            testset = data.construct_testset(raw_testset)

            yield trainset, testset


    def custom_rating_split(self, nonboycott, boycott, boycott_uid_sets, like_boycott_uid_sets):
        '''function to iterate over trainsets and testsets.

        Args:
            nonboycott - the nonboycott ratigns
            boycott - the boycott ratings
            boycott_uid_sets - a dict of identifier keys and set values (uids)
            like_boycott_uid_sets - a dict of identifier keys and set values (uids)

        You can pass many boycott uid sets at once


        Returns:
            list of [trainset, nonboycott_testset, boycott_testset, like_boycott_but_testset, all_like_boycott_testset, all_testset]
        
        Note 7/24
        Making .raw_ratings into structured numpy array made this WAY (2-3x) slower
        List comprehensions are represensible!
        '''

        if self.n_splits > len(nonboycott.raw_ratings) or self.n_splits < 2:
            raise ValueError('Incorrect value for n_splits={0}. '
                             'Must be >=2 and less than the number '
                             'of ratings'.format(len(nonboycott.raw_ratings)))


        # We use indices to avoid shuffling the original data.raw_ratings list.
        indices = np.arange(len(nonboycott.raw_ratings))
        boycott_indices = np.arange(len(boycott.raw_ratings))
        #ret = []

        if self.shuffle:
            get_rng(self.random_state).shuffle(indices)
            get_rng(self.random_state).shuffle(boycott_indices)
        start, stop = 0, 0
        boycott_start, boycott_stop = 0, 0
        for i_fold in range(self.n_splits):
            starttime_fold = time.time()

            start = stop
            stop += len(indices) // self.n_splits
            if i_fold < len(indices) % self.n_splits:
                stop += 1

            boycott_start = boycott_stop
            boycott_stop += len(boycott_indices) // self.n_splits
            if i_fold < len(boycott_indices) % self.n_splits:
                boycott_stop += 1

            tic = time.time()
            # raw_trainset = [nonboycott.raw_ratings[i] for i in chain(indices[:start],
            #                                                    indices[stop:])]
            # NMV 7/24: numpy-ify this
            raw_trainset = nonboycott.raw_ratings[np.hstack([indices[:start],indices[stop:]])]
            #print('raw_trainset took {} seconds'.format(time.time() - tic))

            tic = time.time()
            #nonboycott_ratings_for_test = [nonboycott.raw_ratings[i] for i in indices[start:stop]]
            # NMV 7/24: numpy-ify this
            nonboycott_ratings_for_test = nonboycott.raw_ratings[indices[start:stop]]
            #print('nonboycott_ratings_for_test took {} seconds'.format(time.time() - tic))

            # nonboycott is a Data() object with the construct_trainset methods
            # whether we call nonboycott.construct_ or boycott.construct_ is arbitrary
            trainset = nonboycott.construct_trainset(raw_trainset)
            row = {}
            
            #print('Stuff before boycott_uid_set took {} seconds'.format(time.time() - starttime_fold))
            for (
                identifier, boycott_uid_set
            ), (
                identifier2, like_boycott_uid_set
            ) in zip(
                    boycott_uid_sets.items(), like_boycott_uid_sets.items()
            ):
                tic = time.time()                
                # this is probably a good assert to keep b/c iteration of python dictionaries is weird
                # but this works as is, so no need for OrderedDict
                assert identifier == identifier2
                boycott_testratings = []
                nonboycott_testratings = []
                like_boycott_but_testratings = []

                boycott_ratings_in_nonboycott_indices = []
                nonboycott_indices = []
                like_boycott_indices = []
                for i, uid in enumerate(nonboycott_ratings_for_test['uid'].tolist()):
                    if uid in boycott_uid_set:
                        boycott_ratings_in_nonboycott_indices.append(i)
                    else:
                        nonboycott_indices.append(i)
                        if uid in like_boycott_uid_set:
                            like_boycott_indices.append(i)
                
                boycott_testratings = nonboycott_ratings_for_test[boycott_ratings_in_nonboycott_indices]
                nonboycott_testratings = nonboycott_ratings_for_test[nonboycott_indices]
                like_boycott_but_testratings = nonboycott_ratings_for_test[like_boycott_indices]

                
                # Nick Vincent 7/21/2018
                # running through each element in a list comp and appending said element to a list
                # is equivalent to just adding the contents of that list comp to the list
                # boycott_testratings += [
                #     boycott.raw_ratings[i] for i in boycott_indices[boycott_start:boycott_stop]
                # ]

                # NMV 7/24
                # numpy-ify
                boycott_testratings = np.concatenate(
                    [boycott_testratings, boycott.raw_ratings[boycott_indices[boycott_start:boycott_stop]]], axis=0
                )
                all_like_boycott_testratings = np.concatenate([boycott_testratings, like_boycott_but_testratings], axis=0)
                all_testratings = np.concatenate([boycott_testratings, nonboycott_testratings], axis=0)

                nonboycott_testset = nonboycott.construct_testset(nonboycott_testratings)
                boycott_testset = nonboycott.construct_testset(boycott_testratings)
                like_boycott_but_testset = nonboycott.construct_testset(like_boycott_but_testratings)
                all_like_boycott_testset = nonboycott.construct_testset(all_like_boycott_testratings)
                all_testset = nonboycott.construct_testset(all_testratings)

                # using a list instead of a dict here leaves room for error.

                row[identifier] = [trainset, nonboycott_testset, boycott_testset, like_boycott_but_testset, all_like_boycott_testset, all_testset]
                #print('identifier {} took {} seconds'.format(identifier, time.time() - tic))
            yield row
        #     ret.append(row)
        # return ret 

    def custom_user_split_fraction(self, data, all_user_ids, out_user_ids):
        '''Generator function to iterate over trainsets and testsets.

        Args:
            data(:obj:`Dataset<surprise.dataset.Dataset>`): The data containing
                ratings that will be devided into trainsets and testsets.

        Yields:
            tuple of (trainset, testset)
        '''
        if self.n_splits > len(data.raw_ratings) or self.n_splits < 2:
            raise ValueError('Incorrect value for n_splits={0}. '
                             'Must be >=2 and less than the number '
                             'of ratings'.format(len(data.raw_ratings)))

        n_chunks = 2 # 2 chunks means each user will have ratings split into 2 chunks
        out_user_ids = set(out_user_ids)
        ret = []
        
        # We use indices to avoid shuffling the original data.raw_ratings list.
        user_id_indices = np.arange(len(all_user_ids))

        if self.shuffle:
            get_rng(self.random_state).shuffle(user_id_indices)
        start, stop = 0, 0
        for i_fold in range(self.n_splits):
            start = stop
            stop += len(user_id_indices) // self.n_splits
            if i_fold < len(user_id_indices) % self.n_splits:
                stop += 1

            uids_for_raw_trainset = [
                all_user_ids[i] for i in chain(
                    user_id_indices[:start],
                    user_id_indices[stop:])
            ]
            # kick out boycotting users
            uids_for_raw_trainset = set([
                x for x in uids_for_raw_trainset if x not in out_user_ids
            ])
            uids_for_raw_testset = set([
                all_user_ids[i] for i in user_id_indices[start:stop]
            ])
            # boycotting users can't be in the In-Group testset
            uids_for_in_testset = set([
                x for x in uids_for_raw_testset if x not in out_user_ids
            ])

            # for each uid that will be tested upon, need to get a list of each rating row
            uid_to_list_of_rating_rows = defaultdict(list)
            # for each uid that will be tested upon, let's randomly create chunks and put them into the dict below
            uid_to_chunks = defaultdict(dict)
            base_raw_trainset = []
            for raw_rating_row in data.raw_ratings:
                uid = raw_rating_row[0]
                # if the uid is approved for training (i.e. the user is not boycotting and not in the testset)
                if uid in uids_for_raw_trainset:
                    base_raw_trainset.append(raw_rating_row)
                elif uid in uids_for_raw_testset:
                    uid_to_list_of_rating_rows[uid].append(raw_rating_row)
            assert len(uid_to_list_of_rating_rows) == len(uids_for_raw_testset)
            for uid, list_of_rating_rows in uid_to_list_of_rating_rows.items():
                get_rng(self.random_state).shuffle(list_of_rating_rows)
                num_ratings_rows = len(list_of_rating_rows)
                chunksize = num_ratings_rows // n_chunks
                chunkstart, chunkstop = 0, 0
                for i_chunk in range(n_chunks):
                    chunkstart = chunkstop
                    chunkstop += chunksize
                    if i_chunk < num_ratings_rows % n_chunks:
                        chunkstop += 1
                    uid_to_chunks[uid][i_chunk] = list_of_rating_rows[chunkstart:chunkstop]
            assert(len(uid_to_list_of_rating_rows) == len(uid_to_chunks))

            for i_chunk in range(n_chunks):
                raw_testset, in_testset, out_testset = [], [], []
                # copy the base raw trainset into a NEW list.
                # we are going to want to add more ratings, but make sure not to add overlapping ratings
                raw_trainset_for_chunk_i = list(base_raw_trainset)
                # let's figure out all the other possible chunk indices
                other_chunk_indices = []
                for j_chunk in range(n_chunks):
                    if j_chunk != i_chunk:
                        other_chunk_indices.append(j_chunk)
                for uid in uid_to_list_of_rating_rows:
                    chunk = uid_to_chunks[uid][i_chunk]
                    raw_testset += chunk
                    if uid in uids_for_in_testset:
                        # this user is in the In-Group! So add these ratings to the in testset
                        in_testset += chunk
                        # we've allocated one chunk for testing
                        # so we should put the other chunks into the testset now!
                        # but only if this user is in the In-Group
                        for j_chunk in other_chunk_indices:
                            raw_trainset_for_chunk_i += uid_to_chunks[uid][j_chunk]
                    else:
                        out_testset += chunk
                assert(len(in_testset) + len(out_testset) == len(raw_testset))
                # print('This corresponds to {} testset ratings'.format(len(raw_testset)))
                # print('Train: {}, In: {}, Out: {}'.format(
                #     len(raw_trainset_for_chunk_i), len(in_testset), len(out_testset),
                # ))
                trainset = data.construct_trainset(raw_trainset_for_chunk_i)
                testset = data.construct_testset(raw_testset)
                in_testset = data.construct_testset(in_testset)
                out_testset = data.construct_testset(out_testset)

                ret.append([trainset, testset, in_testset, out_testset])
        assert(len(ret) == self.n_splits * n_chunks)
        return ret

    def get_n_folds(self):

        return self.n_splits


class RepeatedKFold():
    '''
    Repeated :class:`KFold` cross validator.

    Repeats :class:`KFold` n times with different randomization in each
    repetition.

    See an example in the :ref:`User Guide <use_cross_validation_iterators>`.

    Args:
        n_splits(int): The number of folds.
        n_repeats(int): The number of repetitions.
        random_state(int, RandomState instance from numpy, or ``None``):
            Determines the RNG that will be used for determining the folds. If
            int, ``random_state`` will be used as a seed for a new RNG. This is
            useful to get the same splits over multiple calls to ``split()``.
            If RandomState instance, this same instance is used as RNG. If
            ``None``, the current RNG from numpy is used. ``random_state`` is
            only used if ``shuffle`` is ``True``.  Default is ``None``.
        shuffle(bool): Whether to shuffle the ratings in the ``data`` parameter
            of the ``split()`` method. Shuffling is not done in-place. Default
            is ``True``.
    '''

    def __init__(self, n_splits=5, n_repeats=10, random_state=None):

        self.n_repeats = n_repeats
        self.random_state = random_state
        self.n_splits = n_splits

    def split(self, data):
        '''Generator function to iterate over trainsets and testsets.

        Args:
            data(:obj:`Dataset<surprise.dataset.Dataset>`): The data containing
                ratings that will be devided into trainsets and testsets.

        Yields:
            tuple of (trainset, testset)
        '''

        rng = get_rng(self.random_state)

        for _ in range(self.n_repeats):
            cv = KFold(n_splits=self.n_splits, random_state=rng, shuffle=True)
            for trainset, testset in cv.split(data):
                yield trainset, testset

    def get_n_folds(self):

        return self.n_repeats * self.n_splits


class ShuffleSplit():
    '''A basic cross-validation iterator with random trainsets and testsets.

    Contrary to other cross-validation strategies, random splits do not
    guarantee that all folds will be different, although this is still very
    likely for sizeable datasets.

    See an example in the :ref:`User Guide <use_cross_validation_iterators>`.

    Args:
        n_splits(int): The number of folds.
        test_size(float or int ``None``): If float, it represents the
            proportion of ratings to include in the testset. If int,
            represents the absolute number of ratings in the testset. If
            ``None``, the value is set to the complement of the trainset size.
            Default is ``.2``.
        train_size(float or int or ``None``): If float, it represents the
            proportion of ratings to include in the trainset. If int,
            represents the absolute number of ratings in the trainset. If
            ``None``, the value is set to the complement of the testset size.
            Default is ``None``.
        random_state(int, RandomState instance from numpy, or ``None``):
            Determines the RNG that will be used for determining the folds. If
            int, ``random_state`` will be used as a seed for a new RNG. This is
            useful to get the same splits over multiple calls to ``split()``.
            If RandomState instance, this same instance is used as RNG. If
            ``None``, the current RNG from numpy is used. ``random_state`` is
            only used if ``shuffle`` is ``True``.  Default is ``None``.
        shuffle(bool): Whether to shuffle the ratings in the ``data`` parameter
            of the ``split()`` method. Shuffling is not done in-place. Setting
            this to `False` defeats the purpose of this iterator, but it's
            useful for the implementation of :func:`train_test_split`. Default
            is ``True``.
    '''

    def __init__(self, n_splits=5, test_size=.2, train_size=None,
                 random_state=None, shuffle=True):

        if n_splits <= 0:
            raise ValueError('n_splits = {0} should be strictly greater than '
                             '0.'.format(n_splits))
        if test_size is not None and test_size <= 0:
            raise ValueError('test_size={0} should be strictly greater than '
                             '0'.format(test_size))

        if train_size is not None and train_size <= 0:
            raise ValueError('train_size={0} should be strictly greater than '
                             '0'.format(train_size))

        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.shuffle = shuffle

    def validate_train_test_sizes(self, test_size, train_size, n_ratings):

        if test_size is not None and test_size >= n_ratings:
            raise ValueError('test_size={0} should be less than the number of '
                             'ratings {1}'.format(test_size, n_ratings))

        if train_size is not None and train_size >= n_ratings:
            raise ValueError('train_size={0} should be less than the number of'
                             ' ratings {1}'.format(train_size, n_ratings))

        if np.asarray(test_size).dtype.kind == 'f':
            test_size = ceil(test_size * n_ratings)

        if train_size is None:
            train_size = n_ratings - test_size
        elif np.asarray(train_size).dtype.kind == 'f':
            train_size = floor(train_size * n_ratings)

        if test_size is None:
            test_size = n_ratings - train_size

        if train_size + test_size > n_ratings:
            raise ValueError('The sum of train_size and test_size ({0}) '
                             'should be smaller than the number of '
                             'ratings {1}.'.format(train_size + test_size,
                                                   n_ratings))

        return int(train_size), int(test_size)

    def split(self, data):
        '''Generator function to iterate over trainsets and testsets.

        Args:
            data(:obj:`Dataset<surprise.dataset.Dataset>`): The data containing
                ratings that will be devided into trainsets and testsets.

        Yields:
            tuple of (trainset, testset)
        '''

        test_size, train_size = self.validate_train_test_sizes(
            self.test_size, self.train_size, len(data.raw_ratings))
        rng = get_rng(self.random_state)

        for _ in range(self.n_splits):

            if self.shuffle:
                permutation = rng.permutation(len(data.raw_ratings))
            else:
                permutation = np.arange(len(data.raw_ratings))

            raw_trainset = [data.raw_ratings[i] for i in
                            permutation[:test_size]]
            raw_testset = [data.raw_ratings[i] for i in
                           permutation[test_size:(test_size + train_size)]]

            trainset = data.construct_trainset(raw_trainset)
            testset = data.construct_testset(raw_testset)

            yield trainset, testset

    def get_n_folds(self):

        return self.n_splits


def train_test_split(data, test_size=.2, train_size=None, random_state=None,
                     shuffle=True):
    '''Split a dataset into trainset and testset.

    See an example in the :ref:`User Guide <train_test_split_example>`.

    Note: this function cannot be used as a cross-validation iterator.

    Args:
        data(:obj:`Dataset <surprise.dataset.Dataset>`): The dataset to split
            into trainset and testset.
        test_size(float or int ``None``): If float, it represents the
            proportion of ratings to include in the testset. If int,
            represents the absolute number of ratings in the testset. If
            ``None``, the value is set to the complement of the trainset size.
            Default is ``.2``.
        train_size(float or int or ``None``): If float, it represents the
            proportion of ratings to include in the trainset. If int,
            represents the absolute number of ratings in the trainset. If
            ``None``, the value is set to the complement of the testset size.
            Default is ``None``.
        random_state(int, RandomState instance from numpy, or ``None``):
            Determines the RNG that will be used for determining the folds. If
            int, ``random_state`` will be used as a seed for a new RNG. This is
            useful to get the same splits over multiple calls to ``split()``.
            If RandomState instance, this same instance is used as RNG. If
            ``None``, the current RNG from numpy is used. ``random_state`` is
            only used if ``shuffle`` is ``True``.  Default is ``None``.
        shuffle(bool): Whether to shuffle the ratings in the ``data``
            parameter. Shuffling is not done in-place. Default is ``True``.
    '''
    ss = ShuffleSplit(n_splits=1, test_size=test_size, train_size=train_size,
                      random_state=random_state, shuffle=shuffle)
    return next(ss.split(data))


class LeaveOneOut():
    '''Cross-validation iterator where each user has exactly one rating in the
    testset.

    Contrary to other cross-validation strategies, ``LeaveOneOut`` does not
    guarantee that all folds will be different, although this is still very
    likely for sizeable datasets.

    See an example in the :ref:`User Guide <use_cross_validation_iterators>`.

    Args:
        n_splits(int): The number of folds.
        random_state(int, RandomState instance from numpy, or ``None``):
            Determines the RNG that will be used for determining the folds. If
            int, ``random_state`` will be used as a seed for a new RNG. This is
            useful to get the same splits over multiple calls to ``split()``.
            If RandomState instance, this same instance is used as RNG. If
            ``None``, the current RNG from numpy is used. ``random_state`` is
            only used if ``shuffle`` is ``True``.  Default is ``None``.
        min_n_ratings(int): Minimum number of ratings for each user in the
            trainset. E.g. if ``min_n_ratings`` is ``2``, we are sure each user
            has at least ``2`` ratings in the trainset (and ``1`` in the
            testset). Other users are discarded. Default is ``0``, so some
            users (having only one rating) may be in the testset and not in the
            trainset.
    '''

    def __init__(self, n_splits=5, random_state=None, min_n_ratings=0):

        self.n_splits = n_splits
        self.random_state = random_state
        self.min_n_ratings = min_n_ratings

    def split(self, data):
        '''Generator function to iterate over trainsets and testsets.

        Args:
            data(:obj:`Dataset<surprise.dataset.Dataset>`): The data containing
                ratings that will be devided into trainsets and testsets.

        Yields:
            tuple of (trainset, testset)
        '''

        # map ratings to the users ids
        user_ratings = defaultdict(list)
        for uid, iid, r_ui, _ in data.raw_ratings:
            user_ratings[uid].append((uid, iid, r_ui, None))

        rng = get_rng(self.random_state)

        for _ in range(self.n_splits):
            # for each user, randomly choose a rating and put it in the
            # testset.
            raw_trainset, raw_testset = [], []
            for uid, ratings in iteritems(user_ratings):
                if len(ratings) > self.min_n_ratings:
                    i = rng.randint(0, len(ratings))
                    raw_testset.append(ratings[i])
                    raw_trainset += [rating for (j, rating)
                                     in enumerate(ratings) if j != i]

            if not raw_trainset:
                raise ValueError('Could not build any trainset. Maybe '
                                 'min_n_ratings is too high?')
            trainset = data.construct_trainset(raw_trainset)
            testset = data.construct_testset(raw_testset)

            yield trainset, testset

    def get_n_folds(self):

        return self.n_splits


class PredefinedKFold():
    '''A cross-validation iterator to when a dataset has been loaded with the
    :meth:`load_from_folds <surprise.dataset.Dataset.load_from_folds>`
    method.

    See an example in the :ref:`User Guide <load_from_folds_example>`.
    '''

    def split(self, data):
        '''Generator function to iterate over trainsets and testsets.

        Args:
            data(:obj:`Dataset<surprise.dataset.Dataset>`): The data containing
                ratings that will be devided into trainsets and testsets.

        Yields:
            tuple of (trainset, testset)
        '''

        self.n_splits = len(data.folds_files)
        for train_file, test_file in data.folds_files:

            raw_trainset = data.read_ratings(train_file)
            raw_testset = data.read_ratings(test_file)
            trainset = data.construct_trainset(raw_trainset)
            testset = data.construct_testset(raw_testset)

            yield trainset, testset

    def get_n_folds(self):

        return self.n_splits
