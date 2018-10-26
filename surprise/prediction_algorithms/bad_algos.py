"""
Intentionally bad algorithms to use as reference
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from six import iteritems
import heapq
from collections import defaultdict

from .predictions import PredictionImpossible
from .algo_base import AlgoBase


# Important note: as soon as an algorithm uses a similarity measure, it should
# also allow the bsl_options parameter because of the pearson_baseline
# similarity. It can be done explicitely (e.g. KNNBaseline), or implicetely
# using kwargs (e.g. KNNBasic).

class GuessThree(AlgoBase):

    def __init__(self):

        # Always call base method before doing anything.
        AlgoBase.__init__(self)

    def estimate(self, u, i):

        return 3


class GlobalMean(AlgoBase):

    def __init__(self):

        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.the_mean = self.trainset.global_mean
        #print(self.the_mean)
        # 3.581564453108
        return self

    def estimate(self, u, i):

        return self.the_mean


class MovieMean(AlgoBase):

    def __init__(self):

        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        movie_to_sum = defaultdict(int)
        movie_to_count = defaultdict(int)
        # (uid, iid, rating)
        for u, i, r in trainset.all_ratings():
            movie_to_sum[i] += r
            movie_to_count[i] += 1
        self.movie_to_mean = {}
        for movie_id, val in movie_to_sum.items():
            self.movie_to_mean[movie_id] = val / movie_to_count[movie_id]
        self.the_mean = self.trainset.global_mean
        return self

    def estimate(self, u, i):
        """takes inner ids"""
        if not self.trainset.knows_item(i):
            return self.the_mean
        return self.movie_to_mean[i]

class TwentyMean(AlgoBase):

    def __init__(self):

        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        movie_to_sum = defaultdict(int)
        movie_to_count = defaultdict(int)
        # (uid, iid, rating)
        uid_to_skip = {}
        for u, i, r in trainset.all_ratings():
            if u not in uid_to_skip:
                if np.random.rand() > 0.2:
                    uid_to_skip[u] = True
                    continue
                else:
                    uid_to_skip[u] = False
            else:
                if uid_to_skip[u] is True:
                    continue
            movie_to_sum[i] += r
            movie_to_count[i] += 1
        self.movie_to_mean = {}
        for movie_id, val in movie_to_sum.items():
            self.movie_to_mean[movie_id] = val / movie_to_count[movie_id]
        self.the_mean = self.trainset.global_mean
        return self

    def estimate(self, u, i):
        """takes inner ids"""
        if not self.trainset.knows_item(i):
            return self.the_mean
        try:
            return self.movie_to_mean[i]
        except:
            return self.the_mean