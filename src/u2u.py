import dataprep
import pandas as pd

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import norm
import sklearn.preprocessing as pp

from sklearn import svm
from sklearn.metrics import precision_score

from joblib import dump, load


def get_imported_data():
    used_features = ["engaging_user_id", "tweet_id"]
    target_features = ["retweet", "reply", "like", "retweet_with_comment"]
    # reply, like, retweet_with_comment
    return dataprep.import_data(source_features=used_features, target_features=target_features)


class Predictor:

    def __init__(self, interaction_category, input_data, dump_to_file=False):
        self.k = 5
        self.interaction_category = interaction_category

        self.filtered_data = input_data[input_data[interaction_category] == 1]

        tweet_ids = self.filtered_data["tweet_id"].unique()
        tweet_ids.sort()
        userIds = self.filtered_data["engaging_user_id"].unique()
        userIds.sort()

        self.m = userIds.size
        self.n = tweet_ids.size
        numRatings = len(input_data)

        ## create internal ids for movies and users, that have consecutive indexes starting from 0
        self.tweetId_to_tweetIDX = dict(zip(tweet_ids, range(0, tweet_ids.size)))
        self.tweetIDX_to_tweetId = dict(zip(range(0, tweet_ids.size), tweet_ids))

        self.userId_to_userIDX = dict(zip(userIds, range(0, userIds.size)))
        self.userIDX_to_userId = dict(zip(range(0, userIds.size), userIds))

        ## drop timestamps
        data = pd.concat([self.filtered_data['engaging_user_id'].map(self.userId_to_userIDX),
                          self.filtered_data['tweet_id'].map(self.tweetId_to_tweetIDX),
                          self.filtered_data[interaction_category]], axis=1)
        data.columns = ['engaging_user_id', 'tweet_id', 'interaction']

        self.R = sp.csr_matrix((data.interaction, (data.engaging_user_id, data.tweet_id)))
        self.R_dok = self.R.todok()
        if dump_to_file:
            dump(self, "model_content/u2u_"+interaction_category+".joblib")



    def compute_pairwise_user_similarity(self, u_id, v_id):
        u = self.R[u_id, :].copy()
        v = self.R[v_id, :].copy()

        # using the formula on slide 25 of slide deck 2

        # calculate sqrt of sum of (r_ui - mean(r_u))^2
        u_denominator = np.sqrt(sum(u.data))

        # calculate sqrt of sum of (r_vi - mean(r_v))^2
        v_denominator = np.sqrt(sum(v.data))

        denominator = u_denominator * v_denominator

        # calculate numerator
        numerator = 0

        def calculate_nth_summand(index):
            if (u_id, index) in self.R_dok and (v_id, index) in self.R_dok:
                return 1
            else:
                return 0

        numerator = np.array([calculate_nth_summand(t) for t in range(0, n)]).sum()

        if denominator == 0:
            similarity = 0.;
        else:
            similarity = numerator / denominator

        return similarity

    def compute_user_similarities(self, u_id):
        """
        Much faster matrix-based approach
        """

        R_copy = self.R.copy()

        u = self.R[u_id, :].copy()

        return R_copy.dot(u.T).toarray()[:, 0]

    def create_user_neighborhood(self, u_id, i_id):
        nh = {}  ## the neighborhood dict with (user id: similarity) entries
        ## nh should not contain u_id and only include users that have rated i_id; there should be at most k neighbors
        uU = self.compute_user_similarities(u_id)
        uU_copy = uU.copy()  ## so that we can modify it, but also keep the original
        sort_index = np.flip(np.argsort(uU_copy))

        taken = 0

        # select the top-k other users that have rated the target item.
        # usually, the neighborhood will be very small and often only contain users with a similarity of 0
        # this is because for most tweets, the user will not have another user that has previously interacted with a common tweet.
        for i in sort_index:
            if i == u_id:
                continue
            if self.R_dok[i, i_id] != 0:
                nh[i] = uU_copy[i]
                taken += 1
                if taken >= self.k:
                    break
            if uU_copy[i] == 0:
                break

        return nh

    def predict_rating(self, u_id, i_id):
        nh = self.create_user_neighborhood(u_id, i_id)


        numerator = np.array([nh[x] * (self.R[x, i_id]) for x in nh]).sum()
        denominator = np.array([abs(nh[x]) for x in nh]).sum()

        if denominator != 0:
            neighborhood_weighted_avg = numerator / denominator
        else:
            neighborhood_weighted_avg = 0

        prediction = neighborhood_weighted_avg

        return prediction

    def predict(self, user_original_id, tweet_original_id, binary=False):
        try:
            tweet_id = self.tweetId_to_tweetIDX[tweet_original_id]
            user_id = self.userId_to_userIDX[user_original_id]
            if binary:
                return 1 if self.predict_rating(user_id, tweet_id) > 0.5 else 0
            else:
                return self.predict_rating(user_id, tweet_id)
        except:
            return 0