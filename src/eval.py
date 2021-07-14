# -*- coding: utf-8 -*-
#
# Created on 2021-07-04 at 16:28.
# Created by andreasmerckel (coffeecodecruncher@gmail.com)
#
import csv
import os

from matplotlib import pyplot as plt
from scikitplot.metrics import (
    plot_precision_recall,
    plot_roc,
)
import model_nn
import sklearn.metrics as skms

import model_content

path_to_data = '../shared/data/project/training/'
dataset_type = 'one_hour'

all_features = [
        "text_tokens",
        "hashtags",
        "tweet_id",
        "present_media",
        "present_links",
        "present_domains",
        "tweet_type",
        "language",
        "tweet_timestamp",
        "engaged_with_user_id",
        "engaged_with_user_follower_count",
        "engaged_with_user_following_count",
        "engaged_with_user_is_verified",
        "engaged_with_user_account_creation",
        "engaging_user_id",
        "enaging_user_follower_count",
        "enaging_user_following_count",
        "enaging_user_is_verified",
        "enaging_user_account_creation",
        "engagee_follows_engager"
]

target_features = [
            "retweet",
            "reply",
            "like",
            "retweet_with_comment"
]

all_features_to_idx = dict(zip(all_features, range(len(all_features))))


def parse_input_line(line):
    features = line  # .split("\x01")
    tweet_id = features[all_features_to_idx['tweet_id']]
    user_id = features[all_features_to_idx['engaging_user_id']]
    input_feats = features[all_features_to_idx['text_tokens']]
    tweet_timestamp = features[all_features_to_idx['tweet_timestamp']]
    return tweet_id, user_id, input_feats, tweet_timestamp


def evaluate_test_set(reply_pred_model,
                      retweet_pred_model,
                      quote_pred_model,
                      fav_pred_model):
    expanded_path = os.path.expanduser(path_to_data)
    part_files = [os.path.join(expanded_path, f) for f in
                  os.listdir(expanded_path) if dataset_type in f]
    part_files = sorted(part_files, key=lambda x: x[-5:])

    reply_results = []
    retweet_results = []
    quote_results = []
    fav_results = []

    for file in part_files:
        with open(file, 'r') as f:
            linereader = csv.reader(f, delimiter='\x01')
            for row in linereader:
                tweet_id, user_id, features, tweet_timestamp = parse_input_line(row)
                reply_pred = reply_pred_model(features)  # reply_model
                retweet_pred = retweet_pred_model(features)  # retweet_model
                quote_pred = quote_pred_model(features)  # pred_model
                fav_pred = fav_pred_model(features)  # fav_model
                reply_results.append(reply_pred)
                retweet_results.append(retweet_pred)
                quote_results.append(quote_pred)
                fav_results.append(fav_pred)

    return reply_results, retweet_results, quote_results, fav_results


def evaluate_models():
    print(f"EVALUATE MODELS")
    print(f"content based model")
    evaluate_test_set(model_content.reply_pred_model,
                      model_content.retweet_pred_model,
                      model_content.quote_pred_model,
                      model_content.fav_pred_model)
    print('=' * 50)
    print(f"machine learning model")
    # evaluate_test_set(model_nn.reply_pred_model,
    #                   model_nn.retweet_pred_model,
    #                   model_nn.quote_pred_model,
    #                   model_nn.fav_pred_model)
    # print('=' * 50)
    # print(ROC Curve)
    # fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
    # skms.roc_curve(y_test, y_test_prob)
    # plot_roc(y_test, y_probas, figsize=(16, 9))
    #
    # print(PR Curve)
    # plot_precision_recall(y_test, y_probas, figsize=(16, 9))
    # plt.show()


def main() -> None:
    evaluate_models()


if __name__ == "__main__":
    main()
