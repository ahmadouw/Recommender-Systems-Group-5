import pandas as pd
import numpy as np


def encode_hashtags(hashtags):
    if type(hashtags) == str:
        return hashtags.split()
    else:
        return []


def encode_text_tokens(hashtags):
    if type(hashtags) == str:
        return hashtags.split()
    else:
        return []


def import_data(filepath, limit_dataset=False):
    all_all_features = ["text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",
                        "tweet_type", "language", "tweet_timestamp", "engaged_with_user_id",
                        "engaged_with_user_follower_count",
                        "engaged_with_user_following_count", "engaged_with_user_is_verified",
                        "engaged_with_user_account_creation",
                        "engaging_user_id", "enaging_user_follower_count", "enaging_user_following_count",
                        "enaging_user_is_verified",
                        "enaging_user_account_creation", "engagee_follows_engager", "retweet", "reply", "like",
                        "retweet_with_comment"]

    label_features = ["retweet", "reply", "like", "retweet_with_comment"]

    ratings_raw = pd.read_csv(filepath, delimiter='\x01', names=all_all_features)

    # set labels to True or False, omit timestamp
    for label in label_features:
        ratings_raw[label] = np.isnan(ratings_raw[label]) == False

    # factorize
    ratings_raw["language"] = ratings_raw["language"].factorize()[0]
    ratings_raw["tweet_type"] = ratings_raw["tweet_type"].factorize()[0]
    ratings_raw["present_media"] = ratings_raw["present_media"].factorize()[0]

    # strings to lists by delimiter
    ratings_raw["hashtags"] = ratings_raw["hashtags"].apply(encode_hashtags)
    ratings_raw["text_tokens"] = ratings_raw["text_tokens"].apply(encode_text_tokens)

    if limit_dataset:
        ratings_raw = ratings_raw[:300]

    return ratings_raw
