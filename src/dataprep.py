# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# default values
default_filepath = "../shared/data/project/training/one_hour"
validation_filepath = "../shared/data/project/training/one_hour"
default_features_all = [
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
        "engagee_follows_engager",
        "retweet",
        "reply",
        "like",
        "retweet_with_comment"
]
default_targets = [
        "retweet",
        "reply",
        "like",
        "retweet_with_comment"
]


# one-hot-encoding, e.g. language
def encode_one_hot(dataframe, feature_to_encode):
    target_column = dataframe.pop(feature_to_encode).apply(pd.Series)
    dummies = pd.get_dummies(target_column.apply(pd.Series).stack()).sum(level=0)
    return dummies


# target exists? 1-0, e.g. retweet
def encode_exists(element):
    return 0 if np.isnan(element) else 1


def split_string_to_list(target):
    if type(target) == str:
        return target.split()
    else:
        return []
    
def bool_to_int(target):
    if type(target) == str:
        if target == "false":
            return 0
        elif target == "true":
            return 1
        else:
            raise TypeError


# split string to list by whitespace then count elements, e.g. hashtags
def encode_length(target):
    if type(target) == list:
        return len(target)
    else:
        return 0


def transform_data(data):
    features_string_to_list = [
            "hashtags",
            "present_media",
            "present_links",
            "present_domains",
            "text_tokens"
    ]
    
    features_bool_to_int = [
        "engagee_follows_engager",
        "engaged_with_user_is_verified"
    ]
    
    for feature in features_string_to_list:
        if feature in data:
            data[feature] = data[feature].apply(split_string_to_list)
            
    for feature in features_bool_to_int:
        if feature in data:
            data[feature] = data[feature].apply(bool_to_int)

    features_to_one_hot = [
            "language",
            "present_media",
            "tweet_type"
    ]
    for feature in features_to_one_hot:
        if feature in data:
            one_hot = encode_one_hot(data, feature)
            data = pd.concat([one_hot, data], axis=1)

    features_to_target_encode = [
            "retweet",
            "reply",
            "like",
            "retweet_with_comment"
    ]
    for feature in features_to_target_encode:
        if feature in data:
            data[feature] = data[feature].where(
                    data[feature].isnull(), 1).fillna(0).astype(int)

    features_to_sum = [
            "hashtags",
            "present_links",
            "present_domains",
            "text_tokens"
    ]
    for feature in features_to_sum:
        if feature in data:
            data[feature] = data[feature].apply(lambda x: encode_length(x))

    # better strategy?
    data = data.fillna(0)

    return data


def import_data(filepath=default_filepath,
                source_features=default_features_all,
                target_features=default_targets,
                nrows=None,
                skiprows=0,
                use_transform_data=True):

    if target_features is None:
        target_features = default_targets

    all_features = np.unique(
            np.concatenate(
                    (source_features, target_features)
            )
    )
    ratings_raw = pd.read_csv(filepath,
                              delimiter='\x01',
                              names=default_features_all,
                              nrows=nrows,
                              skiprows=skiprows)

    # select needed features
    data = ratings_raw.loc[:, all_features]

    if use_transform_data:
        data = transform_data(data)

    return data


def split_train_test(data, test_size=0.2, shuffle=False):
    train_data_int, test_data_int = train_test_split(data,
                                                     test_size=test_size,
                                                     shuffle=shuffle)
    return train_data_int, test_data_int
