import pandas as pd
import numpy as np


def import_data(filepath):
    all_all_features = ["text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",\
                    "tweet_type","language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",\
                "engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",\
                "engaging_user_id", "enaging_user_follower_count", "enaging_user_following_count", "enaging_user_is_verified",\
                "enaging_user_account_creation", "engagee_follows_engager","retweet","reply","like","retweet_with_comment"]

    label_features = ["retweet","reply","like","retweet_with_comment"]
    
    ratings_raw = pd.read_csv(filepath, delimiter='\x01',names = all_all_features)

    # set labels to True or False, omit timestamp
    for label in label_features:
        ratings_raw[label] = np.isnan(ratings_raw[label]) == False

    # factorize
    ratings_raw["language"] = ratings_raw["language"].factorize()[0]
    ratings_raw["tweet_type"] = ratings_raw["tweet_type"].factorize()[0]
    ratings_raw["present_media"] = ratings_raw["present_media"].factorize()[0]

    return ratings_raw

