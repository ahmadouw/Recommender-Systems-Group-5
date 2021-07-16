import model_content
import pandas as pd
import os
import csv
import dataprep

path_to_data = '../shared/data/project/validation/'
dataset_type = 'one_hour' # all_sorted, one_day, one_hour, one_week


used_features = model_content.used_features

all_features = ["text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains", \
                "tweet_type", "language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count", \
                "engaged_with_user_following_count", "engaged_with_user_is_verified",
                "engaged_with_user_account_creation", \
                "engaging_user_id", "enaging_user_follower_count", "enaging_user_following_count",
                "enaging_user_is_verified", \
                "enaging_user_account_creation", "engagee_follows_engager"]

all_features_to_idx = dict(zip(all_features, range(len(all_features))))


def parse_input_line(line):
    features = line  # .split("\x01")

    hashtags = features[all_features_to_idx['hashtags']]
    present_media = features[all_features_to_idx['present_media']]
    present_links = features[all_features_to_idx['present_links']]
    present_domains = features[all_features_to_idx['present_domains']]
    tweet_type = features[all_features_to_idx['tweet_type']]
    engaged_with_user_following_count = features[all_features_to_idx['engaged_with_user_following_count']]
    engaged_with_user_is_verified = features[all_features_to_idx['engaged_with_user_is_verified']]
    engaged_with_user_account_creation = features[all_features_to_idx['engaged_with_user_account_creation']]
    engagee_follows_engager = features[all_features_to_idx['engagee_follows_engager']]

    return (hashtags, present_media, present_links, present_domains, tweet_type, engaged_with_user_following_count,
            engaged_with_user_is_verified, engaged_with_user_account_creation, engagee_follows_engager)


def evaluate_test_set():
    expanded_path = os.path.expanduser(path_to_data)
    part_files = [os.path.join(expanded_path, f) for f in os.listdir(expanded_path) if dataset_type in f]
    part_files = sorted(part_files, key=lambda x: x[-5:])
    with open('results_content.csv', 'w') as output:
        for file in part_files:
            with open(file, 'r') as f:
                print("doing", file)
                linereader = csv.reader(f, delimiter='\x01')
                last_timestamp = None
                df = pd.DataFrame(columns=all_features)
                i = 0
                for row in linereader:
                    df.loc[i] = row[:20]
                    i += 1
                    print("doing line ", i)
                    if i > 100000:
                        break

                df_complete = df.copy()
                df = df.loc[:, used_features]
                df = dataprep.transform_data(df)

                df = pd.DataFrame(df.values, columns=df.columns, index=df.index)

                print("got pd")
                for index, row in df.iterrows():
                    tweet_id = df_complete.iloc[[index]]["tweet_id"]
                    user_id = df_complete.iloc[[index]]["engaging_user_id"]

                    reply_pred = model_content.reply_pred_model(df.iloc[[index]])
                    retweet_pred = model_content.retweet_pred_model(df.iloc[[index]])
                    quote_pred = model_content.quote_pred_model(df.iloc[[index]])
                    fav_pred = model_content.fav_pred_model(df.iloc[[index]])

                    output.write(
                        f'{tweet_id.values[0]},{user_id.values[0]},{reply_pred[0]},{retweet_pred[0]},{quote_pred[0]},{fav_pred[0]}\n')


evaluate_test_set()