# -*- coding: utf-8 -*-
#
# Created on 2021-06-19 at 15:03.
# Created by andreasmerckel (coffeecodecruncher@gmail.com)
#


from sklearn.metrics import precision_score, mean_squared_error
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import matplotlib.pyplot as plt

import dataprep


def init_reply_model(data, target_name):
    feature_data = data.copy()
    target_data = data.loc[:, target_name]
    feature_data = feature_data.drop(labels=target_name,
                                     axis=1,
                                     inplace=False)

    nn_model = MLPClassifier(hidden_layer_sizes=(50, 50,),
                             activation="logistic",
                             solver='adam',
                             batch_size=100,
                             learning_rate_init=0.001,
                             max_iter=15000,
                             early_stopping=True,
                             tol=1e-3,
                             n_iter_no_change=300,
                             verbose=False)

    nn_model.fit(feature_data, target_data.values.ravel())

    return nn_model


def reply_pred_model(model, target_data, target_name):
    target_data = target_data.copy()
    target_data = target_data.drop(labels=target_name,
                                   axis=1,
                                   inplace=False)

    return model.predict(target_data)


def sba_plotting(df, x_axis, y_axis, color, title, xlabel, ylabel):
    plt.subplots(figsize=(16, 9))
    sns.barplot(x=x_axis, y=y_axis, color=color, data=df)
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.show()
    # print(df.DisbursementGross.describe())


def test():
    used_features = [
            # "text_tokens",
            # "hashtags",
            # "tweet_id",
            # "present_media",
            # "present_links",
            # "present_domains",
            # "tweet_type",
            # "language",
            "tweet_timestamp",
            "engaged_with_user_id",
            "engaged_with_user_follower_count",
            # "engaged_with_user_following_count",
            # "engaged_with_user_is_verified",
            # "engaged_with_user_account_creation",
            # "engaging_user_id",
            # "enaging_user_follower_count",
            # "enaging_user_following_count",
            # "enaging_user_is_verified",
            # "enaging_user_account_creation",
            # "engagee_follows_engager"
    ]
    target_features = [
            "retweet",
            # "reply",
            # "like",
            # "retweet_with_comment",
    ]
    data = dataprep.import_data(source_features=used_features,
                                target_features=target_features,
                                nrows=50)

    # print(data.iloc[0:3, 0:])
    # plt.subplots(figsize=(16, 9))
    # sns.heatmap(data.corr())
    # plt.show()

    used_features = [
            # "engaged_with_user_id",
            "engaged_with_user_follower_count",
            "engaged_with_user_following_count",
    ]

    data = dataprep.import_data(source_features=used_features,
                                target_features=target_features,
                                nrows=5000)

    print(data.head())

    train_data, test_data = dataprep.split_train_test(data)

    sub_1 = data.loc[data['retweet'] == 1]
    print(f"1: {sub_1.shape[0]}")
    sub_0 = data.loc[data['retweet'] == 0]
    print(f"0: {sub_0.shape[0]}")

    model_nn = init_reply_model(train_data, target_features)
    pred = reply_pred_model(model_nn, test_data, target_features)
    print(pred)
    precision = precision_score(test_data[target_features],
                                pred,
                                average="binary",
                                zero_division=1)
    print(f"Precision: {precision}")
    print(f"Error: {mean_squared_error(test_data[target_features], pred)}")


if __name__ == "__main__":
    test()