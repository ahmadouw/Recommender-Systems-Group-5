# -*- coding: utf-8 -*-
#
# Created on 2021-06-19 at 15:03.
# Created by andreasmerckel (coffeecodecruncher@gmail.com)
#
import pickle
import matplotlib.pyplot as plt
import pandas as pd
# import scikitplot as skplt
import numpy as np
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import (
    MLPRegressor
)
from sklearn.preprocessing import StandardScaler
# from xgboost import XGBRegressor
import dataprep


def prepare_input_features(input_features, used_features):
    print(f"input_features: {input_features}")
    print(f"used_features: {used_features}")
    input_features = input_features.loc[:, used_features]
    return input_features
    #return dataprep.transform_data(input_features)


def prepare_input(data, target_feature):
    return rebalancer(data, target_feature)


def reply_pred_model(target_data):
    #hashtags, present_media, present_links, present_domains, tweet_type, engaged_with_user_following_count, engaged_with_user_is_verified, engaged_with_user_account_creation, engagee_follows_engager = target_data
    #target_data = prepare_input_features(target_data, get_reply_used_features())
    #target_data = pd.DataFrame(target_data, columns=["hashtags", "present_media", "present_links", "present_domains", "tweet_type", "engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation", "engagee_follows_engager"])
    #target_data = prepare_input_features(target_data, None)
    #print(f"post-prepare_input_features: {target_data}")
    with open("model_nn/nn_model_reply.pickle", "rb") as f:
        return predict(f, target_data)


def retweet_pred_model(target_data):
    #target_data = prepare_input_features(target_data, get_retweet_used_features())
    with open("model_nn/nn_model_retweet.pickle", "rb") as f:
        return predict(f, target_data)


def quote_pred_model(target_data):
    #target_data = prepare_input_features(target_data, get_quote_used_features())
    with open("model_nn/nn_model_retweet_with_comment.pickle", "rb") as f:
        return predict(f, target_data)


def fav_pred_model(target_data):
    #target_data = prepare_input_features(target_data, get_fav_used_features())
    with open("model_nn/nn_model_like.pickle", "rb") as f:
        return predict(f, target_data)


def predict(file, target_data):
    #target_data = normalize(target_data)
    model = pickle.load(file)
    #print(f"normalized: {target_data}")
    pred = model.predict(target_data)
    # return pred
    return np.clip(pred, 0, 1)


def init_retweet_model_nn(nrows=1000) -> None:
    used_features = get_retweet_used_features()
    target_features = [
            "retweet",
            # "reply",
            # "like",
            # "retweet_with_comment",
    ]

    data = dataprep.import_data(source_features=used_features,
                                target_features=target_features,
                                nrows=nrows)
    data = prepare_input(data, target_features[0])
    _init_model_nn(data, target_features[0])


def init_reply_model_nn(nrows=1000) -> None:
    used_features = get_reply_used_features()
    target_features = [
            # "retweet",
            "reply",
            # "like",
            # "retweet_with_comment",
    ]

    data = dataprep.import_data(source_features=used_features,
                                target_features=target_features,
                                nrows=nrows)
    data = prepare_input(data, target_features[0])
    _init_model_nn(data, target_features[0])


def init_fav_model_nn(nrows=1000) -> None:
    used_features = get_fav_used_features()
    target_features = [
            # "retweet",
            # "reply",
            "like",
            # "retweet_with_comment",
    ]

    data = dataprep.import_data(source_features=used_features,
                                target_features=target_features,
                                nrows=nrows)
    data = prepare_input(data, target_features[0])
    _init_model_nn(data, target_features[0])


def init_quote_model_nn(nrows=1000) -> None:
    used_features = get_quote_used_features()
    target_features = [
            # "retweet",
            # "reply",
            # "like",
            "retweet_with_comment",
    ]

    data = dataprep.import_data(source_features=used_features,
                                target_features=target_features,
                                nrows=nrows)
    data = prepare_input(data, target_features[0])
    _init_model_nn(data, target_features[0])


def get_retweet_used_features() -> [str]:
    used_features = [
            # "text_tokens",
            "hashtags",
            # "tweet_id",
            "present_media",
            "present_links",
            "present_domains",
            "tweet_type",
            # "language",
            # "tweet_timestamp",
            # "engaged_with_user_id",
            # "engaged_with_user_follower_count",
            "engaged_with_user_following_count",
            "engaged_with_user_is_verified",
            "engaged_with_user_account_creation",
            # "engaging_user_id",
            # "enaging_user_follower_count",
            # "enaging_user_following_count",
            # "enaging_user_is_verified",
            # "enaging_user_account_creation",
            "engagee_follows_engager"
    ]

    return used_features


def get_quote_used_features() -> [str]:
    used_features = [
            # "text_tokens",
            "hashtags",
            # "tweet_id",
            "present_media",
            "present_links",
            "present_domains",
            "tweet_type",
            # "language",
            # "tweet_timestamp",
            # "engaged_with_user_id",
            # "engaged_with_user_follower_count",
            "engaged_with_user_following_count",
            "engaged_with_user_is_verified",
            "engaged_with_user_account_creation",
            # "engaging_user_id",
            # "enaging_user_follower_count",
            # "enaging_user_following_count",
            # "enaging_user_is_verified",
            # "enaging_user_account_creation",
            "engagee_follows_engager"
    ]
    return used_features


def get_reply_used_features() -> [str]:
    used_features = [
            # "text_tokens",
            "hashtags",
            # "tweet_id",
            "present_media",
            "present_links",
            "present_domains",
            "tweet_type",
            # "language",
            # "tweet_timestamp",
            # "engaged_with_user_id",
            # "engaged_with_user_follower_count",
            "engaged_with_user_following_count",
            "engaged_with_user_is_verified",
            "engaged_with_user_account_creation",
            # "engaging_user_id",
            # "enaging_user_follower_count",
            # "enaging_user_following_count",
            # "enaging_user_is_verified",
            # "enaging_user_account_creation",
            "engagee_follows_engager"
    ]
    return used_features


def get_fav_used_features() -> [str]:
    used_features = [
            # "text_tokens",
            "hashtags",
            # "tweet_id",
            "present_media",
            "present_links",
            "present_domains",
            "tweet_type",
            # "language",
            # "tweet_timestamp",
            # "engaged_with_user_id",
            # "engaged_with_user_follower_count",
            "engaged_with_user_following_count",
            "engaged_with_user_is_verified",
            "engaged_with_user_account_creation",
            # "engaging_user_id",
            # "enaging_user_follower_count",
            # "enaging_user_following_count",
            # "enaging_user_is_verified",
            # "enaging_user_account_creation",
            "engagee_follows_engager"
    ]
    return used_features


def _init_model_nn(data, target_name) -> None:
    feature_data = data.copy()
    target_data = data.loc[:, target_name]
    feature_data = feature_data.drop(labels=target_name,
                                     axis=1,
                                     inplace=False)

    nn_model = MLPRegressor(hidden_layer_sizes=(50, 50,),
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
    with open(f"model_nn/nn_model_{target_name}.pickle", "wb") as f:
        pickle.dump(nn_model, f)


def init_reply_model_xg(data, target_name) -> None:
    feature_data = data.copy()
    target_data = data.loc[:, target_name]
    feature_data = feature_data.drop(labels=target_name,
                                     axis=1,
                                     inplace=False)

    nn_model = XGBRegressor()

    nn_model.fit(feature_data, target_data.values.ravel())
    with open("model_nn/xg_model.pickle", "wb") as f:
        pickle.dump(nn_model, f)


def rebalancer(df,
               target):
    """
    """
    y = df[target]
    x = df.drop(columns=[target], axis=1)
    x_scaled = normalize(x)

    over = SMOTE(sampling_strategy='minority')
    under = RandomUnderSampler()
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    x, y = pipeline.fit_resample(x_scaled, y)
    x = pd.DataFrame(x)
    x[target] = y
    x.columns = df.columns
    # x.append(y)
    return x


def normalize(x):
    scale = StandardScaler()
    return scale.fit_transform(x)


def model_nn_opt(estimator,
                 parameters,
                 x_train,
                 y_train,
                 algo,
                 x_test,
                 y_test,
                 scoring="f1"):
    fit = GridSearchCV(estimator=estimator,
                       param_grid=parameters,
                       scoring=scoring,
                       n_jobs=-1,
                       cv=3,
                       verbose=True).fit(x_train, y_train)

    best = fit.best_estimator_
    best_model = best.fit(x_train, y_train)

    y_test_ypred = best_model.predict(x_test)
    y_test_prob = best_model.predict_proba(x_test)[:, -1]
    y_probas = best_model.predict_proba(x_test)

    print(f"Best parameters: {fit.best_params_}")
    plt.show()

    #skplt.metrics.plot_roc(y_test, y_probas, figsize=(16, 9))
    #skplt.metrics.plot_precision_recall(y_test, y_probas, figsize=(16, 9))
    #plt.show()


def plotter(df,
            x_axis,
            y_axis,
            color="blue",
            title=""):
    print(title)
    plt.subplots(figsize=(16, 9))
    # sns.countplot(x=x_axis, data=df)
    # plt.hist(x_axis)
    # plt.show()
    # sns.barplot(x=x_axis, y=y_axis, color=color, data=df)
    # plt.title(title, fontsize=20)
    # plt.xlabel(xlabel, fontsize=15)
    # plt.ylabel(ylabel, fontsize=15)
    # plt.show()
    # print(data.iloc[0:3, 0:])
    ax = plt.axes()
    sns.heatmap(df.corr(), vmin=0, vmax=1)
    ax.set_title(title)
    with pd.option_context('display.max_rows',
                           None,
                           'display.max_columns',
                           None):
        print(df.corr()["retweet"])
    plt.show()


def shape_analysis(data, feature):
    print(f"{feature} & "
          f"{data.loc[data[feature] == 1].shape[0]} & "
          f"{data.loc[data[feature] == 0].shape[0]} \\\\")
    # false = data.loc[data[feature] == 0]
    # print(f"1: {retweet_1.shape[0]}")
    # print(f"0: {retweet_0.shape[0]}")


def test():
    used_features = [
            # "text_tokens",
            "hashtags",
            # "tweet_id",
            "present_media",
            "present_links",
            "present_domains",
            "tweet_type",
            # "language",
            # "tweet_timestamp",
            # "engaged_with_user_id",
            # "engaged_with_user_follower_count",
            "engaged_with_user_following_count",
            "engaged_with_user_is_verified",
            "engaged_with_user_account_creation",
            # "engaging_user_id",
            # "enaging_user_follower_count",
            # "enaging_user_following_count",
            # "enaging_user_is_verified",
            # "enaging_user_account_creation",
            "engagee_follows_engager"
    ]
    target_features = [
            "retweet",
            "reply",
            "like",
            "retweet_with_comment"
    ]
    # data = preprocessing.import_data(source_features=used_features,
    #                                  target_features=target_features,
    #                                  nrows=nrows)
    #
    # with open("out/out-1.csv", 'w') as f:
    #     print(data.to_string(index=False), file=f)
    #
    data = dataprep.import_data(source_features=used_features,
                                target_features=target_features,
                                nrows=nrows)
    #
    # with open("out/out-2.csv", 'w') as f:
    #     print(data.to_string(index=False), file=f)
    #
    # print(data.describe())
    #

    # plotter(data,
    #         target_features,
    #         used_features,
    #         title="Before rebalance")
    # data = rebalancer(data, target_features[0])
    # plotter(data,
    #         target_features,
    #         used_features,
    #         title="After rebalance")
    # print(data.head())
    # train_data, test_data = dataprep.split_train_test(data)
    #
    # init_reply_model_nn()
    # init_retweet_model_nn()
    # init_quote_model_nn()
    # init_fav_model_nn()

    # test_data = dataprep.import_data(use_transform_data=False, nrows=nrows)
    # pred = retweet_pred_model(test_data)
    # print(pred)
    #
    # pred = reply_pred_model(test_data)
    # print(pred)
    #
    # pred = fav_pred_model(test_data)
    # print(pred)
    #
    # pred = quote_pred_model(test_data)
    # print(pred)

    # print(test_data.head())
    # print(train_data.head())
    # with pd.option_context('display.max_rows',
    #                        None,
    #                        'display.max_columns',
    #                        None):
    print(f"label & 0 & 1 \\\\")
    for i in target_features:
        shape_analysis(data, i)

    # print(f"Error: {mean_squared_error(test_data[{'retweet'}], pred)}")
    # print(np.median(pred))


def main() -> None:
    test()


if __name__ == "__main__":
    main()
