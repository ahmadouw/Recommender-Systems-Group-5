# -*- coding: utf-8 -*-
#
# Created on 2021-06-19 at 15:03.
# Created by andreasmerckel (coffeecodecruncher@gmail.com)
#
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import scikitplot as skplt
import numpy as np
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import (
    classification_report,
    mean_squared_error, plot_confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import (
    MLPRegressor
)
from sklearn.preprocessing import StandardScaler
# from xgboost import XGBRegressor

import dataprep


def reply_pred_model(target_data):
    with open("out/nn_model.pickle", "rb") as f:
        model = pickle.load(f)
        pred = model.predict(target_data)
        return np.clip(pred, 0, 1)
        #return pred


def retweet_pred_model(input_features):
    # TODO fill in your implementation of the model
    raise NotImplementedError


def quote_pred_model(input_features):
    # TODO fill in your implementation of the model
    raise NotImplementedError


def fav_pred_model(input_features):
    # TODO fill in your implementation of the model
    raise NotImplementedError


def init_reply_model_nn(data, target_name) -> None:
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
    with open("out/nn_model.pickle", "wb") as f:
        pickle.dump(nn_model, f)


def init_reply_model_xg(data, target_name) -> None:
    feature_data = data.copy()
    target_data = data.loc[:, target_name]
    feature_data = feature_data.drop(labels=target_name,
                                     axis=1,
                                     inplace=False)

    nn_model = XGBRegressor()

    nn_model.fit(feature_data, target_data.values.ravel())
    with open("out/xg_model.pickle", "wb") as f:
        pickle.dump(nn_model, f)


def rebalancer(df,
               target):
    """
    """
    y = df[target]
    x = df.drop(columns=[target], axis=1)
    scale = StandardScaler()
    x_scaled = scale.fit_transform(x)

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
    plot_confusion_matrix(algo, x_test, y_test)
    plt.show()
    print(f"Classification Report: \n"
          f"{classification_report(y_test, y_test_ypred, digits=3)}")

    fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
    skplt.metrics.plot_roc(y_test, y_probas, figsize=(16, 9))
    skplt.metrics.plot_precision_recall(y_test, y_probas, figsize=(16, 9))
    plt.show()


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
                           None):  # more options can be specified also
        print(df.corr()["retweet"])
    # plt.show()


def test(nrows=50):
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
            # "reply",
            # "like",
            # "retweet_with_comment",
    ]
    # data = preprocessing.import_data(source_features=used_features,
    #                                  target_features=target_features,
    #                                  nrows=nrows)

    # with open("out/out-1.csv", 'w') as f:
    #     print(data.to_string(index=False), file=f)

    data = dataprep.import_data(source_features=used_features,
                                target_features=target_features,
                                nrows=nrows)

    # with open("out/out-2.csv", 'w') as f:
    #     print(data.to_string(index=False), file=f)

    print(data.describe())

    plotter(data,
            target_features,
            used_features,
            title="Before rebalance")
    data = rebalancer(data, target_features[0])
    plotter(data,
            target_features,
            used_features,
            title="After rebalance")
    # print(data.head())
    train_data, test_data = dataprep.split_train_test(data)

    sub_1 = data.loc[data['retweet'] == 1]
    sub_0 = data.loc[data['retweet'] == 0]

    init_reply_model_nn(train_data, target_features)

    nn_input = test_data.drop(labels=target_features, axis=1)

    pred = reply_pred_model(nn_input)

    print(f"1: {sub_1.shape[0]}")
    print(f"0: {sub_0.shape[0]}")
    print(test_data.head())
    print(train_data.head())
    with pd.option_context('display.max_rows',
                           None,
                           'display.max_columns',
                           None):
        print(pred)
    print(f"Error: {mean_squared_error(test_data[target_features], pred)}")
    print(np.median(pred))


def main() -> None:
    test(1000)


if __name__ == "__main__":
    main()
