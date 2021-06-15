import dataprep
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB


def get_data(engagement_type):
    data = dataprep.import_data("../shared/data/project/training/one_hour", limit_dataset=True)
    data = data.loc[:, ["text_tokens", "hashtags", "language", engagement_type]]
    data = data.fillna(0)

    train_data_int, test_data_int = train_test_split(data, test_size=0.2, shuffle=False)
    return train_data_int, test_data_int


def reply_create_model(data):
    features = data.loc[:, ["language", "text_tokens", "hashtags"]]
    samples = data["reply"].tolist()

    print("features: \n", features)
    print(type(features))

    regr = svm.LinearSVC()
    regr.fit(features, samples)
    return regr


def reply_create_bayes_model(data):
    X = data.loc[:, ["language", "text_tokens", "hashtags"]]
    y = data["reply"].tolist()

    print("features: \n", X)
    print(type(X))

    clf = MultinomialNB()
    clf.fit(X, y)

    return clf


def reply_pred_model(model, test_data):
    prediction = model.predict(test_data.loc[:, ["language", "text_tokens", "hashtags"]])
    return prediction


# test it
train_data, test_data = get_data("reply")
print("Test data set:")
print(test_data)

# SVR
model_1 = reply_create_model(train_data)
pred = reply_pred_model(model_1, test_data)
print("Prediction: \n", pred)


# Naive Bayes
model_2 = reply_create_bayes_model(train_data)
pred_bayes = reply_pred_model(model_2, test_data)
print("Prediction Bayes: \n", pred_bayes)
