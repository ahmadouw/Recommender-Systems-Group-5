import dataprep
import pandas as pd
from sklearn import svm


# use: "Text tokens", "Language", "Hashtags"

def get_data(engagement_type):
    data = dataprep.import_data("../shared/data/project/training/one_hour", limit_dataset=True)
    data = data.loc[:, ["text_tokens", "hashtags", "language", engagement_type]]
    return data


def get_test_data(engagement_type, number):
    data = dataprep.import_data("../shared/data/project/training/one_hour")
    data = data.loc[:, ["text_tokens", "hashtags", "language", engagement_type]]
    return data.iloc[number]


def reply_create_model():
    data = get_data("reply")
    features = data.loc[:, ["language"]]
    samples = data["reply"]

    regr = svm.SVR()
    regr.fit(features, samples)

    return regr


def reply_pred_model(input_features):
    model = reply_create_model()
    prediction = model.predict(input_features)
    return prediction


# test it
test_tweet = get_test_data("reply", 301)
print("=== TEST ===")
print(test_tweet)
print("----------")
pred = reply_pred_model(test_tweet)
print("Input: ", test_tweet, "Prediction:", pred)
