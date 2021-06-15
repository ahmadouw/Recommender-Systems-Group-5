import dataprep
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import precision_score


def get_data(engagement_type):
    data = dataprep.import_data("../shared/data/project/training/one_hour", limit_dataset=False)
    data = data.loc[:, ["text_tokens", "hashtags", "language", engagement_type]]

    # one hot encode text-tokens -- pd.Series not very efficient
    # TODO check if better onehot with pd.getdummydata
    text_tokens = data.pop("text_tokens")
    text_tokens = text_tokens.apply(pd.Series)
    data = text_tokens.join(data)

    # better strategy?
    data = data.fillna(0)

    train_data_int, test_data_int = train_test_split(data, test_size=0.2, shuffle=False)
    return train_data_int, test_data_int


def reply_create_model(data, target_name):
    feature_data = data.copy()
    target_data = feature_data.pop(target_name)

    regr = svm.LinearSVC(max_iter=2000)
    regr.fit(feature_data, target_data)
    return regr


def reply_pred_model(model, target_data, target_name):
    target_data = target_data.copy()
    target_data.pop(target_name)

    prediction = model.predict(target_data)
    return prediction


# test it
target = "reply"
train_data, test_data = get_data(target)

# SVR
model_1 = reply_create_model(train_data, target)
pred = reply_pred_model(model_1, test_data, target)
precision = precision_score(test_data[target], pred)
print("precision: ", precision)
