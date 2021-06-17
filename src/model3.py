import dataprep
import pandas as pd

from sklearn import svm
from sklearn.metrics import precision_score

used_features = ["text_tokens", "language"]
target_features = ["retweet"]
data = dataprep.import_data(source_features=used_features, target_features=target_features, nrows=3000)
train_data, test_data = dataprep.split_train_test(data)


def reply_create_model(data, target_name):
    feature_data = data.copy()
    target_data = data.loc[:, target_name]
    feature_data = feature_data.drop(labels=target_name, axis=1, inplace=False)

    regr = svm.LinearSVC(max_iter=2000)
    regr.fit(feature_data, target_data.values.ravel())
    return regr


def reply_pred_model(model, target_data, target_name):
    target_data = target_data.copy()
    target_data = target_data.drop(labels=target_name, axis=1, inplace=False)

    prediction = model.predict(target_data)
    return prediction


# test it
print("data: ", data.shape)
print("test_data: ", test_data.shape)
print("train_data: ", train_data.shape)
print("test_data \n", test_data)

sub_1 = data.loc[data['retweet'] == 1]
print("1: ", sub_1.shape[0])
sub_0 = data.loc[data['retweet'] == 0]
print("0: ", sub_0.shape[0])

# LinearSVC
model_1 = reply_create_model(train_data, target_features)
pred = reply_pred_model(model_1, test_data, target_features)
print(pred)
precision = precision_score(test_data[target_features], pred)
print("precision: ", precision)
