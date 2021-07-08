import dataprep
import os

from sklearn.ensemble import GradientBoostingRegressor
from joblib import dump, load

used_features = ["tweet_type", "language", "hashtags", "present_media", "present_links", "present_domains", "text_tokens"]
target_features = ["retweet", "reply", "like", "retweet_with_comment"]
data = dataprep.import_data(source_features=used_features, target_features=target_features)

model_reply_path = "model_content/model_reply.joblib"
model_retweet_path = "model_content/model_retweet.joblib"
model_quote_path = "model_content/model_quote.joblib"
model_fav_path = "model_content/model_fav.joblib"


def create_model(train_data, target_name):
    feature_data = train_data.copy()

    # dependent Y => to be predicted
    target_data = train_data.loc[:, target_name]

    #print(f"training Data for ${target_name}: \n", target_data.shape)
    #print(train_data.head())

    # independent X
    feature_data = feature_data.drop(labels=target_features, axis=1, inplace=False)

    regr = GradientBoostingRegressor(random_state=1)
    regr.fit(feature_data, target_data.values.ravel())
    return regr


def export_model_files():
    model_reply_new = create_model(data, ["reply"])
    dump(model_reply_new, model_reply_path)

    model_retweet_new = create_model(data, ["retweet"])
    dump(model_retweet_new, model_retweet_path)

    model_quote_new = create_model(data, ["retweet_with_comment"])
    dump(model_quote_new, model_quote_path)

    model_fav_new = create_model(data, ["like"])
    dump(model_fav_new, model_fav_path)


#export_model_files()

if os.path.isfile(model_reply_path):
    model_reply = load(model_reply_path)
else:
    model_reply = create_model(data, ["reply"])
    dump(model_reply, model_reply_path)

if os.path.isfile(model_retweet_path):
    model_retweet = load(model_retweet_path)
else:
    model_retweet = create_model(data, ["retweet"])
    dump(model_retweet, model_retweet_path)

if os.path.isfile(model_quote_path):
    model_quote = load(model_quote_path)
else:
    model_quote = create_model(data, ["retweet_with_comment"])
    dump(model_quote, model_quote_path)

if os.path.isfile(model_fav_path):
    model_fav = load(model_fav_path)
else:
    model_fav = create_model(data, ["like"])
    dump(model_fav, model_fav_path)


def prepare_input_features(input_features):
    input_features = input_features.loc[:, used_features]
    return dataprep.transform_data(input_features)


def reply_pred_model(input_features):
    model = model_reply
    prediction = model.predict(prepare_input_features(input_features))
    return prediction


def retweet_pred_model(input_features):
    model = model_retweet
    prediction = model.predict(prepare_input_features(input_features))
    return prediction


def quote_pred_model(input_features):
    model = model_quote
    prediction = model.predict(prepare_input_features(input_features))
    return prediction


def fav_pred_model(input_features):
    model = model_fav
    prediction = model.predict(prepare_input_features(input_features))
    return prediction


# def test(target_data):
#
#     # create new test set, drop labels to be predicted
#     target_data = target_data.copy()
#     ground_truth = target_data.copy()
#     target_data = target_data.drop(labels=["reply", "retweet", "retweet_with_comment", "like"], axis=1, inplace=False)
#
#     # get prediction for "reply"
#     prediction_reply = reply_pred_model(target_data)
#     mse_reply = mean_squared_error(ground_truth["reply"], prediction_reply, squared=False)
#     print("RMSE Reply: ", mse_reply)
#
#     prediction_retweet = retweet_pred_model(target_data)
#     mse_retweet = mean_squared_error(ground_truth["retweet"], prediction_retweet, squared=False)
#     print("RMSE Retweet: ", mse_retweet)
#
#     prediction_quote = quote_pred_model(target_data)
#     mse_quote = mean_squared_error(ground_truth["retweet_with_comment"], prediction_quote, squared=False)
#     print("RMSE Quote: ", mse_quote)
#
#     prediction_fav = fav_pred_model(target_data)
#     mse_fav = mean_squared_error(ground_truth["like"], prediction_fav, squared=False)
#     print("RMSE Fav: ", mse_fav)
