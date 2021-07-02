import dataprep

from sklearn.ensemble import GradientBoostingRegressor


used_features = ["tweet_type", "language", "hashtags", "present_media", "present_links", "present_domains", "text_tokens"]
target_features = ["retweet", "reply", "like", "retweet_with_comment"]
data = dataprep.import_data(source_features=used_features, target_features=target_features, nrows=20000)


def create_model(train_data, target_name):
    feature_data = train_data.copy()

    # dependent Y => to be predicted
    target_data = train_data.loc[:, target_name]

    # independent X
    feature_data = feature_data.drop(labels=target_features, axis=1, inplace=False)

    regr = GradientBoostingRegressor(random_state=1)
    regr.fit(feature_data, target_data.values.ravel())
    return regr


def prepare_input_features(input_features):
    input_features = input_features.loc[:, used_features]
    return dataprep.transform_data(input_features)


def reply_pred_model(input_features):
    model = create_model(data, ["reply"])
    prediction = model.predict(prepare_input_features(input_features))
    return prediction


def retweet_pred_model(input_features):
    model = create_model(data, ["retweet"])
    prediction = model.predict(prepare_input_features(input_features))
    return prediction


def quote_pred_model(input_features):
    model = create_model(data, ["retweet_with_comment"])
    prediction = model.predict(prepare_input_features(input_features))
    return prediction


def fav_pred_model(input_features):
    model = create_model(data, ["like"])
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

