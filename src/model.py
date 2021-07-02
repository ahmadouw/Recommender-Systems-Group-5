import numpy as np
import model3
import dataprep


'''
You need to structure your implementation in a way that in this class the prediction
of the different engagement types of your model are called. Please replace the 
random return placeholder with you actual implementation.

You can train different models for each engagement type or you train one which is able
to predicte multiple classes.
'''

def reply_pred_model(input_features):
    # TODO fill in your implementation of the model
    
    #U2U
    #tweet_id = input_features[all_features_to_idx['tweet_id']]
    #user_id = input_features[all_features_to_idx['engaging_user_id']]
    tweet_id = 1
    user_id = 2
    return u2u_reply.predict(user_id, tweet_id)
    #end U2U
    
    return np.random.rand()

def retweet_pred_model(input_features):
    # TODO fill in your implementation of the model
    
    #U2U
    tweet_id = input_features[all_features_to_idx['tweet_id']]
    user_id = input_features[all_features_to_idx['engaging_user_id']]
    return u2u_retweet.predict(user_id, tweet_id)
    #end U2U
    
    return np.random.rand()

def quote_pred_model(input_features):
    # TODO fill in your implementation of the model
    
    #U2U
    tweet_id = input_features[all_features_to_idx['tweet_id']]
    user_id = input_features[all_features_to_idx['engaging_user_id']]
    return u2u_quote.predict(user_id, tweet_id)
    #end U2U
    
    return np.random.rand()

def fav_pred_model(input_features):
    # TODO fill in your implementation of the model
    
    #U2U
    tweet_id = input_features[all_features_to_idx['tweet_id']]
    user_id = input_features[all_features_to_idx['engaging_user_id']]
    return u2u_fav.predict(user_id, tweet_id)
    #end U2U
    
    return np.random.rand()


# test the function
## TODO REMOVE!
target_data = dataprep.import_data(nrows=20000, use_transform_data=False)
print(reply_pred_model(target_data))