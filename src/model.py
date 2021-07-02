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
    return model3.reply_pred_model(input_features)

def retweet_pred_model(input_features):
    # TODO fill in your implementation of the model
    return np.random.rand()

def quote_pred_model(input_features):
    # TODO fill in your implementation of the model
    return np.random.rand()

def fav_pred_model(input_features):
    # TODO fill in your implementation of the model
    return np.random.rand()


# test the function
## TODO REMOVE!
target_data = dataprep.import_data(nrows=20000, use_transform_data=False)
print(reply_pred_model(target_data))