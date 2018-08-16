import pandas as pd
from keras.models import Sequential
from keras.layers import *
import keras.callbacks as cb
import tensorflow as tf

training_data_df = pd.read_csv("sales_data_training_scaled.csv")

X = training_data_df.drop('total_earnings', axis=1).values
Y = training_data_df[['total_earnings']].values

# Define the model
model = Sequential()
#initial model by declaring a new sequential object
model.add(Dense(50, input_dim=9, activation='relu', name='layer_1'))
#create the first layer
#use densely to connected layers
#50 nodes for first layer
#input_dim: how many feature of dataset?
#activation= relu: rectified linear unit, it will apply all layer except last layer, relu will let us model more complex non-linear functions
model.add(Dense(100,activation = 'relu', name='layer_2'))
#add second layer
model.add(Dense(50, activation='relu', name='layer_3'))
#add third layer
model.add(Dense(1, activation='linear', name='output_layer'))
#last layer, because result is single number so Dense(1, activation = 'linear'
model.compile(loss='mean_squared_error', optimizer='adam')
#compile model, need register loss function, optimization algorithm
#mean_squared_error: công thức tính loss function rồi bình phương, or can write 'MSE'
#adam: good choice in this case


# Create a TensorBoard logger
RUN_NAME="run 1 with 50 nodes"
#see 2 graph same time, rename folder inside
logger = cb.TensorBoard(
    log_dir='logs/{}'.format(RUN_NAME),
    write_graph = True,
    histogram_freq = 0
    #every five passes through the training data, i will log out statistics
)
#training
model.fit(
    X,
    Y,
    epochs = 50,
    #how many cicle want to train
    shuffle = True,
    #xáo trộn hay không
    verbose = 2,
    #just see more details information during training
    callbacks=[logger]
    #call tensorboard logger
)
#-----------------------
#test model


# Load the separate test data set
test_data_df = pd.read_csv("sales_data_test_scaled.csv")

X_test = test_data_df.drop('total_earnings', axis=1).values
Y_test = test_data_df[['total_earnings']].values

test_error_rate = model.evaluate(X_test,Y_test, verbose = 0)
#đáng giá tỉ lệ lỗi
print("The mean squared error (MSE) for the test data set is: {:5}".format(test_error_rate))

#---------------
#save and load model

# Save the model to disk
#model.save("trained_model.h5")
#print("Model saved to disk")

#load model from disk
#model =load_model('trained_model.h5')



#----------------------
#predit model


# Load the data we make to use to make a prediction
X = pd.read_csv("proposed_new_product.csv").values
# Make a prediction with the neural network
prediction =model.predict(X)

# Grab just the first element of the first prediction (since that's the only have one)
prediction = prediction[0][0]

# Re-scale the data from the 0-to-1 range back to dollars
# These constants are from when the data was originally scaled down to the 0-to-1 range
prediction = prediction + 0.1159
prediction = prediction / 0.0000036968

print("Earnings Prediction for Proposed Product - ${}".format(prediction))

#-----------------------
#export model to tensorflow

#let's us save a tensorflow of custom options
#"exported_model": name of folder saved model
model_builder = tf.saved_model.builder.SavedModelBuilder("exported_model")

#let tensorflow know which input and which output
inputs = {
    'input': tf.saved_model.utils.build_tensor_info(model.input)
}
outputs = {
    'earnings': tf.saved_model.utils.build_tensor_info(model.output)
}

signature_def = tf.saved_model.signature_def_utils.build_signature_def(
    inputs=inputs,
    outputs=outputs,
    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
)
#we want to save both structure of model and current weight of model
model_builder.add_meta_graph_and_variables(
    K.get_session(),
    tags=[tf.saved_model.tag_constants.SERVING],
    signature_def_map={
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
    }
)

model_builder.save()
