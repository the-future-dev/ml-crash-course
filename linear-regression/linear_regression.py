import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

def build_model(learning_rate):
    """
    Simple linear regression model
    """
    
    model = tf.keras.models.Sequential()
    
    #model topography -> single node, single layer
    model.add(tf.keras.layers.Dense(units=1, 
                                  input_shape=(1,)))
    
    #compile topography
    # Training aims to minimize the model's mean squared error.
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate = learning_rate),
                 loss= "mean_squared_error",
                 metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    return model

def train_model_synthetic(model, feature, label, epochs, batch_size):
    """
    Train the model by feeding it data.
    """
    
    history = model.fit(x=feature,
                        y= label,
                        batch_size=batch_size,
                        epochs=epochs)

    # Gather the trained model's weight and bias.
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]
    
    # List of epochs
    epochs = history.epoch
    
    # Gather the history (a snapshot) of each epoch.
    hist = pd.DataFrame(history.history)
    
    # Specifically gather the model's root mean squared error at each epoch.
    rmse = hist["root_mean_squared_error"]
    
    return trained_weight, trained_bias, epochs, rmse

def train_model_real_dataset(model, df, feature, label, epochs, batch_size):
    """
    Train the model by feeding it data.
    """

    history = model.fit(x=df[feature],
                        y=df[label],
                        batch_size=batch_size,
                        epochs=epochs)
    
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]
    return trained_weight, trained_bias, epochs, rmse


print("Defined build_model and train_model")
