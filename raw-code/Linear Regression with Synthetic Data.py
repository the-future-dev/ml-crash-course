#!/usr/bin/env python
# coding: utf-8

# # Simple Linear Regression with Synthetic Data
# 
# First exercise: exploring linear regression with a simple database. 
# 
#   * Tune the following [hyperparameters](https://developers.google.com/machine-learning/glossary/#hyperparameter):
#     * [learning rate](https://developers.google.com/machine-learning/glossary/#learning_rate)
#     * number of [epochs](https://developers.google.com/machine-learning/glossary/#epoch)
#     * [batch size](https://developers.google.com/machine-learning/glossary/#batch_size)
#   * Interpret different kinds of [loss curves](https://developers.google.com/machine-learning/glossary/#loss_curve).
# 

# #### Imports

# In[1]:


import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt


# ### Define Model for Linear Regression

# In[2]:


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
    model.compile(optimizer=tf.keras.optimizers.experimental.RMSprop(learning_rate = learning_rate),
                 loss= "mean_squared_error",
                 metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    return model

def train_model(model, feature, label, epochs, batch_size):
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

print("Defined build_model and train_model")


# ### Define The Plotting Functions

# In[3]:


def plot_the_model(trained_weight, trained_bias, feature, label):
    """
    Plot the training feature and label.
    """
    if isinstance(feature, pd.Series) or isinstance(feature, pd.DataFrame):
        feature = feature.dropna().values
    if isinstance(label, pd.Series) or isinstance(label, pd.DataFrame):
        label = label.dropna().values

    plt.xlabel("feature")
    plt.ylabel("label")
    
    plt.scatter(feature, label)
    
    # Creating lines representing the model
    x0 = 0
    y0 = trained_bias
    x1 = feature[-1]
    y1 = trained_bias + (trained_weight * x1)
    
    plt.plot([x0, x1], [y0, y1], c='r')
    
    plt.show()
    

def plot_the_loss_curve(epochs, rmse):
    """
    Plot the loss curve, which shows loss vs. epoch.
    """
    
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")
    
    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plt.ylim([rmse.min()*0.97, rmse.max()])
    plt.show()

print("Defined the plot_the_model and plot_the_loss_curve functions.")


# ### Define the dataset

# In[4]:


my_feature = ([1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0, 12.0])
my_label   = ([5.0, 8.8,  9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2])


# ### Specify The Hyperparameters:
#   * [learning rate](https://developers.google.com/machine-learning/glossary/#learning_rate)
#   * [epochs](https://developers.google.com/machine-learning/glossary/#epoch)
#   * [batch_size](https://developers.google.com/machine-learning/glossary/#batch_size)
# 
# The following code cell initializes these hyperparameters and then invokes the functions that build and train the model.

# In[ ]:


learning_rate=0.14
epochs=10
my_batch_size=12

my_model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, my_feature, 
                                                         my_label, epochs,
                                                         my_batch_size)
plot_the_model(trained_weight, trained_bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)

