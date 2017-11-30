import os
from inkml_Interop import parseItem
import numpy as np
import tensorflow as tf


def start_training(folder):
    #Some constants
    #Real Point Weight (0-1) - The weight of the actual given points from the dataset
    real_point_weight = 1.0
    #Calculated Point Weight (0-1) - The weight of the interpolation points between 2 points (not given by the dataset)
    calculated_point_weight = 0.5

    #Load training data
    #Data is an array of function objects, which have 2 paremeters
    #Fullname (the full equation) and
    #segments which are objects that hold the
    #name of the segment and
    #the array of pixels that make up the image of that segment
    data = []
    for file in os.listdir(folder):
        if ".inkml" in file:
            tempStr = os.path.join(folder, file)
            #Parse item takes in the filename (full path from program dir) and
            #it takes a bool that determins whether a line should be drawn between the points given
            #This may or may not help the network
            data.append(parseItem(tempStr, real_point_weight, calculated_point_weight, True))



    #Load Neural Network
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)