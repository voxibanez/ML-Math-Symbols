import os
from inkml_Interop import parseItem
import numpy as np
import tensorflow as tf


def start_training(folder):
    #Load training data
    data = []
    for file in os.listdir(folder):
        if ".inkml" in file:
            tempStr = os.path.join(folder, file)
            #Parse item takes in the filename (full path from program dir) and
            #it takes a bool that determins whether a line should be drawn between the points given
            #This may or may not help the network
            data.append(parseItem(tempStr, True))



    #Load Neural Network
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)