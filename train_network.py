import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import math
import os
import sys
import input_data

#from sklearn.metrics import confusion_matrix
#from inkml_Interop import *
from target_classes import *

def start_training(folder):
    '''
    #Some constants
    #Real Point Weight (0-1) - The weight of the actual given points from the dataset
    real_point_weight = 1.0
    #Calculated Point Weight (0-1) - The weight of the interpolation points between 2 points (not given by the dataset)
    calculated_point_weight = 1.0

    #Load training data
    #Data is an array of function objects, which have 2 paremeters
    #Fullname (the full equation) and
    #segments which are objects that hold the
    #name of the segment and
    #the array of pixels that make up the image of that segment
    data = []
    i = 0
    for file in os.listdir(folder):
        if ".inkml" in file:
            tempStr = os.path.join(folder, file)
            #Parse item takes in the filename (full path from program dir) and
            #it takes a bool that determines whether a line should be drawn between the points given
            #This may or may not help the network
            data.append(parseItem(tempStr, real_point_weight, calculated_point_weight, i, True))
            i += 1

    target_list = []
    target_list = gen_target_classes(folder, 0)[0]
    picture_data = []
    class_data = []
    for func in data:
        for symb in func.symbols:
            picture_data.append(symb.pictureData)
            for i in range(0,len(target_list)):
                if symb.name == target_list[i]:
                    class_data.append(i)
                    
    print("Size of:")
    #print("- Training-set:\t\t{}".format(len(data.train.labels)))
    #print("- Test-set:\t\t{}".format(len(data.test.labels)))
    #print("- Validation-set:\t{}".format(len(data.validation.labels)))
    '''
    data = input_data.read_data_sets('data/MNIST', one_hot=True)
    neuralNet = neural_net(data, 23)

class neural_net:
    #set up neural network
    # Convolutional Layer 1.
    filter_size1 = 5  # Convolution filters are 5 x 5 pixels.
    num_filters1 = 16  # There are 16 of these filters.

    # Convolutional Layer 2.
    filter_size2 = 5  # Convolution filters are 5 x 5 pixels.
    num_filters2 = 36  # There are 36 of these filters.

    # Fully-connected layer.
    fc_size = 128  # Number of neurons in fully-connected layer.

    # We know that MNIST images are 28 pixels in each dimension.
    img_size = 28

    # Images are stored in one-dimensional arrays of this length.
    img_size_flat = img_size * img_size

    # Tuple with height and width of images used to reshape arrays.
    img_shape = (img_size, img_size)

    # Number of colour channels for the images: 1 channel for gray-scale.
    num_channels = 1

    # Number of classes, one class for each of 10 digits.
    num_classes = 23
    picture_data = []
    class_data = []

    def __init__(self, data, classNum):
        num_classes = classNum
        #picture_data = pictureData
        #class_data = classData

        #Load Neural Network
        #mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        #train_data = mnist.train.images  # Returns np.array
        #train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        #eval_data = mnist.test.images  # Returns np.array
        #eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

        #Make some placeholders
        #First we define the placeholder variable for the input images.
        # This allows us to change the images that are input to the TensorFlow graph.
        # This is a so-called tensor, which just means that it is a multi-dimensional vector or matrix.
        # The data-type is set to float32 and the shape is set to [None, img_size_flat], where None means that the tensor may hold an arbitrary number of images with each image being a vector of length img_size_flat.
        x = tf.placeholder(tf.float32, shape=[None, self.img_size_flat], name='x')

        #The convolutional layers expect x to be encoded as a 4-dim tensor so we have to reshape it so its shape is instead [num_images, img_height, img_width, num_channels].
        # Note that img_height == img_width == img_size and num_images can be inferred automatically by using -1 for the size of the first dimension.
        # So the reshape operation is:
        x_image = tf.reshape(x, [-1, self.img_size, self.img_size, self.num_channels])

        #Next we have the placeholder variable for the true labels associated with the images that were input in the placeholder variable x.
        #  The shape of this placeholder variable is [None, num_classes] which means it may hold an arbitrary number of labels and each label is a vector of length num_classes which is 10 in this case.
        y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

        #We could also have a placeholder variable for the class-number, but we will instead calculate it using argmax.
        # Note that this is a TensorFlow operator so nothing is calculated at this point.
        y_true_cls = tf.argmax(y_true, dimension=1)

        #Create Convolutional Layer 1
        layer_conv1, weights_conv1 = \
            new_conv_layer(input=x_image,
                           num_input_channels=self.num_channels,
                           filter_size=self.filter_size1,
                           num_filters=self.num_filters1,
                           use_pooling=True)

        #Create Convolutional Layer 2
        layer_conv2, weights_conv2 = \
            new_conv_layer(input=layer_conv1,
                           num_input_channels=self.num_filters1,
                           filter_size=self.filter_size2,
                           num_filters=self.num_filters2,
                           use_pooling=True)

                           
        self.layer_flat, self.num_features = flatten_layer(layer_conv2)
        #Create fully connected layer 1
        layer_fc1 = new_fc_layer(input=self.layer_flat,
                                 num_inputs=self.num_features,
                                 num_outputs=self.fc_size,
                                 use_relu=True)

        #create fully connected layer 2
        layer_fc2 = new_fc_layer(input=layer_fc1,
                                 num_inputs=self.fc_size,
                                 num_outputs=num_classes,
                                 use_relu=False)

        #class prediction
        y_pred = tf.nn.softmax(layer_fc2)
        y_pred_cls = tf.argmax(y_pred, dimension=1)

        #NOTE: I stopped at cost-function to be optimized
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                                labels=y_true)
                                                                
        cost = tf.reduce_mean(cross_entropy)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
        
        correct_predition = tf.equal(y_pred_cls, y_true_cls)
        
        accuracy = tf.reduce_mean(tf.cast(correct_predition, tf.float32))
        
        session = tf.Session()
        
        session.run(tf.global_variables_initializer())
        
        train_batch_size = 64

class image:
    def __init__(self,name,pictureData):
        self.name = name
        self.pictureData = pictureData
    name = 0;
    pictureData = []

def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,  # The previous layer.
                   num_input_channels,  # Num. channels in prev. layer.
                   filter_size,  # Width and height of each filter.
                   num_filters,  # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights

#reduce the 4-dim tensor to 2-dim which can be used as input to the fully-connected layer
def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features

#This function creates a new fully-connected layer in the computational graph for TensorFlow
# Nothing is actually calculated here, we are just adding the mathematical formulas to the TensorFlow graph
def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer
    
    
total_iterations = 0
def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations
    global data

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def plot_example_errors(cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.test.cls[incorrect]

    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])

def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = data.test.cls

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# Split the test-set into smaller batches of this size.
test_batch_size = 256

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = len(data.test.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.test.images[i:j, :]

        # Get the associated labels.
        labels = data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.test.cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)
def main():
    start_training("test2")
    
    optimize(num_iterations=99) # We already performed 1 iteration above.



if __name__ == '__main__':
    main()
