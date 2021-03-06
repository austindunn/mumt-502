import tensorflow as tf
import numpy
import os
import sys
import random
from glob import glob
from PIL import Image

"""
This script contains methods for training and testing a tensorflow model
using spectrogram data as input.

Usage: python train.py [filepath] [train loops] [train samples] [test loops] [test samples] [layers]
where 
[filepath] is the location of the data
[train loops] is the number of times to run the training loop
[samples per loop] is the number of samples of each class to be fed to
    the model every iteration of the training loop
[test loops] is the number of times to run the test loop
[test samples] is the number of samples of each class to be fed to
    the model every iteration of the testing loop
[layers] is the number of layers that the model should use (i.e. depth)
"""


def train_and_test(datapath, train_loops, train_samples, test_loops, test_samples, num_layers):

    # set up our model
    training_dir = datapath + "training/"
    width, height, tensor_size, classnames, num_classes = get_data_info(training_dir)
    x = tf.placeholder(tf.float32, [None, tensor_size])
    W = tf.Variable(tf.zeros([tensor_size, num_classes]))
    b = tf.Variable(tf.zeros([num_classes]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, num_classes])

    # Additional tensors, convolutions and pooling according to TF Deep MNIST Tutorial.
    # Note: inputs assumed to be PNGs with 4 values per pixel -- values adjusted
    # accordingly from TF Deep MNIST Tutorial

    x_image = tf.reshape(x, [-1, width, height, 4])
    layers = []
    patch_size = 5
    num_inputs = 4
    num_features = 32
    # first layer
    crt_layer = create_conv_layer(patch_size, num_inputs, num_features, x_image)
    layers.append(crt_layer)

    # loop to create convolution/pooling layers
    for lyr in range(num_layers-1):
        num_inputs = num_features           # inputs will be # outputs of previous layer
        num_features = num_features * 2     # doubling features calculated is consistent with TF Deep MNIST tutorial conventions
        crt_layer = create_conv_layer(patch_size, num_inputs, num_features, layers[lyr])
        layers.append(crt_layer)

    dim_reduce_factor = 2 ** num_layers
    # densley connected layer
    W_fc1 = weight_variable([(width/dim_reduce_factor) * (height/dim_reduce_factor) * num_features, 256 * dim_reduce_factor])
    b_fc1 = bias_variable([256 * dim_reduce_factor])
    last_layer_flat = tf.reshape(layers[-1], [-1, (width/dim_reduce_factor) * (height/dim_reduce_factor) * num_features])
    h_fc1 = tf.nn.relu(tf.matmul(last_layer_flat, W_fc1) + b_fc1)
    # apply dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # softmax readout layer
    W_fc2 = weight_variable([256 * dim_reduce_factor, num_classes])
    b_fc2 = bias_variable([num_classes])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # in each training step, ...
    for step in range(train_loops):
        print "starting step " + str(step+1) + "..."
        batch_samples = []
        batch_labels = []
        # each class is set up with its label and correct number of training samples, and...
        for class_no in range(num_classes):
            class_label = [0] * num_classes
            class_label[class_no] = 1
            classname = classnames[class_no]
            num_samples = len(glob(training_dir + classname + '/*.png'))
            indices = numpy.arange(num_samples)
            numpy.random.shuffle(indices)
            # a random sample is selected and added to the array of samples in the batch
            for sample in range(train_samples):
                im = training_dir + classname + '/' + str(indices[sample]) + '.png'
                flat = flatten_image(im)
                batch_samples.append(flat)
                batch_labels.append(class_label)
        if (step % 5 == 0 and step != 0):
            train_accuracy = accuracy.eval(session=sess, feed_dict={x:batch_samples, y_: batch_labels, keep_prob: 1.0})
            print 'Step: ' + str(step+1) + ', training accuracy: ' + str(train_accuracy)
        train_step.run(session=sess, feed_dict={x: batch_samples, y_: batch_labels, keep_prob: 0.5})

    # test accuracy of the model once trained
    accuracies = []
    for test in range(test_loops):
        testing_data, testing_labels = get_test_data(datapath, classnames, test_samples)
        success_rate = sess.run(accuracy, feed_dict={x: testing_data, y_: testing_labels, keep_prob: 1.0})
        print 'Test number: ' + str(test+1) + '. Success rate: ' + str(success_rate*100) + '%'
        accuracies.append(success_rate)
    avg = numpy.mean(accuracies)
    print 'This model tested with an average success rate of ' + str(avg*100) + '%'


def get_data_info(datapath):
    classnames = os.walk(datapath).next()[1]
    first_classname = classnames[0]
    width, height = Image.open(datapath + first_classname + '/' + '0.png').size
    # verify that spectrograms are (probably) all the same size
    for classname in classnames:
        num_items = len(glob(datapath + classname + '/*.png'))
        rand_item = random.randrange(0, num_items)
        crt_width, crt_height = Image.open(datapath + classname + '/' + str(rand_item) + ".png").size
        assert (crt_width == width and crt_height == height), "Image size error: Found two spectrograms with differing dimensions"
    # 4 values per pixel (red, green, blue, opacity)
    tensor_size = 4 * width * height
    num_classes = len(classnames)
    return width, height, tensor_size, classnames, num_classes


def get_test_data(datapath, classnames, test_samples):
    testing_dir = datapath + 'testing/'
    testing_data = []
    testing_labels = []
    # each class...
    for class_no in range(len(classnames)):
        class_label = [0] * len(classnames)
        class_label[class_no] = 1
        # randomize test samples chosen
        indices = numpy.arange(len(glob(testing_dir + classnames[class_no] + '/*.png')))
        numpy.random.shuffle(indices)
        class_dir = testing_dir + classnames[class_no] + '/'
        total_test_samples = len(glob(class_dir + '*.png'))
        samples_to_get = test_samples if (test_samples < total_test_samples) else total_test_samples
        # each sample in the class
        for sample in range(samples_to_get):
            im = class_dir + str(indices[sample]) + '.png'
            flat = flatten_image(im)
            testing_data.append(flat)
            testing_labels.append(class_label)
    return testing_data, testing_labels


def flatten_image(filepath):
    im = Image.open(filepath)
    data = numpy.array(im)
    flat = data.flatten()
    # tensor placeholder will expect floats
    flat = flat.astype(numpy.float32)
    flat = numpy.multiply(flat, 1.0 / 255.0)
    return flat


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def create_conv_layer(patch_size, num_channels, num_outputs, input_tensor):
    W = weight_variable([patch_size, patch_size, num_channels, num_outputs])
    b = bias_variable([num_outputs])
    h_conv = tf.nn.relu(conv2d(input_tensor, W) + b)
    h_pool = max_pool_2x2(h_conv)
    return h_pool


if __name__ == "__main__":
    # make sure correct number of arguments
    if len(sys.argv) < 5:
        print "Incorrect usage"
        exit()
    train_path = sys.argv[1]
    train_loops = sys.argv[2]
    train_samples = sys.argv[3]
    test_loops = sys.argv[4]
    test_samples = sys.argv[5]
    num_layers = sys.argv[6]
    train_and_test(train_path, int(train_loops), int(train_samples), int(test_loops), int(test_samples), int(num_layers))

