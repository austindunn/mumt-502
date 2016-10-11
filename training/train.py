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

Usage: python train.py [filepath] [train steps] [samples per loop]
where [filepath] is the location of the data
[train steps] is the number of times to run the training loop
[samples per loop] is the number of samples of each class to be fed to
    the model every iteration of the training loop
"""


def train_and_test(datapath, num_train_steps, samples_per_step, num_test_samples):
    # set up our model
    training_dir = datapath + "training/"
    tensor_size, classnames, num_classes = get_data_info(training_dir)
    x = tf.placeholder(tf.float32, [None, tensor_size])
    W = tf.Variable(tf.zeros([tensor_size, num_classes]))
    b = tf.Variable(tf.zeros([num_classes]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, num_classes])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    # in each training step, ...
    for step in range(num_train_steps):
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
            for sample in range(samples_per_step):
                spectro_file = training_dir + classname + '/' + str(sample) + '.png'
                spectro_flat = flatten_image(spectro_file)
                batch_samples.append(spectro_flat)
                batch_labels.append(class_label)

        sess.run(train_step, feed_dict={x: batch_samples, y_: batch_labels})

    # test accuracy of the model once trained
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # training_data, training_labels = get_test_data(datapath, classnames, num_test_samples)


def get_data_info(datapath):
    classnames = os.walk(datapath).next()[1]
    first_classname = classnames[0]
    spectro_width, spectro_height = Image.open(datapath + first_classname + '/' + '0.png').size
    # verify that spectrograms are (probably) the same size
    for classname in classnames:
        print datapath + classname
        num_items = len(glob(datapath + classname + '/*.png'))
        rand_item = random.randrange(0, num_items)
        width, height = Image.open(datapath + classname + '/' + str(rand_item) + ".png").size
        assert (width == spectro_width and height == spectro_height), "Image size error: Found two spectrograms with differing dimensions"
    # 4 values per pixel (red, green, blue, opacity)
    tensor_size = 4 * spectro_width * spectro_height
    num_classes = len(classnames)
    return tensor_size, classnames, num_classes


def get_test_data(datapath, classnames, num_test_samples):
    testing_dir = datapath + 'testing/'
    testing_data = []
    testing_labels = []
    #for clas in classnames:
        #do stuff


def flatten_image(filepath):
    im = Image.open(filepath)
    data = numpy.array(im)
    flat = data.flatten()
    # tensor placeholder will expect floats
    flat = flat.astype(numpy.float32)
    flat = numpy.multiply(flat, 1.0 / 255.0)
    return flat


if __name__ == "__main__":
    # make sure correct number of arguments
    if len(sys.argv) < 5:
        print "Incorrect usage"
        exit()
    train_path = sys.argv[1]
    num_train_steps = sys.argv[2]
    samples_per_step = sys.argv[3]
    num_test_samples = sys.argv[4]
    train_and_test(train_path, int(num_train_steps), int(samples_per_step), int(num_test_samples))
    #test_path = sys.argv[1] + "testing/"
    #test(test_path)

