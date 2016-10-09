import tensorflow as tf
import numpy
import os
import sys
import random
from glob import glob
from PIL import Image


def train(datapath, num_train_steps, samples_per_step):
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
    for i in range(num_train_steps):
        batch_samples = []
        batch_labels = []
        # 
        for class_no in range(num_classes):
            class_label = [0] * num_classes
            class_label[class_no] = 1
            classname = classnames[class_no]
            num_samples = get_class_sample_quantity(training_dir + classname + '/')
            indices = numpy.arange(num_samples)
            numpy.random.shuffle(indices)
            for j in range(samples_per_step):
                spectro_file = training_dir + classname + '/' + classname + '_' + str(j) + '.png'
                spectro_flat = flatten_image(spectro_file)
                batch_samples.append(spectro_flat)
                batch_labels.append(class_label)

        sess.run(train_step, feed_dict={x: batch_samples, y_batch_labels})


def get_data_info(datapath):
    classnames = os.walk(datapath).next()[1]
    first_classname = classnames[0]
    spectro_width, spectro_height = Image.open(datapath + first_classname + '/' + first_classname + "_0.png").size
    # verify that spectrograms are (probably) the same size
    for classname in classnames:
        num_items = len([name for name in os.listdir(datapath + classname) if os.path.isfile(name)])
        rand_item = random.randrange(0, num_items)
        width, height = Image.open(datapath + classname + '/' + classname + "_" + str(rand_item) + ".png").size
        assert (width == spectro_width and height == spectro_height), "Image size error: Found two spectrograms with differing dimensions"
    # 4 values per pixel (red, green, blue, opacity)
    tensor_size = 4 * spectro_width * spectro_height
    num_classes = len(classnames)
    return tensor_size, classnames, num_classes


def get_class_sample_quantity(class_dir):
    """
    Note: the path (datapath) passed to this function must specify
    (include) the testing or training directory at the end.
    """
    num_samples = len(glob(class_dir + '/*.png'))
    return num_samples


def flatten_image(filepath):
    im = Image.open(filepath)
    data = numpy.array(im)
    flat = data.flatten()
    # tensor placeholder will expect floats
    flat = spectro_flat.astype(numpy.float32)
    flat = numpy.multiply(spectro_flat, 1.0 / 255.0)
    return flat


if __name__ == "__main__":
    train_path = sys.argv[1]
    num_train_steps = sys.argv[2]
    samples_per_loop = sys.argv[3]
    train(train_path, num_train_steps, samples_per_step)
    test_path = sys.argv[1] + "testing/"
    test(test_path)

