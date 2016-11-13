import tensorflow as tf
import numpy
import os
import sys
from PIL import Image
from ConfigParser import ConfigParser

"""
This script is used to load previously trained TF models and allow them to
    evaluate real data.

Usage: TODO
"""

def ask(model_path):
    tensor_size, num_classes, label_dict = get_config_data(model_path)
    W = tf.Variable(tf.zeros([tensor_size, num_classes]))
    b = tf.Variable(tf.zeros([num_classes]))
    sess = tf.Session()

    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    x = tf.placeholder(tf.float32, [None, tensor_size])
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    prediction = tf.argmax(y,1)

    test_images = get_test_images()

    predicted = prediction.eval(session=sess, feed_dict={x: test_images})
    print predicted
    # print label_dict[tuple(predicted)]


def get_config_data(model_path):
    config = ConfigParser()
    config.read(model_path + "-config.ini")
    tensor_size = int(config.get('Sizes', 'tensor_size'))
    num_classes = int(config.get('Sizes', 'num_classes'))
    classnames_str = config.get('Classnames', 'classnames')
    classnames = classnames_str.split(',')
    label_dict = {}
    for class_no in range(num_classes):
        class_label = [0] * num_classes
        class_label[class_no] = 1
        classname = classnames[class_no]
        label_dict[tuple(class_label)] = classname
    return tensor_size, num_classes, label_dict


def get_test_images():
    samples = []
    im = '/Users/austindunn/Code/mumt502/samples/clinton_v_trump_v2/testing/trump/2559.png'
    flat = flatten_image(im)
    samples.append(flat)
    return samples


def flatten_image(filepath):
    im = Image.open(filepath)
    data = numpy.array(im)
    flat = data.flatten()
    # tensor placeholder will expect floats
    flat = flat.astype(numpy.float32)
    flat = numpy.multiply(flat, 1.0 / 255.0)
    return flat


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Incorrect usage, please see top of file."
        exit()
    model_path = sys.argv[1]
    ask(model_path)
