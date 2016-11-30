import tensorflow as tf
import numpy
import os
import sys
from glob import glob
from PIL import Image
from ConfigParser import ConfigParser

"""
This script is used to load previously trained TF models and allow them to
    evaluate real data.

Usage: TODO
"""

def ask(model_path, test_path, num_test_samples):
    tensor_size, num_classes, classnames = get_config_data(model_path)
    x = tf.placeholder(tf.float32, [None, tensor_size])
    W = tf.Variable(tf.zeros([tensor_size, num_classes]), name="weights")
    b = tf.Variable(tf.zeros([num_classes]), name="bias")
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    prediction = tf.argmax(y,1)
    sess = tf.Session()

    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    # variables for creating confusion matrix
    # first name is correct, second name is the prediction by the model
    confusion_dict = {
        'hillary_trump': 0,
        'hillary_lester': 0,
        'hillary_hillary': 0,
        'lester_trump': 0,
        'lester_lester': 0,
        'lester_hillary': 0,
        'trump_trump': 0,
        'trump_lester': 0,
        'trump_hillary': 0
    }

    for class_no in range(len(classnames)):
        print 'Looking at files for class: ' + classnames[class_no]
        indices = numpy.arange(len(glob(test_path + classnames[class_no] + '/*.png')))
        numpy.random.shuffle(indices)
        # numpy.random.shuffle(indices)
        class_dir = test_path + classnames[class_no] + '/'
        total_test_samples = len(glob(class_dir + '*.png'))
        samples_to_get = num_test_samples if (num_test_samples < total_test_samples) else total_test_samples
        # each sample in the class
        for sample in range(samples_to_get):
            im = class_dir + str(indices[sample]) + '.png'
            flat = flatten_image(im)
            predicted = prediction.eval(session=sess, feed_dict={x: [flat]})
            confusion_key = classnames[class_no] + '_' + classnames[predicted[0]]
            confusion_dict[confusion_key] += 1
            print classnames[predicted[0]]
        print confusion_dict
    print "Here's the confusion dictionary: first name is the correct answer, second is the predictino made by the model."
    print confusion_dict


def get_config_data(model_path):
    config = ConfigParser()
    config.read(model_path + '-config.ini')
    tensor_size = int(config.get('Sizes', 'tensor_size'))
    num_classes = int(config.get('Sizes', 'num_classes'))
    classnames_str = config.get('Classnames', 'classnames')
    classnames = classnames_str.split(',')
    return tensor_size, num_classes, classnames


def flatten_image(filepath):
    im = Image.open(filepath)
    data = numpy.array(im)
    flat = data.flatten()
    # tensor placeholder will expect floats
    flat = flat.astype(numpy.float32)
    flat = numpy.multiply(flat, 1.0 / 255.0)
    return flat


if __name__ == "__main__":
    print len(sys.argv)
    if len(sys.argv) != 4:
        print "Incorrect usage, please see top of file."
        exit()
    model_path = sys.argv[1]
    test_path = sys.argv[2]
    num_test_samples = int(sys.argv[3])
    ask(model_path, test_path, num_test_samples)
