import tensorflow as tf
import numpy
import os
import sys
import random
from PIL import Image


def train(datapath):
    training_dir = datapath + "training/"
    
    x = tf.placeholder(tf.float32, [None, ])


def get_spectrogram_sizes(datapath):

    dirs = os.walk(datapath).next()[1]
    spectro_width, spectro_height = Image.open(datapath + dirs[0] + '/' + dirs[0] + "_0.png").size

    # verify that spectrograms are (probably) the same size
    for directory in dirs:
        num_items = len([name for name in os.listdir(datapath + directory) if os.path.isfile(name)])
        rand_item = random.randrange(0, num_items)
        width, height = Image.open(datapath + directory + '/' + directory + "_" + str(rand_item) + ".png").size
        assert (width == spectro_width and height == spectro_height), "Image size error: Found two spectrograms with differing dimensions"

    # 4 values per pixel (red, green, blue, opacity)
    return 4 * spectro_width * spectro_height


if __name__ == "__main__":
    train_path = sys.argv[1]
    num_loops = sys.argv[2]
    samples_per_loop = sys.argv[3]
    train(train_path, num_loops, samples_per_loop)

    test_path = sys.argv[1] + "testing/"
    test(test_path)
