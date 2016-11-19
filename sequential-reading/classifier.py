import tensorflow as tf
import pylab
import numpy
import sys
import wave
from PIL import Image, ImageChops
from ConfigParser import ConfigParser

"""
This script will be given a file and will classify audio frame-to-frame.

Usage: TODO
"""

def read_and_predict(model_path, wav_path, frame_length):
    tensor_size, num_classes, classnames = get_config_data(model_path)
    x = tf.placeholder(tf.float32, [None, tensor_size])
    W = tf.Variable(tf.zeros([tensor_size, num_classes]), name="weights")
    b = tf.Variable(tf.zeros([num_classes]), name="bias")
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    prediction = tf.argmax(y,1)
    sess = tf.Session()

    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    wav = wave.open(wav_path, 'r')
    num_frames = wav.getnframes()
    sample_rate = wav.getframerate()
    num_windows = num_frames/frame_length
    print 'Starting classification... Examining ' + str(num_windows) + ' windows.'
    while (wav.tell() + frame_length) < num_frames:
        frames = wav.readframes(frame_length)
        sound_info = pylab.fromstring(frames, 'Int16')
        amps = numpy.absolute(sound_info)
        # ignore frames with low amplitudes
        if (amps.mean() < 1000):
            # print get_time_string(sample_rate, wav.tell()) + ' - amplitude too low to classify.'
            continue

        pylab.figure(num=None, figsize=(19, 12))
        pylab.axis('off')
        pylab.specgram(sound_info, NFFT=frame_length, Fs=sample_rate)
        pylab.savefig('crt.png')
        pylab.close()
        spectro = Image.open('crt.png')
        spectro = squarify(spectro, 256)
        flat_spectro = flatten(spectro)

        predicted = prediction.eval(session=sess, feed_dict={x: flat_spectro})
        print get_time_string(sample_rate, wav.tell()) + ' - ' + classnames[predicted[0]]


def get_config_data(model_path):
    config = ConfigParser()
    config.read(model_path + '-config.ini')
    tensor_size = int(config.get('Sizes', 'tensor_size'))
    num_classes = int(config.get('Sizes', 'num_classes'))
    classnames_str = config.get('Classnames', 'classnames')
    classnames = classnames_str.split(',')
    return tensor_size, num_classes, classnames


def flatten(im):
    data = numpy.array(im)
    flat = data.flatten()
    # tensor placeholder will expect floats
    flat = flat.astype(numpy.float32)
    flat = numpy.multiply(flat, 1.0 / 255.0)
    flat_in_array = []
    flat_in_array.append(flat)
    return flat_in_array


def squarify(im, image_size):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        im = im.crop(bbox)
    im = im.resize((image_size, image_size))
    return im


def get_time_string(sample_rate, current_pos):
    if (current_pos % sample_rate == 0):
        milli = 0
    else:
        milli = int((1.0 / (float(sample_rate) / float(current_pos % sample_rate))) * 100.0)
    sec = (current_pos / sample_rate) % 60
    minute = ((current_pos / sample_rate) - sec) / 60

    if (milli < 10):
        milli_str = '00' + str(milli)
    elif (milli < 100):
        milli_str = '0' + str(milli)
    else:
        milli_str = str(milli)
    sec_str = '0' + str(sec) if sec < 10 else str(sec)
    minute_str = '0' + str(minute) if minute < 10 else str(minute)
    
    return minute_str + ':' + sec_str + '.' + milli_str


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Incorrect usage, please see top of file."
        exit()
    model_path = sys.argv[1]
    wav_path = sys.argv[2]
    # frame_length = sys.argv[3]
    read_and_predict(model_path, wav_path, 256)
