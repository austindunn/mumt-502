import tensorflow as tf
import pylab
import numpy
import sys
import wave
from scipy.stats import mode
from collections import deque
from PIL import Image, ImageChops
from ConfigParser import ConfigParser

"""
This script will be given a file and will classify audio frame-to-frame.

Usage: TODO
"""

def read_and_predict(model_path, wav_path, frame_length, deque_size):
    # set up variables for model to be read into
    tensor_size, num_classes, classnames = get_config_data(model_path)
    x = tf.placeholder(tf.float32, [None, tensor_size])
    W = tf.Variable(tf.zeros([tensor_size, num_classes]), name="weights")
    b = tf.Variable(tf.zeros([num_classes]), name="bias")
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    prediction = tf.argmax(y,1)
    sess = tf.Session()

    #restore trained & saved tensorflow model
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    # information about the wav file
    wav = wave.open(wav_path, 'r')
    num_frames = wav.getnframes()
    sample_rate = wav.getframerate()
    num_windows = num_frames/frame_length

    # variables for analysis
    predictions = deque([], deque_size)
    time_per_class = dict((classname, 0) for classname in classnames)
    predictions_this_sec = []
    samples_remaining_this_sec = sample_rate
    frames_examined_this_sec = 0
    crt_sec = 0
    seconds_ignored = 0
    print crt_sec

    print 'Starting classification... Examining ' + str(num_windows) + ' windows.'
    while (wav.tell() + frame_length) < num_frames:
        #print 'current second: ' + str(crt_sec) + ', wav.tell: ' + str(wav.tell()) + ', samples remaining: ' + str(samples_remaining_this_sec) 
        """
        if get_second(sample_rate, wav.tell()) != crt_sec:
            seconds_ignored += 1
            predictions_this_sec = []
            samples_remaining_this_sec = sample_rate
            frames_examined_this_sec = 0
            crt_sec = get_second(sample_rate, wav.tell())
            print crt_sec
        """

        frames = wav.readframes(frame_length)
        sound_info = pylab.fromstring(frames, 'Int16')
        amps = numpy.absolute(sound_info)

        # samples_remaining_this_sec -= frame_length

        # ignore frames with low amplitudes
        if (amps.mean() < 1000):
            continue

        # print 'DING!'

        # frames_examined_this_sec += 1
        flat_spectro = create_flat_spectrogram(sound_info, frame_length, sample_rate)

        predicted = prediction.eval(session=sess, feed_dict={x: flat_spectro})
        predictions.append(predicted[0])
        #predictions_this_sec.append(predicted[0])
        #if len(predictions) > deque_size/2 and True == False:
        if len(predictions) > deque_size/2:
            m = mode(predictions)
            print '------------------------------------------------'
            print get_crt_time_string(sample_rate, wav.tell())
            print 'mode of prediction list: ' + classnames[m.mode[0]]
            print 'predictions list: ' + str(predictions)

        """
        # second-to-second logic 
        if frames_examined_this_sec > 2 and float(mode(predictions_this_sec).count[0]) / float(len(predictions_this_sec)) >= 0.8:
            time_per_class[classnames[mode(predictions_this_sec).mode[0]]] += 1
            print '----------------------------------------'
            print 'second elapsed: ' + str(crt_sec)
            print time_per_class
            # print stats and exit if not enough samples left for another frame
            if wav.tell() + samples_remaining_this_sec >= num_frames:
                output_stats(crt_sec, seconds_ignored, time_per_class)
                return

            if crt_sec > 0 and crt_sec % 60 == 0:
                m = mode(predictions)
                print '============================================='
                print get_readable_time(crt_sec) + ', currently speaking: ' + classnames[m.mode[0]]
                prettyprint_time_dict(time_per_class)
                print '============================================='

            # print time_per_class
            wav.setpos(wav.tell() + samples_remaining_this_sec)
            predictions_this_sec = []
            samples_remaining_this_sec = sample_rate
            frames_examined_this_sec = 0
            crt_sec += 1
            print crt_sec
            continue
        """
    # output_stats(crt_sec, seconds_ignored, time_per_class)
    return


def get_config_data(model_path):
    config = ConfigParser()
    config.read(model_path + '-config.ini')
    tensor_size = int(config.get('Sizes', 'tensor_size'))
    num_classes = int(config.get('Sizes', 'num_classes'))
    classnames_str = config.get('Classnames', 'classnames')
    classnames = classnames_str.split(',')
    return tensor_size, num_classes, classnames


def create_flat_spectrogram(sound_info, frame_length, sample_rate):
    pylab.figure(num=None, figsize=(19, 12))
    pylab.axis('off')
    pylab.specgram(sound_info, NFFT=frame_length, Fs=sample_rate)
    pylab.savefig('crt.png')
    pylab.close()
    spectro = Image.open('crt.png')
    spectro = squarify(spectro, 256)
    flat_spectro = flatten(spectro)
    return flat_spectro

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


def get_crt_time_string(sample_rate, current_pos):
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



def get_second(sample_rate, current_pos):
    return current_pos / sample_rate

def update_sec_dict(sec_dict, second):
    if second not in sec_dict:
        sec_dict[second] = 1
        return
    sec_dict[second] += 1
    return


def output_stats(crt_sec, seconds_ignored, time_per_class):
    print '======================STATS======================='
    print 'Total time of recording ' + get_readable_time(crt_sec)
    print '--------------------------------------------------'
    print 'Total time ignored ' + get_readable_time(seconds_ignored)
    print '--------------------------------------------------'
    prettyprint_time_dict(time_per_class)
    print '=================================================='


def get_readable_time(total_seconds):
    seconds = '0' + str(total_seconds % 60) if total_seconds % 60 < 10 else str(total_seconds % 60)
    total_minutes = total_seconds / 60
    minutes = '0' + str(total_minutes % 60) if total_minutes % 60 < 10 else str(total_minutes % 60)
    total_hours = total_minutes / 60
    hours = '0' + str(total_hours % 24) if total_hours % 24 < 10 else str(total_hours % 24)
    return str(hours) + ':' + str(minutes) + ':' + str(seconds)


def prettyprint_time_dict(time_dict):
    for key, value in time_dict.iteritems():
        print '------------------------------------------------'
        print key + ' spoke for ' + get_readable_time(value)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Incorrect usage, please see top of file."
        exit()
    model_path = sys.argv[1]
    wav_path = sys.argv[2]
    frame_length = int(sys.argv[3])
    deque_size = int(sys.argv[4])
    # frame_length = sys.argv[3]
    read_and_predict(model_path, wav_path, frame_length, deque_size)
