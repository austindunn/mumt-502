import tensorflow as tf
import numpy
import os
from PIL import Image

def train():
    x = tf.placeholder(tf.float32, [None, 4437272])
    W = tf.Variable(tf.zeros([4437272, 2]))
    b = tf.Variable(tf.zeros([2]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 2])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    # training loop here
    # for i in range(300):
    
    for i in range(1): # testing: make sure accuracy validation runs.
        print "step: " + str(i) # just to track where we are (speed)
        batch_samples = []
        batch_labels = []

        hillary_indices = numpy.arange(13799)
        numpy.random.shuffle(hillary_indices)
        hillary_indices
        for h in range(1):
            hillary_file = getHillaryTrainingSample(h, hillary_indices)
            hillary_flat = flatten_image(hillary_file)
            hillary_flat = hillary_flat.astype(numpy.float32)
            hillary_flat = numpy.multiply(hillary_flat, 1.0 / 255.0)
            batch_samples.append(hillary_flat)
            batch_labels.append([1,0])

        trump_indices = numpy.arange(12747)
        numpy.random.shuffle(trump_indices)
        for t in range(1):
            trump_file = getTrumpTrainingSample(t, trump_indices)
            trump_flat = flatten_image(trump_file)
            trump_flat = trump_flat.astype(numpy.float32)
            trump_flat = numpy.multiply(trump_flat, 1.0 / 255.0)
            batch_samples.append(trump_flat)
            batch_labels.append([0,1])

        sess.run(train_step, feed_dict={x: batch_samples, y_: batch_labels})
        

    # test accuracy of the model once trained
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    print "correct prediction"
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print "accuracy"
    testing_data, testing_labels = getTestingData()
    print "testing data loaded"
    print(sess.run(accuracy, feed_dict={x: testing_data, y_: testing_labels}))


def getHillaryTrainingSample(h, hillary_indices):
    h_index = hillary_indices[h]
    h_filename = ''
    """
    Major improvement needed here: can adjust the way files are named
    and organized (training/testing in order to avoid the following
    obnoxious flow control blocks.
    """
    if (h_index < 5774):
        h_filename = 'hillary1_slice' + str(h_index) + '.png'
    elif (h_index < 10595): # 5774 + 4821 = 10595
        h_filename = 'hillary2_slice' + str(h_index - 5774) + '.png'
    elif (h_index < 12385): # 10595 + 1790
        h_filename = 'hillary3_slice' + str(h_index - 10595) + '.png'
    elif (h_index < 16100): # 12385 + 3715
        h_filename = 'hillary4_slice' + str(h_index - 12385) + '.png'

    h_filename = '/Users/austindunn/Code/mumt502/samples/clinton_v_trump/spectrograms/hillary/training/' + h_filename

    if (not os.path.isfile(h_filename)):
        # no need to worry about spillover since 14872 % 7 != 0
        # THIS IS NOT SUSTAINABLE
        return getHillaryTrainingSample(h+1, hillary_indices)
    else:
        return h_filename


def getTrumpTrainingSample(t, trump_indices):
    t_index = trump_indices[t]
    t_filename = ''
    """
    Major improvement needed here: can adjust the way files are named
    and organized (training/testing in order to avoid the following
    obnoxious flow control blocks.
    """
    if (t_index < 6053):
        t_filename = 'trump1_slice' + str(t_index) + '.png'
    elif (t_index < 13483): # 6053 + 7430
        t_filename = 'trump2_slice' + str(t_index - 6053) + '.png'
    elif (t_index < 14873): # 13483 + 1390
        t_filename = 'trump3_slice' + str(t_index - 13483) + '.png'

    t_filename = '/Users/austindunn/Code/mumt502/samples/clinton_v_trump/spectrograms/trump/training/' + t_filename


    if (not os.path.isfile(t_filename)):
        # no need to worry about spillover since 14872 % 7 != 0
        # THIS IS NOT SUSTAINABLE
        return getTrumpTrainingSample(t+1, trump_indices)
    else:
        return t_filename


def flatten_image(path):
    im = Image.open(path)
    data = numpy.array(im)
    flat = data.flatten()
    return flat


def getTestingData():
    """
    Need to improve by relabelling test files, so that only one loop
    necessary per candidate.

    Note: Using all the testing data (currently about 5000 images)
    resulted in termination with a 'Killed: 9' message. I imagine 
    this has to do with a bound on system resources. Maybe move some 
    of the testing data to training, combined with lowering 
    resolution of spectrograms.
    """
    testing_data = []
    testing_labels = []

    # hillary test data
    for i in range(1):
        print 'i: ' + str(i)
        hillary_test_file = '/Users/austindunn/Code/mumt502/samples/clinton_v_trump/spectrograms/hillary/testing/hillary1_slice' + str(i*7) + '.png'
        hillary_test_flat = flatten_image(hillary_test_file)
        hillary_test_flat = hillary_test_flat.astype(numpy.float32)
        hillary_test_flat = numpy.multiply(hillary_test_flat, 1.0 / 255.0)
        testing_data.append(hillary_test_flat)
        testing_labels.append([1,0])

    """
    Removed to reduce size of training_data and prevent termination
    of program by system.

    for j in range(689):
        print 'j: ' + str(j)
        hillary_test_file = '/Users/austindunn/Code/mumt502/samples/clinton_v_trump/spectrograms/hillary/testing/hillary2_slice' + str(j*7) + '.png'
        hillary_test_flat = flatten_image(hillary_test_file)
        hillary_test_flat = hillary_test_flat.astype(numpy.float32)
        hillary_test_flat = numpy.multiply(hillary_test_flat, 1.0 / 255.0)
        testing_data.append(hillary_test_flat)
        testing_labels.append([1,0])
    for k in range(256):
        print 'k: ' + str(k)
        hillary_test_file = '/Users/austindunn/Code/mumt502/samples/clinton_v_trump/spectrograms/hillary/testing/hillary3_slice' + str(k*7) + '.png'
        hillary_test_flat = flatten_image(hillary_test_file)
        hillary_test_flat = hillary_test_flat.astype(numpy.float32)
        hillary_test_flat = numpy.multiply(hillary_test_flat, 1.0 / 255.0)
        testing_data.append(hillary_test_flat)
        testing_labels.append([1,0])
    for l in range(531):
        print 'l: ' + str(l)
        hillary_test_file = '/Users/austindunn/Code/mumt502/samples/clinton_v_trump/spectrograms/hillary/testing/hillary4_slice' + str(l*7) + '.png'
        hillary_test_flat = flatten_image(hillary_test_file)
        hillary_test_flat = hillary_test_flat.astype(numpy.float32)
        hillary_test_flat = numpy.multiply(hillary_test_flat, 1.0 / 255.0)
        testing_data.append(hillary_test_flat)
        testing_labels.append([1,0])
    """

    # trump test data
    for p in range(1):
        print 'p: ' + str(p)
        trump_test_file = '/Users/austindunn/Code/mumt502/samples/clinton_v_trump/spectrograms/trump/testing/trump1_slice' + str(p*7) + '.png'
        trump_test_flat = flatten_image(trump_test_file)
        trump_test_flat = trump_test_flat.astype(numpy.float32)
        trump_test_flat = numpy.multiply(trump_test_flat, 1.0 / 255.0)
        testing_data.append(trump_test_flat)
        testing_labels.append([0,1])

    """
    Removed to reduce size of training_data and prevent termination
    of program by system.

    for q in range(1062):
        print 'q: ' + str(q)
        trump_test_file = '/Users/austindunn/Code/mumt502/samples/clinton_v_trump/spectrograms/trump/testing/trump2_slice' + str(q*7) + '.png'
        trump_test_flat = flatten_image(trump_test_file)
        trump_test_flat = trump_test_flat.astype(numpy.float32)
        trump_test_flat = numpy.multiply(trump_test_flat, 1.0 / 255.0)
        testing_data.append(trump_test_flat)
        testing_labels.append([0,1])
    for r in range(199):
        print 'r: ' + str(r)
        trump_test_file = '/Users/austindunn/Code/mumt502/samples/clinton_v_trump/spectrograms/trump/testing/trump3_slice' + str(r*7) + '.png'
        trump_test_flat = flatten_image(trump_test_file)
        trump_test_flat = trump_test_flat.astype(numpy.float32)
        trump_test_flat = numpy.multiply(trump_test_flat, 1.0 / 255.0)
        testing_data.append(trump_test_flat)
        testing_labels.append([0,1])
    """

    return testing_data, testing_labels
        

if __name__ == "__main__":
    train()
