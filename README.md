# mumt-502

This repository tracks the progress I've made on my senior project in music technology at McGill University, supervised by Professor Ichiro Fujinaga. 

### Papers read (all or in part)
* Keunwoo Choi, Gyorgy Fazekas, Mark Sandler. Automatic Tagging Using Deep Convolutional Neural Networks. Queen Mary University of London, 2016.
* Jan Schlüter, Thomas Grill. Exploring Data Augmentation For Improved Singing Voice Detection With Neural Networks. Austrian Research Institute for Artificial Intelligence, Vienna, 2015.
 * Examined for downsampling/hop size settings for clips in dataset.
* Philippe Hamel, Simon Lemieux, Yoshua Bengio, Douglas Eck. Temporal Pooling And Multiscale Learning For Automatic Annotation And Ranking Of Music Audio. DIRO, Université de Montréal. CIRMMT.
 * Examined for downsampling/hop size settings for clips in dataset.

### Structure of the data
The directory passed to my train function contains two subdirectories: *training* and *testing*. In each of these subdirectories, there will be a directory named for each of the classes. Each file in these directories, for purposes of clarity, will be named according to the following standard: "[classname]\_[n]", where n is a number from 0 to the total number of training/testing samples for that class.

### A diary of progress made
*September 25, 2016*

* Progress made before today:
 * Read paper: Automatic Tagging Using Deep Convolutional Neural Networks, and learned a bunch of terminology.
 * Collected a set of other papers on the subject of deep learning using convnets in audio.
 * Installed TensorFlow, ran preliminary test program.
 * Put together a program that, given a wav file, will create the spectrogram for the audio sample.
* Today
 * Created this github repository.
 * Ran the TensorFlow tutorial on MNIST data using the Softmax Regression model.
 * Did some research on how to break up longer wav signals into smaller "chunks" in the interest of gaining speed and more data points.
* Questions
 * Testing vs. validation datasets
 * Spectrograms yielded by my script - what's up with the horizontal symmetry?
 * In Singing Voice Detection paper, what does *frame length* refer to? Similarly, in the Automatic Annotation and Ranking paper, there is a value for *frame step*. 
  * These values are not used in calculation for number of frames per second.

*September 27, 2016*

* Downloaded last night's debate between Hillary Clinton and Donald Trump, which I'll use as isolated audio for each person in running an initial session in TensorFlow.
 * Cut out a few clips using Audacity.
* Wrote a script that takes as input (command line for now) a wav file, then creates spectrograms for each 256-sample 'slice' of that wav file and saves it.
* Questions
 * How high should the resolution of the spectrogram be? This could be a good way to save some time and space. I will do research on this on my own, but any info is good.

*September 30, 2016*

* Reorganized some code, fixed chop\_frames.py to take destination of files as argument--easier to use and deposit spectrograms where they should be.

*October 2, 2016*
* Wrote code to flatten images into 1-D arrays using PIL and numpy
* Documented tensorflow solution, to be coded tomorrow and the next day
* Split data into testing and training directories

*October 3, 2016*
* More work on training script...

*October 4, 2016*
* Model is training! Very slowly... 300 loops at ~19s per loop will take around 1.5 hours, which needs to be shorter in the future. For now, we just need an accuracy check, which I'll code up tomorrow morning.

*October 5, 2016*
* Functioning accuracy check written. Only dummy tested so far - one sample for each class, one sample for training. Led to result of 0.5.

*October 7, 2016*
* Started on standardizing file structure and reflecting that in the code.


