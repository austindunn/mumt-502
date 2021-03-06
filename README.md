# mumt-502

This repository tracks the progress I've made on my senior project in music technology at McGill University, supervised by Professor Ichiro Fujinaga. 

### Papers read (all or in part)
* Keunwoo Choi, Gyorgy Fazekas, Mark Sandler. Automatic Tagging Using Deep Convolutional Neural Networks. Queen Mary University of London, 2016.
* Jan Schlüter, Thomas Grill. Exploring Data Augmentation For Improved Singing Voice Detection With Neural Networks. Austrian Research Institute for Artificial Intelligence, Vienna, 2015.
 * Examined for downsampling/hop size settings for clips in dataset.
* Philippe Hamel, Simon Lemieux, Yoshua Bengio, Douglas Eck. Temporal Pooling And Multiscale Learning For Automatic Annotation And Ranking Of Music Audio. DIRO, Université de Montréal. CIRMMT.
 * Examined for downsampling/hop size settings for clips in dataset.

### Structure of the training/testing data
The directory passed to my train function contains two subdirectories: *training* and *testing*. In each of these subdirectories, there will be a directory named for each of the classes. Each file in these directories, for purposes of clarity, will be named according to the following standard: `[n].png`, where n is a number from 0 to the total number of training/testing samples for that class. The meta data for each file is simply contained in its filepath.

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

*October 8-10, 2016*
* Code/file restructuring...

*October 11, 2016*
* Ran some tests:
 * __1__
  * 100 samples per class * 100 training loops = 10,000 randomly selected samples from each class.
  * tested with 20 samples, yeilded a 70% success rate.
 * __2__
  * 100 samples per class * 100 training loops = 10,000 randomly selected samples from each class.
  * test with 2000 samples caused a crash with message `Killed: 9`
 * __3__
  * 50 samples per class * 100 training loops = 2500 randomly selected samples from each class.
  * tested with 200 samples, yeilded a 66.75 % success rate.
 * __4__ (same parameters as 3)
  * 50 samples per class * 100 training loops = 2500 randomly selected samples from each class.
  * tested with 200 samples, yeilded a 68.25 % success rate.

*October 12, 2016*
* Did some research on ways to improve the scores I'm getting.
* Ran a test overnight:
 * 1000 train loops with 50 samples per class per loop.
 * Tested with 200 samples, with a 66.25% accuracy rate.
 * This was the test with the most training loops, yet it yeilded the lowest accuracy of all the tests run so far.

*October 13, 2016*
* Ran a test during the day:
 * 1000 training loops with 100 samples per class per loop, tested with 200 samples yeilded 0.5 accuracy. Very strange. So much for those 13 hours of training.
* Since the training with fewer samples has so far yeilded better results, I'm going to run another test with 100 samples * 100 training loops, tested with 200 samples.
 * This overnight tested yeilded accuracy of .68.

*October 14, 2016*
* Ran another test this morning, with 100 samples per class per loop, 100 training loops, and 200 test samples.
 * Accuracy: .6925.
 * Seems that lots of samples and a low number of training loops leads to best results. So far, the test above (100 samples per class * 100 training loops) has yeilded the best results. Tonight I'll run another test with more training loops just to see if yesterday's test was a fluke.

*October 15, 2016*
* Day test: 50 samples per class x 50 training loops, tested with 20 test loops x 100 samples per class per test loop.
 * Yeilded a low score of 52.05%
* Night test: 100 samples per class x 50 training loops, tested with 20 test loops x 200 samples per class per test loop.
 * Yeilded a better score of 59.15%

*October 16, 2016*
* Found a bunch of ways to fix my data to make it smaller (and therefore faster and less demanding in terms of memory), mostly sticking to tricks I can do with PIL (Python's imaging library). These fixes will also help my data to conform with what TensorFlow does in the Deep MNIST Tutorial, which I'm following for building my deeper model.
* Running a test tonight with 50 training loops, 200 samples per class per loop, tested with 20 test loops, each with 200 samples per class... 
 * 64.96% success rate.

*October 17, 2016*
* Day test: 20 training loops, 300 samples per class per loop, tested with 20 test loops of 200 samples per class per loop.
 * 64.66%
 * In this test, I was looking to find some difference in running fewer loops, but using an especially large number of samples per loop.
* Tonight, I'll run a much longer test, hopefully to get it to run through the day tomorrow so I can see the results after work. For this test, I'll do 150 training loops with 250 samlpes per class each, and test with 40 test loops of 200 samples per class each.

*October 21, 2016*
* Ran a dummy test with mislabelled data, got a success rate of 41.12% (and results of each test loop all over the place, from 20% to 70%). So at least we know that my model is training correctly, if results tend to be both higher and more consistent across test loops.

*October 22, 2016*
* Finalized a script that filters samples by lower amplitude, spent the evening/night generating samples. This script also creates spectrograms with much lower resolution, I'm expecting a speed increase as result.

*October 23, 2016*
* Ran an initial test on new data - using just the old script, with 100 training loops of 100 samples per class each, already the results show roughly a 15% increase in accuracy, with an average accuracy (across 20 training loops of 200 samples per class each) of 80.18%.
* Overnight test (still using simple single-layer model) with 2000 training loops of 200 samples per class each yeilded an average accuracy of 93.14%! This actually beats the accuracy of TensorFlow's MNIST Beginner tutorial. Of course, my problem is only 2 classes instead of 10, but that's still pretty cool. Hoping to test out the deeper model I've been working on sometime in the next day or so.

*October 25, 2016*
* I finished the 'deeper' version of my model, which now features 2 hidden layers. On an initial test, I used 100 training loops of 100 samples per class each, and tested with 20 test loops of 200 samples per class each, and the average accuracy was 91.75%. This is a slightly lower score than my best training session with the simpler model, with the notable difference that the simpler model took 8 hours to reach its result, whereas the deeper model took about 10 minutes. Very excited for the overnight test tonight.
* Overnight test with 1000 training loops with 500 samples per class each tested with an average success rate of 94.25%.

*November 13, 2016*
* After playing with simple models (regular neural net, no hidden layers) and deeper models of varying depths, I found that a simple model with lots of data was highly efficient and suited the two-class problem well. Going forward, I'll be using a regular neural net to create and save models, which will then be used to classify data in a file frame-to-frame (my standard frame size remains 256 samples). I've now created such a model, and will be putting efforts into this next step of classifying audio frames sequentially.

*November 18, 2016*
* After fixing a problem that was happening due to examining a stereo instead of mono .wav file, I've managed to train and use a model (currently in new-model/new-model) that works quite nicely when used to examine a real file with the classifier.py file in sequential-reading/.
* Some notes on this development:
 * The model is *very* good at predicting whether it is Donald Trump speaking, not quite as reliable when examinging samples of Hillary Clinton's voice, though still pretty good
  * May want to implement a testing function that tests a model for each class.
 * Need to figure out an error-prone way to predict who's actually talking: one guess per 256-sample fram is not quite enough. I'll be looking into ways to get my model to look at data as a sequence instead of as independent and identically distributed.
 * More classes: recognizing a moderator, detecting interruptions, detecting crowd noise. The model may need to be trained with samples from each of those classes.
 * Speed is an issue when examining real data, as the spectrograms and image manipulations take some time.

*November 29, 2016*
* fixed local minima error, fixed data so that lester is guessed accurately (problem was a stereo file being used to generate spectrograms, instead of mono).
* updated restore\_test.py to generate a confusion matrix. Results on the current three-class model (stored in three-class/three-class.ckpt) show about 90% accuracy for Hillary and Trump each, around 85% for Lester).

*December 11, 2016*
* fixed organization of spectrograms created by create\_spectrograms.py
 * necessary non-existent directories will be created

*December 14, 2016*
* One version of the classification file is now complete, it determines who's talking frame by frame. 
 * Running a test on this classification file, using the whole first debate and some statistics from ABC on who spoke the most. According to ABC, I should expect to see that Hillary spoke for a total of 41 minutes and 50 seconds, and that Trump spoke for a total of 45 minutes and 3 seconds.

*December 15, 2016*
* To my great frustration, somewhere just at the end of the debate recording (the program logged the statistics at the 1:32:00.000 mark, total length of the file is 1:32:43.130), my computer crashed and I did not receive the final statistics. I'll be running a shorter test tonight to see why this happened - the program hasn't crashed when analyzing much shorter (i.e. < 1 minute) recordings

*December 23, 2016*
* After doing some final code edits and refactoring, I'll be running a full test over the next day or so.
* The portion of code destined for others to pick up and use has been moved to [this repository](https://github.com/austindunn/tf-voice-classifier).

*December 24, 2016*
* Final tests finished
* All important code now over at new repo, mentioned in last entry, as well as wiki. This repository will constitute the final project submission.
