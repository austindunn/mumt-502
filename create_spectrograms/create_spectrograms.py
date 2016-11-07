"""
This script creates a bunch of square spectrograms from short audio clips.

Usage: python create_spectrograms.py [directory] [destination] [frame length]
                                     [image size] [greyscale]
                                     where...
[directory] is the name of the directory containing wav files to be turned into spectrograms
[destination] is the directory to save the spectrograms to
[frame length] is how long each frame to be FFT'd should be
[image size] is the height and width of resulting spectrogram images
[greyscale] is either 0 or 1, indicating whether the images should be in greyscale or color
"""

import sys
import os
import wave
import pylab
import numpy
from glob import glob
from PIL import Image, ImageChops


def create_spectrograms(directory, destination, frame_length, image_size, greyscale):
    classnames = os.walk(directory).next()[1]
    # silence is a special class, as its spectrograms come from wav files of other classes.
    training_dest = destination + 'training/'
    testing_dest = destination + 'testing/'
    silence_count = 0
    for classname in classnames:
        print 'Now starting on class ' + classname + '.'
        class_wav_dir = directory + classname + '/'
        wavs = glob(class_wav_dir + '*.wav')
        class_count = 0
        total_count = 0
        for wav_file in wavs:
            wav = wave.open(wav_file, 'r')
            num_frames = wav.getnframes()
            sample_rate = wav.getframerate()
            num_windows = num_frames/frame_length
            print 'Creating spectrograms for file ' + wav_file + '... Examining ' + str(num_windows) + ' windows.'
            while (wav.tell() + frame_length) < num_frames:
                frames = wav.readframes(frame_length)
                sound_info = pylab.fromstring(frames, 'Int16')
                amps = numpy.absolute(sound_info)
                # move silence to special class directory
                # ambiguous (unsure if sound or silence) clips will be ignored
                # split training:testing 7:1
                if (amps.mean() > 100 and amps.mean() < 400):
                    continue
                elif (amps.mean() <= 100):
                    filename = str(silence_count/7) if (silence_count % 7 == 0) else str(silence_count - (silence_count/7) - 1)
                    full_destination = testing_dest + 'silence/' if (silence_count % 7 == 0) else training_dest + 'silence/'
                    silence_count += 1
                elif (amps.mean() >= 400):
                    filename = str(class_count/7) if (class_count % 7 == 0) else str(class_count - (class_count/7) - 1)
                    full_destination = testing_dest + classname + '/' if (class_count % 7 == 0) else training_dest + classname + '/'
                    class_count += 1
                # create, edit, and place the spectrogram image
                pylab.figure(num=None, figsize=(19, 12))
                pylab.axis('off')
                pylab.specgram(sound_info, NFFT=frame_length, Fs=sample_rate)
                pylab.savefig(filename + '.png')
                pylab.close()
                im = Image.open(filename + '.png')
                im = customize(im, image_size, greyscale)
                im.save(filename + '.png')
                os.rename(filename + '.png', full_destination + filename + '.png') 
                # logging
                if ((total_count+1) % 100 == 0 and total_count > 0):
                    print 'Created ' + str(total_count+1) + ' spectrograms so far. ' + str(wav.tell()/frame_length) + ' windows examined of file ' + wav_file + '.'
                total_count += 1
            wav.close()
        print 'All finished with class ' + classname + '! ' + str(total_count) + ' spectrograms created.'


def customize(im, image_size, greyscale):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        im = im.crop(bbox)
    im = im.resize((image_size, image_size))
    if greyscale:
        im = im.convert('L')
    return im


if __name__ == "__main__":
    directory = sys.argv[1]
    destination = sys.argv[2]
    frame_length = sys.argv[3]
    image_size = sys.argv[4]
    greyscale = sys.argv[5]
    create_spectrograms(directory, destination, int(frame_length), int(image_size), int(greyscale))
