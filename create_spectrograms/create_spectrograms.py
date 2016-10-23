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
    wavs = glob(directory + '*.wav')
    count = 0
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
            # filter out windows with low amplitudes (i.e. no vocal information)
            if (amps.mean() < 1000):
                continue
            # split training:testing 7:1
            if (count > 1):
                filename = str(count/7) if (count % 7 == 0) else str(count - (count/7) - 1)
            else:
                filename = '0'
            directory = destination + 'testing/' if (count % 7 == 0) else destination + 'training/'
            # create, edit, and place the spectrogram image
            pylab.figure(num=None, figsize=(19, 12))
            pylab.axis('off')
            pylab.specgram(sound_info, NFFT=frame_length, Fs=sample_rate)
            pylab.savefig(filename + '.png')
            pylab.close()
            im = Image.open(filename + '.png')
            im = customize(im, image_size, greyscale)
            im.save(filename + '.png')
            os.rename(filename + '.png', directory + filename + '.png') 
            # logging
            if (count % 100 == 1 and count > 1):
                print 'Created ' + str(count-1) + ' spectrograms so far. ' + str(wav.tell()/frame_length) + ' windows examined of file ' + wav_file + '.'
            count += 1
        wav.close()
    print 'All finished! ' + str(count) + ' spectrograms created.'


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
