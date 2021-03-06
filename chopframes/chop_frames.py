"""
Created September 27, 2016
By Austin Dunn

This script is used to take short clips of audio and 'chop'
them into spectrograms. To start, this script will take one
wav file and chop it into frames of 256 samples each.
"""

import sys
import os
import wave

from PIL import Image, ImageChops

import pylab


def create_spectrograms(wav_file, destination):
    wav = wave.open(wav_file, 'r')
    numframes = wav.getnframes()
    sample_rate = wav.getframerate()
    crt = 0
    numslices = numframes/256
    print "Starting now... Creating " + str(numslices) + " slices."
    while (wav.tell() + 256) < numframes:
        crt = wav.tell()/256
        if (crt % 100 == 1 and crt > 1):
            print "Created " + str(crt-1) + " spectrograms so far. " + str(numslices - crt) + " to go. " + str(crt / numslices) + "% complete."
        filename = os.path.splitext(wav_file)[0] + "_slice" + str(crt)
        frames = wav.readframes(256)
        sound_info = pylab.fromstring(frames, 'Int16')
        pylab.figure(num=None, figsize=(19, 12))
        pylab.subplot(111)
        pylab.specgram(sound_info, NFFT=256, Fs=sample_rate)
        pylab.axis('off')
        pylab.savefig(filename)
        pylab.close()
        im = Image.open(filename + ".png")
        im = trim(im)
        im.save(filename + ".png")
        os.rename(filename + ".png", destination + filename + ".png")
    wav.close()
    print "All finished! " + str(crt) + " slices created."


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


if __name__ == "__main__":
    wav_file = sys.argv[1]
    destination = sys.argv[2]
    create_spectrograms(wav_file, destination)
