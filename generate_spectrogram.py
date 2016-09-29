#!/usr/bin/env python
# Corey Goldberg 2013

"""Generate a Spectrogram image for a given WAV audio sample.

A spectrogram, or sonogram, is a visual representation of the spectrum
of frequencies in a sound.  Horizontal axis represents time, Vertical axis
represents frequency, and color represents amplitude.
"""

import sys
import os
import wave

from PIL import Image, ImageChops

import pylab


def graph_spectrogram(wav_file):
    sound_info, frame_rate = get_wav_info(wav_file)
    filename = 'spectrogram - %s' % os.path.splitext(wav_file)[0]
    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    pylab.specgram(sound_info, NFFT=256, Fs=frame_rate)
    pylab.axis('off')
    pylab.savefig('spectrogram - %s' % os.path.splitext(wav_file)[0])
    im = Image.open("spectrogram - Alesis-Fusion-Shakuhachi-C5.png")
    im = trim(im)
    im.save('spec.png')
    

def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'Int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


if __name__ == '__main__':
    wav_file = 'Alesis-Fusion-Shakuhachi-C5.wav'
    graph_spectrogram(wav_file)
