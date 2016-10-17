import sys
import os
import wave
import pylab
from PIL import Image, ImageChops


def create_spectrograms(wav_file, destination):
    wav = wave.open(wav_file, 'r')
    num_frames = wav.getnframes()
    sample_rate = wav.getframerate()
    crt = 0
    num_spectros = num_frames/256
    print "Starting... Creating " + str(num_spectros) _ " spectrograms."
    while (wav.tell() + 256) < num_frames:
        crt = wav.tell()/256
        if (crt % 100 == 1 and crt > 1):
            print "Created " + str(crt-1) + " spectrograms so far. " + str(num_spectros - crt) + " to go. " + str(crt / num_spectros) + "% complete."
        filename = str(crt/7) if ((crt % 7 == 0) and crt > 1)) else str(crt - (crt/7) - 1)
        frames = wav.readframes(256)
        sound_info = pylab.fromstring(frames, 'Int16')


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
