import sys
import os

from PIL import Image
import numpy as np


def flatten_image(path):
    im = Image.open(path)
    data = np.array(im)
    flat = data.flatten()
    return flat


if (__name__ == "__main__"):
    path = sys.argv[1]
    flat = flatten_image(path)
