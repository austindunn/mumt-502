import os
import sys
import string
from glob import glob

"""
This program will rename .png files in a folder simply to
incrementing numbers from 0 to the number of files in the
directory.

Usage: python reorganize.py [path]
where the path is the directory of the files to be renamed
according to standard set for names of testing/training
data.
"""

def reorganize(folder):
    files = glob(folder + '*.png')
    count = 0
    newName = folder + str(count) + '.png'
    for fyle in files:
        newName = folder + str(count) + '.png'
        os.rename(fyle, newName)
        count += 1
    print 'Finished! ' + str(count) + ' files renamed.'


if __name__ == "__main__":
    folder = sys.argv[1]
    reorganize(folder)
