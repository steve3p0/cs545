from PIL import Image
import PIL.ImageOps
import numpy as np
import sys
import os
import csv

def createFileList(myDir, format='.jpg'):
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith('.jpg'):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList

mydir='images'

#load the original image
fileList = createFileList(mydir)

for file in fileList:
    print(file)
    img_file = Image.open(file) # imgfile.show()
    img_file = PIL.ImageOps.invert(img_file)

    # get original image parameters...
    width, height = img_file.size
    format = img_file.format
    mode = img_file.mode

    # Make image Greyscale
    img_grey = img_file.convert('L')
    #img_grey.save('result.png')
    #img_grey.show()

    # Save Greyscale values
    value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))
    value = value.flatten()
    print(value)
    with open("img_pixels.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(value)
