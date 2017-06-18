"""
Script for converting from csv file datafiles to a directory for each image. (which is how it is loaded by MAML code)

To process, put images.zip in the miniImagenet directory, unzip it (such that it is in the directory 'images/'), then run the following:
    cd data/miniImagenet/
    python proc_images.py
"""
import csv
import glob
import os

from PIL import Image

path_to_images = 'images/'

all_images = glob.glob(image_path + '*')

# Resize images
i = 0
for image_file in all_images:
    im = Image.open(image_file)
    im = im.resize((84,84), resample=Image.LANCZOS)
    im.save(image_file)
    i += 1
    if i % 500 == 0:
        print i

# Put in correct directory
for datatype in ['train', 'val', 'test']:
    os.system('mkdir ' + datatype)

    with open(datatype+'.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        i = 0
        last_label = ''
        for row in reader:
            i+=1
            if i == 1: continue
            label = row[1]
            image_name = row[0]
            if label != last_label:
                cur_dir = datatype + '/' + label + '/'
                os.system('mkdir ' + cur_dir)
                last_label = label
            os.system('mv images/'+image_name+' ' + cur_dir)

