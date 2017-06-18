"""
Usage instructions:
    First download the omniglot dataset and put it in data/omniglot/

    Then, run the following:
    cd data/omniglot_resized/
    cp -r omniglot/* omniglot_resized/
    python resize_images.py
"""
from PIL import Image
import glob

image_path = '*/*/'

all_images = glob.glob(image_path + '*')

i = 0

for image_file in all_images:
    im = Image.open(image_file)
    im = im.resize((28,28), resample=Image.LANCZOS)
    im.save(image_file)
    i += 1

    if i % 200 == 0:
        print i

