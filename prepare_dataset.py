from PIL import Image, ImageOps
from imutils import paths
import cv2
import os
import random
import numpy as np

def compute_hash(im, hash_size=16):

    # convert the image to grayscale and resize the grayscale image,
    # adding a single column (width) so we can compute the horizontal
    # gradient
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size))

    # compute the (relative) horizontal gradient between adjacent
    # column pixels
    diff = resized[:, 1:] > resized[:, :-1]

    # convert the difference image to a hash and return it
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def remove_duplicates(folder_path):

    image_paths = list(paths.list_images(folder_path))
    hashes = {}

    for imagePath in image_paths:

        # load the input image and compute the hash
        image = cv2.imread(imagePath)
        image_hash = compute_hash(image)

        # add hash to dictionary of hashes
        p = hashes.get(image_hash, [])
        p.append(imagePath)
        hashes[image_hash] = p

    for (h, hashedPaths) in hashes.items():

        # check to see if there is more than one image with the same hash
        if len(hashedPaths) > 1:

            # loop over all image paths with the same hash *except*
            # for the first image in the list (since we want to keep
            # one, and only one, of the duplicate images)
            for p in hashedPaths[1:]:
                os.remove(p)


def rename_by_folder(folder_path, class_name):

    # files in classes
    num = len(os.listdir(folder_path))

    # generate random names for this class
    names = random.sample(range(num), num)

    for i, f_name in enumerate(os.listdir(folder_path)):
        os.rename(folder_path + f_name, folder_path + class_name + '_' + str(names[i]+1) + '.jpg')


def make_squared(im):

    # get dimensions and black-border size
    large, short = (im.shape[0], im.shape[1]) if im.shape[0] > im.shape[1] else (im.shape[1], im.shape[0])
    border = int((large - short) / 2)

    # initialize squared image
    squared = np.zeros((large, large), dtype=np.int8)

    # fill squared image on right dimension
    if im.shape[0] == large:
        # add columns
        squared[:, border: border + short] = im
    else:
        # add rows
        squared[border: border + short, :] = im

    return squared


# select the filepath and class names:
path = './brain_tumor_clean/preprocessed/'
final_path = './brain_tumor_clean/preprocessed/'

final_classes = ['health', 'tumor']
classes = ['health', 'tumor']

# execute the script
for cl in classes:

    print("[INFO] removing duplicates in: " + cl)
    remove_duplicates(path + cl + '/')

    print("[INFO] preprocessing images in: " + cl)
    for filename in os.listdir(path + cl):

        # load and convert to grayscale numpy array
        im_gray = np.asarray(ImageOps.grayscale(Image.open(path + cl + '/' + filename)),
                             dtype=np.int8)

        # make it squared extending borders
        if im_gray.shape[0] != im_gray.shape[1]:
            im_gray = make_squared(im_gray)

        # reshape to 256 x 256 and save
        im_gray = Image.fromarray(im_gray, 'L').resize((256, 256))
        im_gray.save(final_path + final_classes[classes.index(cl)] + '/' + filename)

# remove duplicates a second time
for cl in final_classes:

    print("[INFO] removing duplicates in final path: " + cl)
    remove_duplicates(final_path + cl + '/')

    print("[INFO] renaming and shuffling images in: " + cl)
    rename_by_folder(final_path + cl + '/', cl)
