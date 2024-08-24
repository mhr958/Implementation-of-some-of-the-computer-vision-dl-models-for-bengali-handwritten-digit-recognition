import glob
import os
import cv2
import numpy as np
import pandas as pd
from itertools import chain


def get_numtadb_training_data(dataset_directory='NumtaDB_Bengali Handwritten Digits', img_resize_size=(32,32), img_fmt='grayscale'):
    """
    Arguments:
    dataset_directory -- path to dataset directory relative to current working directory.
    img_resize_size -- a tuple containing the height and width of the rezied image, defaults to 32x32.
    img_fmt -- in which format the image should be loaded. either 'grayscale' (default) or 'color'

    Returns:
    images, labels -- the images and labels as numpy arrays.

    """

    dataset_directory_files = glob.glob(f"{dataset_directory}/*")


    dataset_training_image_paths = [
        glob.glob(f"{_}/**/*.png", recursive=True)
        for _ in dataset_directory_files
        if os.path.isdir(_) and _.split("\\")[-1].split("-")[0] == "training"
    ]
    dataset_training_image_paths = list(
        chain.from_iterable(dataset_training_image_paths)
    )  # as the original list would be [[training-a files], [training-b files]........]
    dataset_training_labels_paths = [
        _ for _ in dataset_directory_files if not os.path.isdir(_)
    ]

    labels_df = pd.concat(map(pd.read_csv, dataset_training_labels_paths), ignore_index=True)

    # setting the filename as index so that it can be used as key to retrieve related label for each image
    labels_df.set_index('filename', inplace=True)

    images = []
    labels = []
    if img_fmt=='grayscale':
        for img_path in dataset_training_image_paths:
            key = img_path.split(os.sep)[-1]
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, img_resize_size)
            img = img/255.0
            images.append(img)
            label = labels_df.loc[key]['digit']
            labels.append(label)

    if img_fmt=='color':
        for img_path in dataset_training_image_paths:
            key = img_path.split(os.sep)[-1]
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_resize_size)
            img = img/255.0
            images.append(img)
            label = labels_df.loc[key]['digit']
            labels.append(label)

    # converting to numpy array
    images = np.array(images)
    labels = np.array(labels)
    print('Dataset Loading is Complete')
    return images, labels