import pandas as pd
import argparse
import os
import sys
from shutil import copy
from torchvision import transforms as t
from torchvision.io import write_jpeg
from torchvision.io import read_image
from torchvision.utils import make_grid
from PIL import Image
from sklearn import model_selection
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(
                    prog='Augmentation',
                    description='Programme that shows augmentation'
                    ' done if passed one image'
                    'or creates balanced dataset from directory')
    parser.add_argument('directory', metavar='directory', type=str, nargs=1,
                        help='directory to augment'
                        'information about images from')
    args = parser.parse_args()
    return args


def flip(image, path=None):
    t_image = t.RandomHorizontalFlip(1)(image)
    if path is not None:
        write_jpeg(t_image, path + "_Flip.jpg")
    return t_image


def rotate(image, path=None):
    t_image = t.functional.rotate(image, 30,
                                  interpolation=Image.Resampling.BILINEAR)
    if path is not None:
        write_jpeg(t_image, path + "_Rotation.jpg")
    return t_image


def blur(image, path=None):
    t_image = t.GaussianBlur(9)(image)
    if path is not None:
        write_jpeg(t_image, path + "_Blur.jpg")
    return t_image


def contrast(image, path=None):
    t_image = t.functional.adjust_contrast(image, 1.5)
    if path is not None:
        write_jpeg(t_image, path + "_Contrast.jpg")
    return t_image


def brightness(image, path=None):
    t_image = t.ColorJitter((1.8, 2))(image)
    if path is not None:
        write_jpeg(t_image, path + "_Brightness.jpg")
    return t_image


def perspective(image, path=None):
    t_image = t.RandomPerspective(0.5, p=1)(image)
    if path is not None:
        write_jpeg(t_image, path + "_Distortion.jpg")
    return t_image


def scaling(image, path=None):
    t_image = t.RandomAffine(degrees=0, scale=(1.3, 1.7))(image)
    if path is not None:
        write_jpeg(t_image, path + "_Crop.jpg")
    return t_image


transformation_functions = [
    flip, rotate, blur, contrast,
    brightness, perspective, scaling
    ]


def augment_images(
        max_augmentations, directory, plant,
        class_name, list_of_images, dst=None):
    current_class = os.path.join(directory, plant, class_name)
    for image in list(list_of_images):
        if not os.path.exists(current_class):
            os.makedirs(current_class)
        copy(image, current_class)
        for augmentation in transformation_functions:
            if max_augmentations == 0:
                break
            path = os.path.join(
                                directory, plant, class_name,
                                image.split('/')[-1].split('.')[0]
                                )
            augmentation(read_image(image), path)
            max_augmentations -= 1
    if max_augmentations != 0:
        print("Balancing is not achieved")


def create_dataframe(list_of_files):
    dataframe = pd.DataFrame(list_of_files, columns=['filename'])
    dataframe['plant'] = dataframe['filename'].apply(
        lambda x: x.split('/')[-3]
        )
    dataframe['disease'] = dataframe['filename'].apply(
        lambda x: x.split('/')[-2]
        )
    return dataframe


def stratified_sample_df(df, col, n_samples):
    n = min(n_samples, df[col].value_counts().min())
    df_ = df.groupby(col).apply(lambda x: x.sample(n))
    df_.index = df_.index.droplevel(0)
    return df_


def val_train_split(directory):
    dst = 'augmentations'
    list_of_images = walk_through_dir(directory)
    dataframe = create_dataframe(list_of_images)
    num_of_aug = len(transformation_functions)
    for (plant, frame) in dataframe.groupby('plant'):
        aug_counter = {}
        # X_train, X_test, y_train, y_test = model_selection.train_test_split(
        #     frame['filename'], frame['disease'], test_size=0.1,
        #     random_state=42, stratify=frame['disease'])
        df_test = stratified_sample_df(frame, 'disease', 25)
        df_train = frame.drop(df_test.index)
        X_train = df_train['filename']
        y_train = df_train['disease']
        X_test = df_test['filename']
        y_test = df_test['disease']
        # df = pd.concat([X_train, y_train], axis=1)
        max_number_augment = df_train.value_counts('disease').min() \
            * num_of_aug
        for idx in df_train.value_counts('disease').index:
            aug_for_class = max_number_augment \
                - df_train.value_counts('disease').loc[idx]
            aug_counter[idx] = aug_for_class
        for x_row, y_row in zip(X_train, y_train):
            path = os.path.join(dst, plant, 'train', y_row)
            if not os.path.exists(path):
                os.makedirs(path)
            for augmentation in transformation_functions:
                if aug_counter[y_row] > 0:
                    save_path = os.path.join(
                                        path,
                                        x_row.split('/')[-1].split('.')[0])
                    augmentation(read_image(x_row), save_path)
                    aug_counter[y_row] -= 1
            copy(x_row, path)
        for x_row, y_row in zip(X_test, y_test):
            path = os.path.join(dst, plant, 'val', y_row)
            if not os.path.exists(path):
                os.makedirs(path)
            copy(x_row, path)


def create_balanced_dataset(directory):
    aug_dir = "augmented_directory"
    if not os.path.exists(aug_dir):
        os.mkdir(aug_dir)
    list_of_images = walk_through_dir(directory)
    dataframe = create_dataframe(list_of_images)
    num_of_aug = len(transformation_functions)

    for (plant, frame) in dataframe.groupby('plant'):
        print("Class: ", plant)
        max_number_augment = frame.value_counts('disease').min() * num_of_aug
        print(
            "The least represented class: ",
            frame.value_counts('disease').idxmin())
        print(
            "Max number of transformed images",
            "for less represented class: ",
            max_number_augment)
        print(
            "The most represented class: ",
            frame.value_counts('disease').idxmax())
        print(
            "Number of images for the most represented class: ",
            frame.value_counts('disease').max())
        print("-" * 10)
        for idx in frame.value_counts('disease').index:
            aug_for_class = max_number_augment
            - frame.value_counts('disease').loc[idx]
            print(idx, aug_for_class)
            augment_images(
                        aug_for_class, aug_dir, plant, idx,
                        dataframe[
                                    (dataframe['disease'] == idx) &
                                    (dataframe['plant'] == plant)]['filename'])


def walk_through_dir(directory):
    list_of_images = []
    for plant in os.listdir(directory):
        for disease in os.listdir(os.path.join(directory, plant)):
            for file in os.listdir(os.path.join(directory, plant, disease)):
                if os.path.isfile(os.path.join(
                                                directory, plant,
                                                disease, file)):
                    list_of_images.append(os.path.join(
                                                        directory, plant,
                                                        disease, file))
    return list_of_images


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def one_image(img_path):
    img = read_image(img_path)
    list_of_augs = []
    for augmentation in transformation_functions:
        list_of_augs.append(augmentation(img))
    grid = make_grid(list_of_augs)
    show(grid)


if __name__ == '__main__':
    args = parse_arguments()
    path = args.directory[0]
    if not os.path.exists(path):
        sys.exit('This directory doesn\'t exist or is not accessible')
    if os.path.isfile(path):
        one_image(path)
    if os.path.isdir(path):
        val_train_split(path)
