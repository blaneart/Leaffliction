import pandas as pd
import argparse
import os
import sys 
from shutil import copy
from sklearn import model_selection

def create_dataframe(list_of_files):
    dataframe = pd.DataFrame(list_of_files, columns=['filename'])
    dataframe['plant'] = dataframe['filename'].apply(lambda x: x.split('/')[-3])
    dataframe['disease'] = dataframe['filename'].apply(lambda x: x.split('/')[-2])
    return dataframe


def walk_through_dir(directory):
    list_of_images = []
    for plant in os.listdir(directory):
        for disease in os.listdir(os.path.join(directory,plant)):
            for file in os.listdir(os.path.join(directory, plant, disease)):

                if os.path.isfile(os.path.join(directory,plant, disease, file)):
                    list_of_images.append(os.path.join(directory,plant,disease, file))    
    return list_of_images

def is_augmented(name):
    for substring in ['Flip', 'Blur', 'Brightness', 'Contrast', 'Crop', 'Distortion', 'Rotation']:
        if substring in name:
            return True
    return False


def distribute(directory):
    dst = 'train_aug'
    list_of_images = walk_through_dir(directory)
    dataframe = create_dataframe(list_of_images)
    dataframe['is_augmented'] = dataframe['filename'].apply(is_augmented)
    for (plant, frame) in dataframe.groupby('plant'):
        X_train, X_test, y_train, y_test = model_selection.train_test_split(frame['filename'], frame['disease'], test_size=0.1, random_state=0, stratify=frame['disease'])
        print(X_train)
        for x_row, y_row in zip(X_train, y_train):
            path = os.path.join(dst, plant, 'train', y_row)
            if not os.path.exists(path):
                os.makedirs(path)
            copy(x_row, path)

        for x_row, y_row in zip(X_test, y_test):
            path = os.path.join(dst, plant, 'val', y_row)
            if not os.path.exists(path):
                os.makedirs(path)
            copy(x_row, path)

def parse_arguments():
    parser = argparse.ArgumentParser(
                    prog='Augmentation',
                    description='Programme that shows augmentation done if passed one image\
                        or creates balanced dataset from directory')
    
    parser.add_argument('directory', metavar='directory', type=str, nargs=1,
                        help='directory to augment information about images from')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    path = args.directory[0]
    if not os.path.exists(path):
        sys.exit('This directory doesn\'t exist or is not accessible')
    distribute(path)