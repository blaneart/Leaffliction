#!python
import seaborn as sns
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import sys
from shutil import copy

from time import sleep

def plot(fig, axes, key, data, labels, colors):
    """
    Creates a pie chart and a pie chart for the given data
    """
    
    ax1, ax2 = axes

    ax1.pie(data, labels=labels, colors=colors, autopct='%.0f%%')
    ax1.set_title(key)
    ax2.bar(x=labels, height=data, color=colors)
    ax2.set_title(key)



def create_dataframe(list_of_files):
    dataframe = pd.DataFrame(list_of_files, columns=['filename'])
    dataframe['plant'] = dataframe['filename'].apply(lambda x:
                                                     x.split('/')[-3])
    dataframe['disease'] = dataframe['filename'].apply(lambda x:
                                                       x.split('/')[-2])
    return dataframe


def distributed_images(save_directory, directory):
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)    
    list_of_images = []
    for plant in os.listdir(directory):
        for disease in os.listdir(os.path.join(directory, plant)):
            for file in os.listdir(os.path.join(directory, plant, disease)):

                if os.path.isfile(os.path.join(directory,
                                               plant, disease, file)):
                    list_of_images.append(os.path.join(directory,
                                                       plant, disease, file))
    dataframe = create_dataframe(list_of_images)
    colors = sns.color_palette('pastel')[0:5]

    plant_count = dataframe['plant'].nunique()

    fig, axes = plt.subplots(plant_count, 2)

    fig.tight_layout(pad=7.0)

    i = 0
    for (plant, frame) in dataframe.groupby('plant'):
        counts = frame.value_counts('disease')

        plot(fig, axes[i], plant, counts, counts.index, colors)
        i += 1

    plt.show()


def create_dict(args):
    args_dict = {
        'directory': args.directory[0],
        'save_directory': args.save_directory,
        'depth': args.depth,
    }
    return args_dict


def parse_arguments():
    parser = argparse.ArgumentParser(
                    prog='Distribution',
                    description='Programme that shows the distribution\
                          of different classes in the image directory')
    parser.add_argument('directory', metavar='directory',
                        type=str, nargs=1,
                        help='directory to parse'
                        'information about images from')
    parser.add_argument('--save_directory', '-s',  type=str, default='graphs')
    parser.add_argument('--depth', '-d',  type=int, default=3)

    args = parser.parse_args()
    return create_dict(args)


if __name__ == "__main__":
    args = parse_arguments()
    path = args['directory']

    if not os.path.exists(path):
        sys.exit('This directory doesn\'t exist or is not accessible')

    plt.rcParams["figure.figsize"] = (10, 10)
    # not_distributed_images(path)
    distributed_images(args['save_directory'], path)


