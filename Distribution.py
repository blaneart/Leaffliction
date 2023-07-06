import seaborn as sns
import os
import argparse
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import sys
from shutil import copy



def plot_pie(key, data, labels, colors):
    plt.title(key)
    plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
        
    plt.savefig(key + '_pie.png')
    plt.pause(0.0001)
    plt.clf()

def plot_bar(key, data, labels, colors):
    sns.barplot(y=data,x=labels, palette = colors, orient='v').set_title(key)
    plt.savefig(key + '_hist.png')
    plt.pause(0.0001)
    plt.clf()

def create_dataframe(list_of_files):
    dataframe = pd.DataFrame(list_of_files, columns=['filename'])
    dataframe['plant'] = dataframe['filename'].apply(lambda x: x.split('/')[-3])
    dataframe['disease'] = dataframe['filename'].apply(lambda x: x.split('/')[-2])
    return dataframe

def distributed_images(directory):
    list_of_images = []
    for plant in os.listdir(directory):
        for disease in os.listdir(os.path.join(directory,plant)):
            for file in os.listdir(os.path.join(directory, plant, disease)):

                if os.path.isfile(os.path.join(directory,plant, disease, file)):
                    list_of_images.append(os.path.join(directory,plant,disease, file))
    dataframe = create_dataframe(list_of_images)
    colors = sns.color_palette('pastel')[0:5]

    for (plant, frame) in dataframe.groupby('plant'):
        counts = frame.value_counts('disease')       
        print(counts)
        # frame.plot.pie(y='disease', colors=colors)
        plot_pie(plant + 'lol', counts, counts.index, colors)
        plot_bar(plant + 'lol', counts, counts.index, colors)

    # for plant in dataframe['plant'].unique():
    #     plot_pie(plant, )
def not_distributed_images(directory):
    list_of_images = []
    for entry in os.listdir(directory):
        for file in os.listdir(os.path.join(directory,entry)):
            if os.path.isfile(os.path.join(directory,entry, file)):
                list_of_images.append(os.path.join(directory,entry, file))

    dataframe = create_dataframe(list_of_images)

    colors = sns.color_palette('pastel')[0:5]

    for (plant, frame) in dataframe.groupby('plant'):
        counts = frame.value_counts('disease')       
        print(counts)
        # frame.plot.pie(y='disease', colors=colors)
        plot_pie(plant, counts, counts.index, colors)
        plot_bar(plant, counts, counts.index, colors)

        # plt.title(key)
        # plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
        
        # plt.savefig(key + '_pie.png')
        # plt.pause(0.0001)
        # plt.clf()



def parse_arguments():
    parser = argparse.ArgumentParser(
                    prog='Distribution',
                    description='Programme that shows the distribution\
                          of different classes in the image directory')
    parser.add_argument('directory', metavar='directory', type=str, nargs=1,
                        help='directory to parse information about images from')
    args = parser.parse_args()
    return args


def redistribute_files(directory, new_directory_name=None):
    '''
    Given zip archive and subject have different folder structure:
        zip archive: dir/Plant_disease/image.jpg
        subject: dir/Plant/disease/image.jpg
    This function takes directory name and redistributes files in the same way as subject says
    '''
    if new_directory_name and  not os.path.exists(new_directory_name):
        os.mkdir(new_directory_name)

    list_of_images = []
    for entry in os.listdir(directory):
        for file in os.listdir(os.path.join(directory, entry)):
            if os.path.isfile(os.path.join(directory, entry, file)):
                list_of_images.append(os.path.join(directory, entry, file))

    for file in list_of_images:
        plant_and_category = file.split("/")[-2].split("_")
        plant_dir = os.path.join(new_directory_name, plant_and_category[0])
        disease_dir = os.path.join(new_directory_name, plant_and_category[0], '_'.join(plant_and_category[1:]))
        if new_directory_name and  not os.path.exists(plant_dir):
            os.mkdir(plant_dir)
        if new_directory_name and  not os.path.exists(disease_dir):
            os.mkdir(disease_dir)
        copy(file, disease_dir)


if __name__ == "__main__":
    args = parse_arguments()
    path = args.directory[0]
    if not os.path.exists(path):
        sys.exit('This directory doesn\'t exist or is not accessible')
    # not_distributed_images(path)
    # redistribute_files(path, 'images2')
    distributed_images('augmented_directory')
    # for file in walk(path, 1):
    #     print(file)