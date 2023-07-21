import seaborn as sns
import os
import argparse
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import sys
from shutil import copy



def plot_pie(key, data, labels, colors, save_directory):
    plt.title(key)
    plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
        
    plt.savefig(os.path.join(save_directory, key + '_pie.png'))
    plt.pause(0.0001)
    plt.clf()

def plot_bar(key, data, labels, colors, save_directory):
    sns.barplot(y=data,x=labels, palette = colors, orient='v').set_title(key)
    plt.savefig(os.path.join(save_directory, key + '_hist.png'))
    plt.pause(0.0001)
    plt.clf()

def create_dataframe(list_of_files):
    dataframe = pd.DataFrame(list_of_files, columns=['filename'])
    dataframe['plant'] = dataframe['filename'].apply(lambda x: x.split('/')[-3])
    dataframe['disease'] = dataframe['filename'].apply(lambda x: x.split('/')[-2])
    return dataframe

def distributed_images(save_directory, directory):
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
        plot_pie(plant + 'lol', counts, counts.index, colors, save_directory)
        plot_bar(plant + 'lol', counts, counts.index, colors, save_directory)

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
        plot_pie(plant, counts, counts.index, colors)
        plot_bar(plant, counts, counts.index, colors)

def redistribute_files(directory, new_directory_name='new_images'):
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

def create_dict(args):
    args_dict = {
        'directory': args.directory[0],
        'save_directory': args.save_directory,
        'depth': args.depth,
        'redistribute': args.redistribute,
    }
    return args_dict


def parse_arguments():
    parser = argparse.ArgumentParser(
                    prog='Distribution',
                    description='Programme that shows the distribution\
                          of different classes in the image directory')
    parser.add_argument('directory', metavar='directory', type=str, nargs=1,
                        help='directory to parse information about images from')
    parser.add_argument('--save_directory', '-s',  type=str, default='graphs')
    parser.add_argument('--depth', '-d',  type=int, default=3)
    parser.add_argument('--redistributed_dir', '-rd',  type=str, default='graphs')
    parser.add_argument('--redistribute', '-r', action='store_true')
    args = parser.parse_args()
    return create_dict(args)

if __name__ == "__main__":
    args = parse_arguments()
    path = args['directory']
    if not os.path.exists(path):
        sys.exit('This directory doesn\'t exist or is not accessible')
    if args['redistribute']:
        redistribute_files(path, args['redistributed_dir'])
    # not_distributed_images(path)
    distributed_images(args['save_directory'], path)
