import os
from shutil import copy
import sys


def redistribute_files(directory, new_directory_name='new_images'):
    '''
    Given zip archive and subject have different folder structure:
        zip archive: dir/Plant_disease/image.jpg
        subject: dir/Plant/disease/image.jpg
    This function takes directory name and redistributes files in the same way
      as subject says
    '''
    if new_directory_name and not os.path.exists(new_directory_name):
        os.mkdir(new_directory_name)

    list_of_images = []
    for entry in os.listdir(directory):
        for file in os.listdir(os.path.join(directory, entry)):
            if os.path.isfile(os.path.join(directory, entry, file)):
                list_of_images.append(os.path.join(directory, entry, file))

    for file in list_of_images:
        plant_and_category = file.split("/")[-2].split("_")
        plant_dir = os.path.join(new_directory_name, plant_and_category[0])
        disease_dir = os.path.join(new_directory_name, plant_and_category[0],
                                   '_'.join(plant_and_category[1:]))
        if new_directory_name and not os.path.exists(plant_dir):
            os.mkdir(plant_dir)
        if new_directory_name and not os.path.exists(disease_dir):
            os.mkdir(disease_dir)
        copy(file, disease_dir)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python redistribute.py directory new_directory")
        exit()
    directory = sys.argv[1]
    new_directory = sys.argv[2]
    redistribute_files(directory, new_directory)
