#!python
import argparse
import os
from plantcv import plantcv as pcv
import sys


def walk_through_dir(directory):
    list_of_images = []
    for plant in os.listdir(directory):
        # for disease in os.listdir(os.path.join(directory,plant)):
            # for file in os.listdir(os.path.join(directory, plant, disease)):
                if os.path.isfile(os.path.join(directory,plant)):
                    list_of_images.append(os.path.join(directory,plant))
    return list_of_images

def apply_guassian_blur(img):
    s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')
    s_thresh = pcv.threshold.binary(gray_img=s, threshold=40, max_value=255, object_type='light')
    s_mblur = pcv.median_blur(gray_img=s_thresh, ksize=5)
    gaussian_img = pcv.gaussian_blur(img=s_thresh, ksize=(5, 5), sigma_x=0, sigma_y=None)
    return gaussian_img

def apply_mask(img):
    s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')
    s_thresh = pcv.threshold.binary(gray_img=s, threshold=40, max_value=255, object_type='dark')
    s_mblur = pcv.median_blur(gray_img=s_thresh, ksize=5)
    b = pcv.rgb2gray_lab(rgb_img=img, channel='b')

    b_thresh = pcv.threshold.binary(gray_img=b, threshold=120, max_value=255,
                                    object_type='dark')
    bs = pcv.logical_or(bin_img1=s_mblur, bin_img2=b_thresh)
    mask = pcv.invert(bs)

    masked = pcv.apply_mask(img=img, mask=mask, mask_color='white')

    masked_a = pcv.rgb2gray_lab(rgb_img=masked, channel='a')
    masked_b = pcv.rgb2gray_lab(rgb_img=masked, channel='b')

    maskeda_thresh = pcv.threshold.binary(gray_img=masked_a, threshold=115,
                                        max_value=255, object_type='dark')
    maskeda_thresh1 = pcv.threshold.binary(gray_img=masked_a, threshold=135,
                                        max_value=255, object_type='light')
    maskedb_thresh = pcv.threshold.binary(gray_img=masked_b, threshold=128,
                                        max_value=255, object_type='light')
    ab1 = pcv.logical_or(bin_img1=maskeda_thresh, bin_img2=maskedb_thresh)
    ab = pcv.logical_or(bin_img1=maskeda_thresh1, bin_img2=ab1)
    ab_fill = pcv.fill(bin_img=ab, size=200)
    masked = pcv.apply_mask(img=masked, mask=ab_fill, mask_color='white')
    return masked, ab_fill
 
def get_mask(img):
    masked, ab_fill = apply_mask(img)
    return masked

import fnmatch
def find(pattern, path):
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                return os.path.join(root, name)
    return None


def apply_roi_objects(img):
    s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')
    s_thresh = pcv.threshold.binary(gray_img=s, threshold=40, max_value=255, object_type='dark')
    s_mblur = pcv.median_blur(gray_img=s_thresh, ksize=5)
    gaussian_img = pcv.gaussian_blur(img=s_thresh, ksize=(5, 5), sigma_x=0, sigma_y=None)
    b = pcv.rgb2gray_lab(rgb_img=img, channel='b')

    b_thresh = pcv.threshold.binary(gray_img=b, threshold=120, max_value=255,
                                    object_type='dark')
    bs = pcv.logical_or(bin_img1=s_mblur, bin_img2=b_thresh)
    mask = pcv.invert(bs)

    masked = pcv.apply_mask(img=img, mask=mask, mask_color='white')

    masked_a = pcv.rgb2gray_lab(rgb_img=masked, channel='a')
    masked_b = pcv.rgb2gray_lab(rgb_img=masked, channel='b')

    maskeda_thresh = pcv.threshold.binary(gray_img=masked_a, threshold=115,
                                        max_value=255, object_type='dark')
    maskeda_thresh1 = pcv.threshold.binary(gray_img=masked_a, threshold=135,
                                        max_value=255, object_type='light')
    maskedb_thresh = pcv.threshold.binary(gray_img=masked_b, threshold=128,
                                        max_value=255, object_type='light')
    ab1 = pcv.logical_or(bin_img1=maskeda_thresh, bin_img2=maskedb_thresh)
    ab = pcv.logical_or(bin_img1=maskeda_thresh1, bin_img2=ab1)
    clean_mask = pcv.fill(bin_img=ab, size=200)
    pcv.params.debug_outdir = '.'
    objects, object_hierarchy = pcv.find_objects(img, clean_mask)

    contour, hierarchy = pcv.roi.rectangle(img, 0, 0,
                                           img.shape[0],
                                           img.shape[1])
    pcv.params.debug = 'print'
    pcv.roi_objects(img=img, roi_contour=contour, roi_hierarchy=hierarchy,
                    object_contour=objects, obj_hierarchy=object_hierarchy,
                    roi_type='partial')
    roi_file = find("*_obj_on_img.png", '.')
    trash_file = find("*_roi_mask.png", '.')
    transformed_img, path, filename = pcv.readimage(roi_file)
    os.remove(trash_file)
    os.remove(roi_file)
    os.remove('input_image.png')
    return transformed_img

def apply_analysis(img):
    image, mask = apply_mask(img)
    objects, object_hierarchy = pcv.find_objects(img, mask)
    obj, mask = pcv.object_composition(img=img,
                                       contours=objects,
                                       hierarchy=object_hierarchy)
    return pcv.analyze_object(img, obj, mask)

def apply_pseudolandmarks(img):
    pcv.params.debug_outdir = '.'
    image, mask = apply_mask(img)
    objects, object_hierarchy = pcv.find_objects(img, mask)
    obj, mask = pcv.object_composition(img=img,
                                       contours=objects,
                                       hierarchy=object_hierarchy)
    pcv.params.debug = 'print'
    pcv.y_axis_pseudolandmarks(img=img, obj=obj, mask=mask, label="default")
    pcv.params.debug = 'None'
    pseudolandmarks_file = find("*_pseudolandmarks.png", '.')
    transformed_img, path, filename = pcv.readimage(pseudolandmarks_file)
    os.remove(pseudolandmarks_file)
    return transformed_img

def color_histogram(img):
    image, mask = apply_mask(img)
    t_img = pcv.analyze_color(rgb_img=img, mask=mask,
                             colorspaces='all', label="default")
    return t_img

def create_dict(args):
    args_dict = {
        'dst': args.dst,
        'src': args.src,
        'blur': args.blur,
        'mask': args.mask,
        'roi': args.roi,
        'analysis': args.analysis,
        'landmarks': args.landmarks,
        'all': args.all,
        'colors': args.colors,
    }
    return args_dict

def parse_arguments():
    parser = argparse.ArgumentParser(
                    prog='Transformation',
                    description='Program to apply different transformation\
                                 to image or directory of images Usage: ')
    parser.add_argument('-dst',  type=str, default='transformations')
    parser.add_argument('-src',  type=str, default='redistributed')
    parser.add_argument('-blur', action='store_true')
    parser.add_argument('-mask', action='store_true')

    parser.add_argument('-roi', action='store_true')
    parser.add_argument('-analysis', action='store_true')
    parser.add_argument('-landmarks', action='store_true')
    parser.add_argument('-colors', action='store_true')
    parser.add_argument('-all', action='store_true')
    parser.add_argument('-v', action='store_true')
    args = parser.parse_args()
    return create_dict(args)

transformations = {
    "blur": apply_guassian_blur, 
    "mask": get_mask, 
    "roi": apply_roi_objects, 
    "analysis": apply_analysis,
    "landmarks": apply_pseudolandmarks,
    "colors": color_histogram
}

def transform_directory(args):
    list_of_files = walk_through_dir(args['src'])
    for file in list_of_files:
        one_image(file, args)

def one_image(img_path, args):
    if args['all']:
        for key in transformations.keys():
            pcv.params.debug = 'None'
            img, path, filename = pcv.readimage(img_path)
            t_img = transformations[key](img)
            print(os.path.join(args['dst'], filename.split('.')[0],  key, '.png'))
            pcv.print_image(t_img, os.path.join(args['dst'], filename.split('.')[0] + '_' + key + '.png'))
    else:
        for key in transformations.keys():
            if args[key]:
                img, path, filename = pcv.readimage(img_path)
                t_img = transformations[key](img)
                print(os.path.join(args['dst'], filename.split('.')[0],  key, '.png'))
                pcv.print_image(t_img, os.path.join(args['dst'], filename.split('.')[0] + '_' + key + '.png'))


if __name__ == '__main__':
    args = parse_arguments()

    path = args['src']
    if not os.path.exists(path):
        sys.exit('This directory doesn\'t exist or is not accessible')
    if not os.path.exists(args['dst']):
        os.mkdir(args['dst'])
    if os.path.isfile(path):
        one_image(path, args)
    if os.path.isdir(path): 
        transform_directory(args)