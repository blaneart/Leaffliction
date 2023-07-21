import numpy as np
import cv2 
from plantcv import plantcv as pcv


def preprocess(image):
    #Converting to numpy array from numpy tensor with rank 3
    image = np.array(image, dtype=np.uint8)
    #Gaussian Blur
    gaussian_blur = cv2.GaussianBlur(image,(5,5),0)
    img = np.asarray(gaussian_blur, dtype=np.float64)
    return img
image= 'images/Apple_scab/image (100).JPG'
# img = preprocess(cv2.imread(image))
img, path, filename = pcv.readimage(filename=image)


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
# mask = pcv.erode(gray_img=mask, ksize=3, i=1)
# gaussian_blur = pcv.gaussian_blur(mask, ksize=(5, 5))
# masked = pcv.apply_mask(img=img, mask=mask, mask_color='white')

# mask = pcv.invert(s_thresh)
# mask = pcv.erode(gray_img=mask, ksize=3, i=1)
# gaussian_blur = pcv.gaussian_blur(mask, ksize=(5, 5))
# masked = pcv.apply_mask(img=img, mask=mask, mask_color='white')

pcv.print_image(b_thresh, "b_thresh.png")

pcv.print_image(masked, "test_2.png")

cv2.imwrite('test.png', s_thresh)

