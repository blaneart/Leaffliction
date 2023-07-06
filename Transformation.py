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
image= '/home/blanar/cv/artem/kaggle/leaf/images/Apple_Black_rot/image (2).JPG'
img = preprocess(cv2.imread('/home/blanar/cv/artem/kaggle/leaf/images/Apple_Black_rot/image (2).JPG'))
img, path, filename = pcv.readimage(filename=image)

s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')
s_thresh = pcv.threshold.binary(gray_img=s, threshold=60, max_value=255, object_type='dark')
mask = pcv.invert(s_thresh)
mask = pcv.erode(gray_img=mask, ksize=3, i=1)
gaussian_blur = pcv.gaussian_blur(mask, ksize=(3, 3))
masked = pcv.apply_mask(img=img, mask=mask, mask_color='white')
pcv.print_image(masked, "test_2.png")

cv2.imwrite('test.png', s_thresh)

