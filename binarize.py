# -*- coding: utf-8 -*-
"""
The this file contains code to convert a colored image to a pure
black and white image.
"""

from PIL import Image
from scipy.misc import imsave, imresize, toimage
import os
import cv2
import numpy as np

img = cv2.imread('test2.png')
img_bw = 255*(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 5).astype('uint8')

se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)

mask = np.dstack([mask, mask, mask]) / 255
out = img * mask


cv2.imwrite(os.path.join('img','output.png'), out)




def image_to_binary(src, dest, threshold):
    """
    The purpose of this method is to take the image at the src path
    and save it as a black and white only image at the destination path.

    args:
        -src (string): path to image to convert
        -dest (string): path to save the image too
        -threshold (integer): threshold value for pixels (where the white
                                and black cut-off is)

    return:
        -N/A
    """
    image_to_convert = Image.open(src)

    #convert the image to monochrome
    monochrome_image = image_to_convert.convert('L')

    #get pixel array
    monochrome_image = np.array(monochrome_image)

    #binarize array
    binary_image_array = make_binary_array(monochrome_image, threshold)

    return binary_image_array

def make_binary_array(convert_array, threshold=200):
    """
    The purpose of this function is to convert an array of pixel values
    to a binary array based on the threshold.

    args:
        -convert_array (numpy array): the array to convert
        -theshold (integer) (default=200): the cuttoff value for pixels

    return:
        -convert_array (numpy array): the converted binary array:
    """
    for i in range(len(convert_array)):
        for j in range(len(convert_array[0])):
            if convert_array[i][j] > threshold:
                convert_array[i][j] = 0
            else:
                convert_array[i][j] = 255
    return convert_array


def image_resize(img, size, interp='bilinear', mode=None):
    """
    This method takes in an array of images and resizes them.

    args:
        -img (img) The image to be resized.
        -size (int, float or tuple)
            -int - Percentage of current size.
            -float - Fraction of current size.
            -tuple - Size of the output image.
        -interp (str) (default=bilinear): Interpolation to use for re-sizing
                       (‘nearest’, ‘lanczos’, ‘bilinear’, ‘bicubic’ or ‘cubic’)
        -mode (str): (default):The PIL image mode (‘P’, ‘L’, etc.) to convert
                     arr before resizing.

    return:
        - array of resized images
    """
    return imresize(image, size, interp='bilinear', mode=None)
    
    
def main():
    bin_arr = image_to_binary('test2.png', os.path.join('img','new.png'), 50)

    imsave(os.path.join('img','binary_img.png'), bin_arr)

    res_img = image_resize([bin_arr], (100, 100))
    imsave(os.path.join('img','res_img.png'), res_img[0])
    
    img = cv2.imread(os.path.join('img','res_img.png'))

    dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    cv2.imwrite(os.path.join('img','output2.png'), dst)


if __name__ == '__main__':
    main()
