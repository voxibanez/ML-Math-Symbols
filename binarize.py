# -*- coding: utf-8 -*-
"""
The this file contains code to convert a colored image to a pure
black and white image. 
"""

from PIL import Image
from scipy.misc import imsave
import numpy

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
    monochrome_image = numpy.array(monochrome_image)
    
    #binarize array
    binary_image_array = make_binary_array(monochrome_image, threshold)
    
    #save the image
    imsave(dest, binary_image_array)
    
def make_binary_array(convert_array, theshold=200)
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
                convert_array[i][j] = 255
            else:
                convert_array[i][j] = 0
    return convert_array

def main():
    image_to_binary('test.png', 'new.png', 200)
    
    
if __name__ == '__main'__':
    main()