"""
============================================
📌 RGB to Grayscale Image Converter (Python)
============================================

Description:
------------
This program converts an RGB (color) image into a grayscale image 
using OpenCV. Grayscale images are widely used in computer vision 
and machine learning tasks where color information is not essential, 
reducing computational complexity while preserving important 
structural details.

Features:
---------
✅ Converts any RGB image into grayscale
✅ Supports common image formats (.png, .jpg, .jpeg, .bmp, .tiff)
✅ Lightweight and easy to integrate into larger image-processing pipelines
✅ Useful for preprocessing in computer vision or deep learning applications

Usage:
------
1. Provide the path of the input image (e.g., "sample_image.jpg").
2. Call the function `rgb_to_gray_converter(image_path)`.
3. The function returns the grayscale image as a numpy array.

Example:
--------
python rgb_to_gray_converter.py

This will load the input image and return its grayscale version.

Author: Fillipus Aditya Nugroho
============================================
"""

import cv2

def rgb_to_gray_converter(image):
    """
    Convert an RGB image to grayscale.

    Parameters
    ----------
    image : str
        Path to the input RGB image file.

    Returns
    -------
    gray_image : numpy.ndarray
        The converted grayscale image.
    """
    image = cv2.imread(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return gray_image
