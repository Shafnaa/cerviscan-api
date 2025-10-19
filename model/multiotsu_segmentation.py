"""
============================================
📌 Multi-Otsu Thresholding for Image Segmentation (Python)
============================================

Description:
------------
This program performs image segmentation using the Multi-Otsu thresholding 
method. Multi-Otsu is an extension of Otsu’s method that calculates multiple 
threshold values to separate an image into different classes (regions of 
intensity). The program generates a binary mask that highlights the most 
prominent segmented region in the image.

Features:
---------
✅ Supports grayscale or single-channel images
✅ Uses Multi-Otsu thresholding with customizable number of classes
✅ Converts segmented regions into a clean binary mask (0 and 255 values)
✅ Compatible with scikit-image for easy integration into computer vision pipelines
✅ Useful for preprocessing tasks in machine learning, medical imaging, or object detection

Usage:
------
1. Provide the path of an input image (e.g., "sample_image.png").
2. Call the function `multiotsu_masking(image_path)`.
3. The function returns a binary mask (numpy array) where the segmented region is highlighted.

Example:
--------
python multiotsu_segmentation.py

This will segment the input image using Multi-Otsu method 
and return the binary mask.

Author: Fillipus Aditya Nugroho
============================================
"""

import matplotlib.pyplot as plt
import numpy as np
from skimage import io, img_as_ubyte
from skimage.filters import threshold_multiotsu

def multiotsu_masking(image_path):
    """
    Perform image segmentation using Multi-Otsu thresholding.

    Parameters
    ----------
    image_path : str
        Path to the input image file.

    Returns
    -------
    output : numpy.ndarray
        Binary mask of the segmented image, with values 0 (background) 
        and 255 (foreground region).
    """
    image = io.imread(image_path)
    
    # Compute multi-Otsu thresholds
    threshold = threshold_multiotsu(image, classes=5)

    # Digitize (segment) the image based on the thresholds
    regions = np.digitize(image, bins=threshold)

    # Convert regions to uint8 explicitly to avoid the warning
    output = (regions * (255 // (regions.max() + 1))).astype(np.uint8)
    output[output < np.unique(output)[-1]] = 0
    output[output >= np.unique(output)[-1]] = 255

    return output
