"""
============================================
📌 Image Segmentation using Mask Overlay (Python)
============================================

Description:
------------
This program applies a binary or grayscale mask to an original image to generate 
a segmented output image. The mask highlights specific regions of interest 
(e.g., objects, lesions, or areas for further analysis), while non-relevant 
areas are suppressed.

It is particularly useful for preprocessing tasks in computer vision and 
machine learning pipelines where only the masked region of an image should be 
considered for feature extraction or model training.

Features:
---------
✅ Supports grayscale masks for segmentation  
✅ Ensures mask is automatically resized to match the input image  
✅ Converts mask to 3 channels for compatibility with original RGB images  
✅ Performs bitwise AND operation to retain only masked regions  
✅ Returns the segmented image for further processing  

Usage:
------
1. Provide the original image (as a NumPy array, e.g., loaded with OpenCV).  
2. Provide the path to the mask image (binary or grayscale).  
3. Call the function `get_segmented_image(original_image, mask_path)`.  
4. The function returns the segmented output image.  

Example:
--------
import cv2  
from segmentation import get_segmented_image  

original = cv2.imread("image.png")  
segmented = get_segmented_image(original, "mask.png")  
cv2.imwrite("segmented_output.png", segmented)  

Author: Fillipus Aditya Nugroho
============================================
"""

import cv2

def get_segmented_image(original_image, mask_path):
    """
    Apply a mask to an original image to generate a segmented output.

    Parameters
    ----------
    original_image : numpy.ndarray
        The original input image (typically loaded using cv2.imread).
    mask_path : str
        Path to the mask image (binary or grayscale). The mask will be resized 
        to match the dimensions of the original image.

    Returns
    -------
    dest_and : numpy.ndarray
        The segmented image, where only the masked regions from the original 
        image are retained, and other areas are suppressed.
    """
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Ensure the mask image has the same dimensions as the original image
    mask_image = cv2.resize(mask_image, (original_image.shape[1], original_image.shape[0]))

    # Convert the grayscale mask to a 3-channel image
    mask_image_3channel = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2RGB)

    # Perform bitwise AND operation
    dest_and = cv2.bitwise_and(original_image, mask_image_3channel)

    return dest_and
