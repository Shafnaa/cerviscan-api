"""
============================================
📌 LAB Color Moment Feature Extraction (Python)
============================================

Description:
------------
This program extracts **color moment features** from images in the LAB color space.
Color moments are statistical measures (mean, standard deviation, skewness) that
summarize the color distribution of an image. They are widely used in image
processing, computer vision, and machine learning tasks such as image retrieval,
classification, and object recognition.

Features:
---------
✅ Converts images to LAB color space using scikit-image
✅ Computes mean, standard deviation, and skewness for each LAB channel
✅ Returns features as a list for easy integration with ML pipelines
✅ Includes a helper function to provide descriptive feature names

Usage:
------
1. Provide an image path as input to `get_lab_color_moment_features()`.
2. The function will return a list of 9 extracted features.
3. Use `get_lab_color_moment_feature_names()` to get the ordered names
   corresponding to the extracted features.

Example:
--------
features = get_lab_color_moment_features("sample_image.jpg")
feature_names = get_lab_color_moment_feature_names()

print(dict(zip(feature_names, features)))

Author: Fillipus Aditya Nugroho
============================================
"""

import numpy as np
import skimage
import cv2
from scipy.stats import skew


class LABColorMomentExtractor:
    def __init__(self, color_moment_features=None):
        self.color_moment_features = (
            color_moment_features
            if color_moment_features is not None
            else {
                "l": ["mean", "std", "skew"],
                "a": ["mean", "std", "skew"],
                "b": ["mean", "std", "skew"],
            }
        )

    def get_color_moment_feature_names(self):
        """
        Get the list of feature names for the LAB color moment extraction.

        Returns
        -------
        list
            Names for mean, standard deviation, and skewness for each LAB channel.
        """
        return [
            f"{feature}_{channel}"
            for channel, features in self.color_moment_features.items()
            for feature in features
        ]

    def get_color_moment_features(self, image_path):
        """
        Extract color moment features from an image in the LAB color space.

        Parameters
        ----------
        image_path : str
            Path to the image file to be analyzed.

        Returns
        -------
        list
            A list of 9 extracted statistical features in the following order:
            - Mean (L, A, B channels)
            - Standard deviation (L, A, B channels)
            - Skewness (L, A, B channels)
        """
        # Read the image from file (BGR format by default in OpenCV)
        image = cv2.imread(image_path)

        # Convert BGR to RGB for correct color conversion
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the RGB image to a numpy array
        image_array = np.array(rgb_image)

        # Normalize RGB values to [0, 1] for skimage compatibility
        rgb_img_normalized = [
            [[element / 255 for element in sublist] for sublist in inner_list]
            for inner_list in image_array
        ]

        # Convert normalized RGB image to LAB color space using skimage
        lab_image = skimage.color.rgb2lab(rgb_img_normalized)

        extracted_features = {
            "l": [],
            "a": [],
            "b": [],
        }

        for channel in self.color_moment_features.keys():
            for feature in self.color_moment_features[channel]:
                if channel == "l":
                    data = lab_image[:, :, 0]
                elif channel == "a":
                    data = lab_image[:, :, 1]
                elif channel == "b":
                    data = lab_image[:, :, 2]

                if feature == "mean":
                    extracted_features[channel].append(np.mean(data))
                elif feature == "std":
                    extracted_features[channel].append(np.std(data))
                elif feature == "skew":
                    extracted_features[channel].append(skew(data.flatten()))

        # Return all color moment features as a list
        return [
            *extracted_features["l"],
            *extracted_features["a"],
            *extracted_features["b"],
        ]
