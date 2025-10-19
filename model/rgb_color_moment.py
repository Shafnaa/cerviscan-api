"""
============================================
📌 RGB Color Moment Feature Extraction (Python)
============================================

Description:
------------
This program extracts color moment features from an input image in the RGB
color space. Specifically, it calculates the first three color moments
(mean, standard deviation, and skewness) for each of the three RGB channels.
These statistical features are commonly used in image analysis,
pattern recognition, and content-based image retrieval tasks.

Features:
---------
✅ Extracts mean, standard deviation, and skewness from each RGB channel
✅ Ensures image is in RGB format before processing
✅ Returns features in a structured list for further use
✅ Provides a helper function to retrieve ordered feature names

Usage:
------
1. Provide the path to an image file (in RGB format).
2. Call `get_rgb_color_moment_features(image_path)` to extract the features.
3. Use `get_rgb_color_moment_feature_names()` to get the names of the features.

Example:
--------
from rgb_color_moment import get_rgb_color_moment_features, get_rgb_color_moment_feature_names

features = get_rgb_color_moment_features("example.jpg")
feature_names = get_rgb_color_moment_feature_names()

print(dict(zip(feature_names, features)))

Author: Fillipus Aditya Nugroho
============================================
"""

import numpy as np
from scipy.stats import skew
from PIL import Image


class RGBColorMomentExtractor:
    def __init__(self, color_moment_features=None):
        self.color_moment_features = (
            color_moment_features
            if color_moment_features is not None
            else {
                "r": ["mean", "std", "skew"],
                "g": ["mean", "std", "skew"],
                "b": ["mean", "std", "skew"],
            }
        )

    def get_color_moment_feature_names(self):
        """
        Get the list of feature names for the RGB color moment extraction.

        Returns
        -------
        list
            Names for mean, standard deviation, and skewness for each RGB channel.
        """
        return [
            f"{feature}_{channel}"
            for channel in self.color_moment_features.keys()
            for feature in self.color_moment_features[channel]
        ]

    def get_color_moment_features(self, image_path):
        """
        Extract color moment features from an image in the RGB color space.

        Parameters
        ----------
        image_path : str
            Path to the input image file.

        Returns
        -------
        list
            A list containing mean, standard deviation, and skewness for R, G, and B.
            Order: [mean_r, mean_g, mean_b, std_r, std_g, std_b, skew_r, skew_g, skew_b]
        """
        # Open image using PIL
        image = Image.open(image_path)

        # Convert image to numpy array
        image_array = np.array(image)

        # Ensure the image has 3 channels (RGB)
        if len(image_array.shape) < 3 or image_array.shape[2] != 3:
            raise ValueError(f"Image at {image_path} is not in RGB format.")

        extracted_features = {
            "r": [],
            "g": [],
            "b": [],
        }

        for channel in self.color_moment_features.keys():
            for feature in self.color_moment_features[channel]:
                if channel == "r":
                    data = image_array[:, :, 0]
                elif channel == "g":
                    data = image_array[:, :, 1]
                elif channel == "b":
                    data = image_array[:, :, 2]

                if feature == "mean":
                    extracted_features[channel].append(np.mean(data))
                elif feature == "std":
                    extracted_features[channel].append(np.std(data))
                elif feature == "skew":
                    extracted_features[channel].append(skew(data.flatten()))

        # Return list of features in order
        return [
            *extracted_features["r"],
            *extracted_features["g"],
            *extracted_features["b"],
        ]
