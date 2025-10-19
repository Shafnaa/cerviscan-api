"""
============================================
📌 GLRLM (Gray Level Run Length Matrix) Feature Extraction (Python)
============================================

Description:
------------
This program extracts GLRLM (Gray Level Run Length Matrix) features from a given image.
GLRLM is a texture analysis method widely used in image processing and pattern
recognition tasks, particularly in medical imaging and computer vision.

It supports computing features in multiple directions (0°, 45°, 90°, 135°) and provides
commonly used statistical measures such as Short Run Emphasis, Long Run Emphasis,
Gray Level Non-Uniformity, Run Percentage, and more.

Features:
---------
✅ Extracts 11 GLRLM texture features per direction
✅ Supports 4 directional angles: 0°, 45°, 90°, 135°
✅ Optional Local Binary Pattern (LBP) preprocessing
✅ Returns both feature values and feature names with direction labels
✅ Easily integrable with machine learning pipelines

Usage:
------
1. Provide the path to the input image.
2. Optionally enable Local Binary Pattern (LBP) preprocessing by setting `lbp='on'`.
3. Call `get_glrlm_features(path)` to extract GLRLM features.
4. Use `get_glrlm_feature_names()` to get the corresponding feature names.

Example:
--------
from glrlm_feature_extraction import get_glrlm_features, get_glrlm_feature_names

features = get_glrlm_features("sample_image.jpg", lbp='off')
feature_names = get_glrlm_feature_names()

print(dict(zip(feature_names, features)))

Author: Fillipus Aditya Nugroho
============================================
"""

import numpy as np
import warnings
from model.GrayRumatrix import getGrayRumatrix

# Suppress warnings to keep output clean
warnings.filterwarnings("ignore")


class GLRLMTextureFeatureExtractor:
    def __init__(self, texture_features=None):
        self.texture_features = (
            texture_features
            if texture_features is not None
            else {
                "deg0": [
                    "SRE",
                    "LRE",
                    "GLN",
                    "RLN",
                    "RP",
                    "LGLRE",
                    "HGL",
                    "SRLGLE",
                    "SRHGLE",
                    "LRLGLE",
                    "LRHGLE",
                ],
                "deg45": [
                    "SRE",
                    "LRE",
                    "GLN",
                    "RLN",
                    "RP",
                    "LGLRE",
                    "HGL",
                    "SRLGLE",
                    "SRHGLE",
                    "LRLGLE",
                    "LRHGLE",
                ],
                "deg90": [
                    "SRE",
                    "LRE",
                    "GLN",
                    "RLN",
                    "RP",
                    "LGLRE",
                    "HGL",
                    "SRLGLE",
                    "SRHGLE",
                    "LRLGLE",
                    "LRHGLE",
                ],
                "deg135": [
                    "SRE",
                    "LRE",
                    "GLN",
                    "RLN",
                    "RP",
                    "LGLRE",
                    "HGL",
                    "SRLGLE",
                    "SRHGLE",
                    "LRLGLE",
                    "LRHGLE",
                ],
            }
        )

    def get_texture_feature_names(self):
        """
        Get the list of feature names for the GLRLM texture extraction.

        Returns
        -------
        list
            Names for each GLRLM feature combined with direction labels.
        """
        glrlm_feature_names = []

        # For each direction and feature, create a combined name
        for deg, features in self.texture_features.items():
            for feature in features:
                glrlm_feature_names.append(f"{feature}_{deg}")
        return glrlm_feature_names

    def get_texture_features(self, image_path, lbp="off"):
        """
        Extract GLRLM texture features from an image.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        lbp : str, optional
            If 'on', apply Local Binary Pattern (LBP) transformation before computing GLRLM.
            Defaults to 'off'.

        Returns
        -------
        list
            List of extracted GLRLM feature values for each specified direction.
        """
        # Initialize GLRLM processing object
        test = getGrayRumatrix()

        # Read and preprocess image (optionally apply LBP)
        test.read_img(image_path, lbp)

        extracted_features = []

        for deg, features in self.texture_features.items():
            test_data = test.getGrayLevelRumatrix(test.data, [deg])

            if "SRE" in features:
                SRE = test.getShortRunEmphasis(test_data)
                extracted_features.append(float(np.squeeze(SRE)))
            if "LRE" in features:
                LRE = test.getLongRunEmphasis(test_data)
                extracted_features.append(float(np.squeeze(LRE)))
            if "GLN" in features:
                GLN = test.getGrayLevelNonUniformity(test_data)
                extracted_features.append(float(np.squeeze(GLN)))
            if "RLN" in features:
                RLN = test.getRunLengthNonUniformity(test_data)
                extracted_features.append(float(np.squeeze(RLN)))
            if "RP" in features:
                RP = test.getRunPercentage(test_data)
                extracted_features.append(float(np.squeeze(RP)))
            if "LGLRE" in features:
                LGLRE = test.getLowGrayLevelRunEmphasis(test_data)
                extracted_features.append(float(np.squeeze(LGLRE)))
            if "HGL" in features:
                HGL = test.getHighGrayLevelRunEmphais(test_data)
                extracted_features.append(float(np.squeeze(HGL)))
            if "SRLGLE" in features:
                SRLGLE = test.getShortRunLowGrayLevelEmphasis(test_data)
                extracted_features.append(float(np.squeeze(SRLGLE)))
            if "SRHGLE" in features:
                SRHGLE = test.getShortRunHighGrayLevelEmphasis(test_data)
                extracted_features.append(float(np.squeeze(SRHGLE)))
            if "LRLGLE" in features:
                LRLGLE = test.getLongRunLow(test_data)
                extracted_features.append(float(np.squeeze(LRLGLE)))
            if "LRHGLE" in features:
                LRHGLE = test.getLongRunHighGrayLevelEmphasis(test_data)
                extracted_features.append(float(np.squeeze(LRHGLE)))

        return extracted_features
