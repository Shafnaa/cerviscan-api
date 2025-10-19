"""
============================================
📌 Single Image Feature Extraction (Python)
============================================

Description:
------------
This program extracts multiple color moment and texture features from a single
segmented image. It is designed for website or application use cases where
features from individual images are required for tasks such as image
classification, clustering, or retrieval.

The extracted features include color moments from different color spaces (RGB, YUV, LAB)
and several texture features (GLRLM, TAMURA, LBP). The program returns the
results as a Pandas DataFrame for easy integration into data analysis or machine
learning pipelines.

Features:
---------
✅ Supports extraction of color moments from RGB, YUV, and LAB color spaces
✅ Supports multiple texture descriptors: LBP, GLRLM, TAMURA
✅ Flexible selection of feature types (color spaces and texture methods)
✅ Outputs results as a clean Pandas DataFrame
✅ Automatically removes constant-valued columns

Usage:
------
1. Import the `SingleImageFeatureExtractor` class.
2. Initialize the extractor: `extractor = SingleImageFeatureExtractor()`.
3. Call `extractor.extract_features(image_path)` with the path to the input image.
4. (Optional) Specify desired color spaces or texture feature sets.

Example:
--------
segmented_path = "path/to/segmented_image.png"
extractor = SingleImageFeatureExtractor()
image_features = extractor.extract_features(segmented_path)
print(image_features)

Author: Fillipus Aditya Nugroho
============================================
"""

import pandas as pd

# Import all feature extraction functions & name getters
from model.rgb_color_moment import (
    RGBColorMomentExtractor,
)
from model.yuv_color_moment import (
    YUVColorMomentExtractor,
)
from model.lab_color_moment import (
    LABColorMomentExtractor,
)
from model.glrlm_feature_extraction import (
    GLRLMTextureFeatureExtractor,
)
from model.tamura_feature_extraction import (
    TamuraTextureFeatureExtractor,
)
from model.lbp_feature_extraction import (
    LBPTextureFeatureExtractor,
)


class SingleImageFeatureExtractor:
    """
    Extract multiple color moment and texture features
    for a single input image (website use case).
    """

    def __init__(self, color_moment_features=None, texture_features=None):
        color_moment_features = (
            color_moment_features
            if color_moment_features is not None
            else {
                "RGB": {
                    "r": ["mean", "std", "skew"],
                    "g": ["mean", "std", "skew"],
                    "b": ["mean", "std", "skew"],
                },
                "YUV": {
                    "y": ["mean", "std", "skew"],
                    "u": ["mean", "std", "skew"],
                    "v": ["mean", "std", "skew"],
                },
                "LAB": {
                    "l": ["mean", "std", "skew"],
                    "a": ["mean", "std", "skew"],
                    "b": ["mean", "std", "skew"],
                },
            }
        )

        # Mapping for color moment features
        self.color_moment_features = {
            "RGB": (
                RGBColorMomentExtractor(color_moment_features["RGB"])
                if "RGB" in color_moment_features
                else None
            ),
            "YUV": (
                YUVColorMomentExtractor(color_moment_features["YUV"])
                if "YUV" in color_moment_features
                else None
            ),
            "LAB": (
                LABColorMomentExtractor(color_moment_features["LAB"])
                if "LAB" in color_moment_features
                else None
            ),
        }

        texture_features = (
            texture_features
            if texture_features is not None
            else {
                "GLRLM": {
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
                },
                "TAMURA": ["coarseness", "contrast", "directionality", "roughness"],
                "LBP": ["mean", "median", "std", "kurtosis", "skewness"],
            }
        )

        # Mapping for texture features
        self.texture_features = {
            "GLRLM": (
                GLRLMTextureFeatureExtractor(texture_features["GLRLM"])
                if "GLRLM" in texture_features
                else None
            ),
            "TAMURA": (
                TamuraTextureFeatureExtractor(texture_features["TAMURA"])
                if "TAMURA" in texture_features
                else None
            ),
            "LBP": (
                LBPTextureFeatureExtractor(texture_features["LBP"])
                if "LBP" in texture_features
                else None
            ),
        }

    def extract_features(self, image_path, color_spaces=None, texture_features=None):
        """
        Extract specified features for a single image.

        Parameters
        ----------
        image_path : str
            Path to the segmented input image.
        color_spaces : list of str, optional
            List of color spaces to extract features from.
            Default is ['RGB', 'YUV', 'LAB'].
        texture_features : list of str, optional
            List of texture features to extract.
            Default is ['LBP', 'GLRLM', 'TAMURA'].

        Returns
        -------
        df_features : pd.DataFrame
            Extracted features in a DataFrame (1 row).
        """
        if color_spaces is None:
            color_spaces = []
            if self.color_moment_features["RGB"] is not None:
                color_spaces.append("RGB")
            if self.color_moment_features["YUV"] is not None:
                color_spaces.append("YUV")
            if self.color_moment_features["LAB"] is not None:
                color_spaces.append("LAB")
        if texture_features is None:
            texture_features = []
            if self.texture_features["LBP"] is not None:
                texture_features.append("LBP")
            if self.texture_features["GLRLM"] is not None:
                texture_features.append("GLRLM")
            if self.texture_features["TAMURA"] is not None:
                texture_features.append("TAMURA")

        features = []
        features_name = []

        # Extract color moment features
        for key in color_spaces:
            features.extend(
                self.color_moment_features[key].get_color_moment_features(image_path)
            )
            features_name.extend(
                self.color_moment_features[key].get_color_moment_feature_names()
            )

        # Extract texture features
        for key in texture_features:
            features.extend(self.texture_features[key].get_texture_features(image_path))
            features_name.extend(self.texture_features[key].get_texture_feature_names())

        # Convert to DataFrame
        df_features = pd.DataFrame([features], columns=features_name)
        df_features = df_features.loc[:, (df_features != 1).any()]  # drop constant cols

        return df_features
