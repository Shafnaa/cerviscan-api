import numpy as np
import warnings
from model.GrayRumatrix import getGrayRumatrix

warnings.filterwarnings("ignore")

# def get_glrlm_names(features, degs):
#     """
#     Generate feature names for GLRLM (Gray Level Run Length Matrix).

#     Parameters:
#         features (list): List of feature names.
#         degs (list of lists): List of directional angles (e.g., ['deg0', 'deg45', 'deg90', 'deg135']).

#     Returns:
#         list: Concatenated feature names with directional angles.
#     """
#     glrlm_features_name = []
#     for deg in degs:
#         for feature in features:
#             glrlm_features_name.append(f"{feature}_{deg[0]}")
#     return glrlm_features_name

def get_glrlm_feature_names():
    return ["SRE_deg0", "SRE_deg45", "LRE_deg135", "RLN_deg90", "LGLRE_deg0", "LGLRE_deg90", "LRHGLE_deg0", "LRHGLE_deg90"]

def get_glrlm_features(path, lbp='off'):
    """
    Calculate GLRLM features for an image.

    Parameters:
        path (str): Path to the input image.
        lbp (str, optional): If 'on', apply Local Binary Pattern (LBP) transformation. Defaults to 'off'.

    Returns:
        list: Extracted GLRLM feature values.
    """
    test = getGrayRumatrix()
    test.read_img(path, lbp)

    DEG = [['deg0'], ['deg45'], ['deg90'], ['deg135']]

    glrlm_features_value = []
    
    ["SRE_deg0", "SRE_deg45", "LRE_deg135", "RLN_deg90", "LGLRE_deg0", "LGLRE_deg90", "LRHGLE_deg0", "LRHGLE_deg90"]

    SRE_deg = [["deg0"], ["deg45"]]
    LRE_deg = [["deg135"]]
    RLN_deg = [["deg0"]]
    LGLRE_deg = [["deg0"], ["deg90"]]
    LRHGLE_deg = [["deg0"], ["deg90"]]
    
    for deg in SRE_deg :
        test_data = test.getGrayLevelRumatrix(test.data, deg)
        SRE = test.getShortRunEmphasis(test_data) 
        SRE = float(np.squeeze(SRE))
        
        glrlm_features_value.append(SRE)
        
    for deg in LRE_deg :
        test_data = test.getGrayLevelRumatrix(test.data, deg)
        LRE = test.getShortRunEmphasis(test_data) 
        LRE = float(np.squeeze(LRE))
        
        glrlm_features_value.append(LRE)
        
    for deg in RLN_deg :
        test_data = test.getGrayLevelRumatrix(test.data, deg)
        RLN = test.getShortRunEmphasis(test_data) 
        RLN = float(np.squeeze(RLN))
        
        glrlm_features_value.append(RLN)
    
    for deg in LGLRE_deg:
        test_data = test.getGrayLevelRumatrix(test.data, deg)
        LGLRE = test.getShortRunEmphasis(test_data) 
        LGLRE = float(np.squeeze(LGLRE))
        
        glrlm_features_value.append(LGLRE)
    
    for deg in LRHGLE_deg:
        test_data = test.getGrayLevelRumatrix(test.data, deg)
        LRHGLE = test.getShortRunEmphasis(test_data) 
        LRHGLE = float(np.squeeze(LRHGLE))
        
        glrlm_features_value.append(LRHGLE)

    return glrlm_features_value