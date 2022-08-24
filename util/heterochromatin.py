import numpy as np
import pandas as pd
import cv2 as cv
from skimage import measure
def hetero_euchro_measures(regionmask: np.ndarray, intensity: np.ndarray, alpha: float = 1.0):
    """Computes Heterochromatin to Euchromatin features
    
    This functions obtains the Heterochromatin (high intensity) and Euchromatin (low intensity)
    and computes features that describe the relationship between the two
    
    Args:
        regionmask : binary background mask
        intensity  : intensity image
        alpha     : threshold for calculating heterochromatin intensity
    from :https://github.com/GVS-Lab/chrometrics
    author: Saradha Venkatachalapathy
    """
    high, low = np.percentile(intensity[regionmask], q=(80, 20))
    hc = np.mean(intensity[regionmask]) + (alpha * np.std(intensity[regionmask]))

    feat = {
        "i80_i20": high / low,
        "nhigh_nlow": np.sum(intensity[regionmask] >= high)/ np.sum(intensity[regionmask] <= low),
        "hc_area_ec_area": np.sum(intensity[regionmask] >= hc) / np.sum(intensity[regionmask] < hc),
        "hc_area_nuc_area": np.sum(intensity[regionmask] >= hc) / np.sum(intensity[regionmask] > 0),
        "hc_content_ec_content": np.sum(np.where(intensity[regionmask] >= hc, intensity[regionmask], 0))
            / np.sum(np.where(intensity[regionmask] < hc, intensity[regionmask], 0)),
        "hc_content_dna_content": np.sum(np.where(intensity[regionmask] >= hc, intensity[regionmask], 0))
            / np.sum(np.where(intensity[regionmask] > 0, intensity[regionmask], 0))

    }
    return pd.DataFrame([feat])

def extract_heterochromatin_feature_3D(raw_image,labelled_image,normalize:bool=True):
    if normalize:
        raw_image = cv.normalize(
         raw_image, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F
     )
        raw_image[raw_image < 0] = 0.0
        raw_image[raw_image > 255] = 255.0
        #imwrite(output_dir+'/'+raw_image_pre+'-norm.tif', raw_image, imagej=True) # luezhen added

    # Get features for the individual nuclei in the image
    props = measure.regionprops(labelled_image, raw_image)
    
    all_features = pd.DataFrame()
    for i in range(len(props)):
        all_features = pd.concat([all_features,pd.concat(
        [pd.DataFrame([i + 1], columns=["label"]),
        hetero_euchro_measures(props[i].image, props[i].intensity_image).reset_index(drop=True)
        ],axis=1,
        )],axis=0,ignore_index=True)
    return all_features