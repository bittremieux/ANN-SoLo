from typing import Union

import numpy as np

from . import math_distance, ms_distance
from .tools import matched_peaks_with_intensity_info, normalize_distance
from spectrum_utils.spectrum import MsmsSpectrum

methods_name = {
    "entropy": "Entropy distance",
    "unweighted_entropy": "Unweighted entropy distance",
    "euclidean": "Euclidean distance",
    "manhattan": "Manhattan distance",
    "chebyshev": "Chebyshev distance",
    "squared_euclidean": "Squared Euclidean distance",
    "fidelity": "Fidelity distance",
    "matusita": "Matusita distance",
    "squared_chord": "Squared-chord distance",
    "bhattacharya_1": "Bhattacharya 1 distance",
    "bhattacharya_2": "Bhattacharya 2 distance",
    "harmonic_mean": "Harmonic mean distance",
    "probabilistic_symmetric_chi_squared": "Probabilistic symmetric χ2 distance",
    "ruzicka": "Ruzicka distance",
    "roberts": "Roberts distance",
    "intersection": "Intersection distance",
    "motyka": "Motyka distance",
    "canberra": "Canberra distance",
    "baroni_urbani_buser": "Baroni-Urbani-Buser distance",
    "penrose_size": "Penrose size distance",
    "mean_character": "Mean character distance",
    "lorentzian": "Lorentzian distance",
    "penrose_shape": "Penrose shape distance",
    "clark": "Clark distance",
    "hellinger": "Hellinger distance",
    "whittaker_index_of_association": "Whittaker index of association distance",
    "symmetric_chi_squared": "Symmetric χ2 distance",
    "pearson_correlation": "Pearson/Spearman Correlation Coefficient",
    "improved_similarity": "Improved Similarity",
    "absolute_value": "Absolute Value Distance",
    "cosine": "Cosine distance",
    "spectral_contrast_angle": "Spectral Contrast Angle",
    "wave_hedges": "Wave Hedges distance",
    "jaccard": "Jaccard distance",
    "dice": "Dice distance",
    "inner_product": "Inner product distance",
    "divergence": "Divergence distance",
    "vicis_symmetric_chi_squared_3": "Vicis-Symmetric χ2 3 distance",
    "ms_for_id_v1": "MSforID distance version 1",
    "ms_for_id": "MSforID distance",
    "weighted_dot_product": "Weighted dot product distance",
}

methods_range = {
    "entropy": [0, np.log(4)],
    "unweighted_entropy": [0, np.log(4)],
    "absolute_value": [0, 2],
    "bhattacharya_1": [0, np.arccos(0) ** 2],
    "bhattacharya_2": [0, np.inf],
    "canberra": [0, np.inf],
    "clark": [0, np.inf],
    "divergence": [0, np.inf],
    "euclidean": [0, np.sqrt(2)],
    "hellinger": [0, np.inf],
    "improved_similarity": [0, np.inf],
    "lorentzian": [0, np.inf],
    "manhattan": [0, 2],
    "matusita": [0, np.sqrt(2)],
    "mean_character": [0, 2],
    "motyka": [-0.5, 0],
    "ms_for_id": [-np.inf, 0],
    "ms_for_id_v1": [0, np.inf],
    "pearson_correlation": [-1, 1],
    "penrose_shape": [0, np.sqrt(2)],
    "penrose_size": [0, np.inf],
    "probabilistic_symmetric_chi_squared": [0, 1],
    "similarity_index": [0, np.inf],
    "squared_chord": [0, 2],
    "squared_euclidean": [0, 2],
    "symmetric_chi_squared": [0, 0.5 * np.sqrt(2)],
    "vicis_symmetric_chi_squared_3": [0, 2],
    "wave_hedges": [0, np.inf],
    "whittaker_index_of_association": [0, np.inf]
}


def all_similarity(spectrum_query: MsmsSpectrum,
                     spectrum_library: MsmsSpectrum = None,
                    matched_peaks: []= None) -> dict:
    """
    Calculate all the similarity between two spectra, find common peaks.
    If both ms2_ppm and ms2_da is defined, ms2_da will be used.

    :param spectrum_query: The query spectrum, need to be in numpy array format.
    :param spectrum_library: The library spectrum, need to be in numpy array format.
    :param ms2_ppm: The MS/MS tolerance in ppm.
    :param ms2_da: The MS/MS tolerance in Da.
    :param need_clean_spectra: Normalize spectra before comparing, required for not normalized spectrum.
    :param need_normalize_result: Normalize the result into [0,1].
    :return: A dict contains all similarity.
    """
    # Calculate similarity
    result = {}
    if matched_peaks is not None and len(matched_peaks) > 0:
        spec_matched = matched_peaks_with_intensity_info(
										spectrum_query=spectrum_query,
										spectrum_library=spectrum_library,
										matched_peaks=matched_peaks)

        for method in methods_name:
            function_name = method + "_distance"
            if hasattr(math_distance, function_name):
                f = getattr(math_distance, function_name)
                dist = f(spec_matched[0, :], spec_matched[1, :])
            elif hasattr(ms_distance, function_name):
                f = getattr(ms_distance, function_name)
                dist = f(spectrum_query, spectrum_library, matched_peaks)

        # Normalize result
            if method not in methods_range:
                dist_range = [0, 1]
            else:
                dist_range = methods_range[method]

            result[method] = normalize_distance(dist, dist_range)
        result[method] = dist
    else:
        for method in methods_name:
            result[method] = 1

    return result

