import numpy as np

from . import tools_fast
from spectrum_utils.spectrum import MsmsSpectrum



def matched_peaks_with_intensity_info(spectrum_query: MsmsSpectrum,
                                      spectrum_library: MsmsSpectrum = None,
                                      matched_peaks: []= None):
    """
    Retreive matched peaks intensities for both query and library spectra
    :return: list. Each element in the list is a list contain three elements:
                              m/z, intensity from spec 1; intensity from spec 2.
    """
    matched_qs_intensities = np.empty(0)
    matched_ls_intensities = np.empty(0)
    for qsi, lsi in matched_peaks:
        matched_qs_intensities = np.append(matched_qs_intensities,[spectrum_query.intensity[qsi]])
        matched_ls_intensities = np.append(matched_ls_intensities,[spectrum_library.intensity[lsi]])
    return np.stack((matched_qs_intensities,matched_ls_intensities), axis=0)

def matched_peaks_with_mz_and_intensity_info(spectrum_query: MsmsSpectrum,
                                             spectrum_library: MsmsSpectrum= None,
                                             matched_peaks: []= None):
    """
    Retreive matched peaks m/z and intensities for both query and library spectra
    :return: list. Each element in the list is a list contain three elements:
                              m/z from spec 1; intensity from spec 1; m/z from spec 2; intensity from spec 2.
    """
    matched_qs_MZs = np.empty(0)
    matched_ls_MZs = np.empty(0)
    matched_qs_intensities = np.empty(0)
    matched_ls_intensities = np.empty(0)
    for qsi, lsi in matched_peaks:
        matched_qs_MZs = np.append(matched_qs_MZs,
								   [spectrum_query.mz[qsi]])
        matched_qs_intensities = np.append(matched_qs_intensities,
										   [spectrum_query.intensity[qsi]])
        matched_ls_MZs = np.append(matched_ls_MZs,
										   [spectrum_library.mz[lsi]])
        matched_ls_intensities = np.append(matched_ls_intensities,
										   [spectrum_library.intensity[lsi]])
    return np.stack((matched_qs_MZs,matched_qs_intensities,matched_ls_MZs,matched_ls_intensities), axis=0)



def normalize_distance(dist, dist_range):
    if dist_range[1] == np.inf:
        if dist_range[0] == 0:
            result = 1 - 1 / (1 + dist)
        elif dist_range[1] == 1:
            result = 1 - 1 / dist
        else:
            raise NotImplementedError()
    elif dist_range[0] == -np.inf:
        if dist_range[1] == 0:
            result = -1 / (-1 + dist)
        else:
            raise NotImplementedError()
    else:
        result = (dist - dist_range[0]) / (dist_range[1] - dist_range[0])

    if result < 0:
        result = 0.
    elif result > 1:
        result = 1.

    return result
