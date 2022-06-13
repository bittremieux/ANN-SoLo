import numpy as np

from .tools import matched_peaks_with_intensity_info, matched_peaks_with_mz_and_intensity_info
from . import math_distance
from spectrum_utils.spectrum import MsmsSpectrum


def weighted_dot_product_distance(spectrum_query: MsmsSpectrum,
                                  spectrum_library: MsmsSpectrum = None,
                                  matched_peaks: []= None): -> float
    r"""
    Weighted Dot-Product distance:

    .. math::

        1 - \frac{(\sum{Q^{'}_{i} P^{'}_{i}})^2}{\sum{Q_{i}^{'2}\sum P_{i}^{'2}}}, here:

        P^{'}_{i} = M_{p,i}^{3}I_{p,i}^{0.6}, Q^{'}_{i} = M_{q,i}^{3}I_{q,i}^{0.6}


    """

    spec_matched = matched_peaks_with_mz_and_intensity_info( spectrum_query=spectrum_query,
                                                           spectrum_library=spectrum_library,
                                                           matched_peaks=matched_peaks)
    m_q = spec_matched[0, :]
    i_q = spec_matched[1, :]
    m_r = spec_matched[2, :]
    i_r = spec_matched[3, :]
    k = 0.6
    l = 3
    w_q = np.power(i_q, k) * np.power(m_q, l)
    w_r = np.power(i_r, k) * np.power(m_r, l)

    return math_distance.dot_product_distance(w_q, w_r)


def ms_for_id_v1_distance(spectrum_query: MsmsSpectrum,
                          spectrum_library: MsmsSpectrum = None,
                          matched_peaks: []= None): -> float
    r"""
	reference: https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/jms.1525
	
    MSforID distance version 1:

    .. math::

        Similarity = \frac{N_m^4}{N_qN_r(\sum|I_{q,i}-I_{r,i}|)^a}\ ,\ a=0.25

        Distance = \frac{1}{Similarity}

    :math:`N_m`: number of matching fragments, :math:`N_q, N_r`: number of fragments for spectrum p,q
    :return: :math:`Distance`
    """

    spec_matched = matched_peaks_with_intensity_info(spectrum_query=spectrum_query,
                                                   spectrum_library=spectrum_library,
                                                    matched_peaks=matched_peaks)

    i_q = spec_matched[0, :]
    i_r = spec_matched[1, :]

    n_m = np.sum(np.bitwise_and(i_q > 0, i_r > 0))
    n_q = np.sum(i_q > 0)
    n_r = np.sum(i_r > 0)

    a = 0.25
    x = n_m ** 4
    y = n_q * n_r * np.power(np.sum(np.abs(i_q - i_r)), a)

    if x == 0:
        dist = np.inf
    else:
        dist = y / x

    return dist


def ms_for_id_distance(spectrum_query: MsmsSpectrum,
                       spectrum_library: MsmsSpectrum = None,
                       matched_peaks: []= None): -> float
    r"""
	reference: https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/jms.1525
	
    MSforID distance:

    .. math::

        -\frac{N_m^b(\sum I_{q,i}+2\sum I_{r,i})^c}{(N_q+2N_r)^d+\sum|I_{q,i}-I_{r,i}|+\sum|M_{q,i}-M_{r,i}|},\ \ b=4,\ c=1.25,\ d=2

    The peaks have been filtered with intensity > 0.05.

    :math:`N_m`: number of matching fragments,

    :math:`N_q, N_r`: number of fragments for spectrum p,q,

    :math:`M_q,M_r`: m/z of peak in query and reference spectrum,

    :math:`I_q,I_r`: intensity of peak in query and reference spectrum
    """

    spec_matched = matched_peaks_with_mz_and_intensity_info(spectrum_query=spectrum_query,
                                                          spectrum_library=spectrum_library,
                                                          matched_peaks=matched_peaks)

    b = 4
    c = 1.25
    d = 2

    i_q = spec_matched[1, :]
    i_r = spec_matched[3, :]
    matched_peak = np.bitwise_and(i_q > 0, i_r > 0)
    n_m = np.sum(matched_peak)
    n_q = np.sum(i_q > 0)
    n_r = np.sum(i_r > 0)
    i_delta = (i_q - i_r)[matched_peak]
    m_delta = (spec_matched[0, :] - spec_matched[2, :])[matched_peak]

    s1 = np.power(n_m, b) * np.power(np.sum(i_q) + 2 * np.sum(i_r), c)
    s2 = np.power(n_q + 2 * n_r, d) + \
        np.sum(np.abs(i_delta)) + \
        np.sum(np.abs(m_delta))

    if s2 == 0:
        similarity = 0.
    else:
        similarity = s1 / s2
    return -similarity
