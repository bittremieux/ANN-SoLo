import numpy as np
import scipy.special
import scipy.stats

from ann_solo import config, spectrum


def frac_n_peaks_query(ssm: spectrum.SpectrumSpectrumMatch) -> float:
    return len(ssm.peak_matches) / len(ssm.query_spectrum.mz)


def frac_n_peaks_library(ssm: spectrum.SpectrumSpectrumMatch):
    return len(ssm.peak_matches) / len(ssm.library_spectrum.mz)


def frac_intensity_query(ssm: spectrum.SpectrumSpectrumMatch):
    return (
        ssm.query_spectrum.intensity[ssm.peak_matches[:, 0]]
        / ssm.query_spectrum.intensity.sum()
    )


def frac_intensity_library(ssm: spectrum.SpectrumSpectrumMatch):
    return (
        ssm.library_spectrum.intensity[ssm.peak_matches[:, 1]]
        / ssm.library_spectrum.intensity.sum()
    )


def hypergeometric_score(ssm: spectrum.SpectrumSpectrumMatch):
    n_peaks = len(ssm.library_spectrum.mz)
    n_matched_peaks = len(ssm.peak_matches)
    n_peak_bins, _, _ = spectrum.get_dim(
        config.min_mz, config.max_mz, config.bin_size
    )
    hypergeometric_score = 0
    for i in range(n_matched_peaks + 1, n_peaks + 1):
        hypergeometric_score += (
                (scipy.special.comb(n_peaks, i)
                 * scipy.special.comb(n_peak_bins - n_peaks, n_peaks - i))
                / scipy.special.comb(n_peak_bins, n_peaks)
        )
    return hypergeometric_score


def bray_curtis_dissimilarity(ssm: spectrum.SpectrumSpectrumMatch):
    # TODO: Do this using intensity instead?
    return (
        1 - (2 * len(ssm.peak_matches))
        / (len(ssm.library_spectrum.mz) + len(ssm.query_spectrum.mz))
    )


def kendalltau(ssm: spectrum.SpectrumSpectrumMatch):
    max_n_peaks = max(len(ssm.query_spectrum.mz), len(ssm.library_spectrum.mz))
    return scipy.stats.kendalltau(
        np.pad(
            ssm.query_spectrum.intensity,
            (0, max_n_peaks - len(ssm.query_spectrum.mz)),
        ),
        np.pad(
            ssm.library_spectrum.intensity,
            (0, max_n_peaks - len(ssm.library_spectrum.mz)),
        ),
    )[0]


def mse(ssm: spectrum.SpectrumSpectrumMatch, axis: str):
    if axis == "mz":
        query_arr = ssm.query_spectrum.mz
        library_arr = ssm.library_spectrum.mz
    elif axis == "intensity":
        query_arr = ssm.query_spectrum.intensity
        library_arr = ssm.library_spectrum.intensity
    else:
        raise ValueError("Unknown axis specified")
    return (
        (query_arr[ssm.peak_matches[:, 0]]
         - library_arr[ssm.peak_matches[:, 1]]) ** 2
        / len(ssm.peak_matches)
    )
