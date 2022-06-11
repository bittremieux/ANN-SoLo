import scipy.special
import scipy.stats

from ann_solo import spectrum
from ann_solo.config import config


def frac_n_peaks_query(ssm: spectrum.SpectrumSpectrumMatch) -> float:
    """
    Get the number of shared peaks as a fraction of the number of peaks in the
    query spectrum.

    Parameters
    ----------
    ssm : spectrum.SpectrumSpectrumMatch
        The match between a query spectrum and a library spectrum.

    Returns
    -------
    float
        The fraction of shared peaks.
    """
    return len(ssm.peak_matches) / len(ssm.query_spectrum.mz)


def frac_n_peaks_library(ssm: spectrum.SpectrumSpectrumMatch) -> float:
    """
    Get the number of shared peaks as a fraction of the number of peaks in the
    library spectrum.

    Parameters
    ----------
    ssm : spectrum.SpectrumSpectrumMatch
        The match between a query spectrum and a library spectrum.

    Returns
    -------
    float
        The fraction of shared peaks.
    """
    return len(ssm.peak_matches) / len(ssm.library_spectrum.mz)


def frac_intensity_query(ssm: spectrum.SpectrumSpectrumMatch) -> float:
    """
    Get the fraction of explained intensity in the query spectrum.

    Parameters
    ----------
    ssm : spectrum.SpectrumSpectrumMatch
        The match between a query spectrum and a library spectrum.

    Returns
    -------
    float
        The fraction of explained intensity.
    """
    return (
        ssm.query_spectrum.intensity[ssm.peak_matches[:, 0]].sum()
        / ssm.query_spectrum.intensity.sum()
    )


def frac_intensity_library(ssm: spectrum.SpectrumSpectrumMatch) -> float:
    """
    Get the fraction of explained intensity in the library spectrum.

    Parameters
    ----------
    ssm : spectrum.SpectrumSpectrumMatch
        The match between a query spectrum and a library spectrum.

    Returns
    -------
    float
        The fraction of explained intensity.
    """
    return (
        ssm.library_spectrum.intensity[ssm.peak_matches[:, 1]].sum()
        / ssm.library_spectrum.intensity.sum()
    )


def mean_squared_error(
    ssm: spectrum.SpectrumSpectrumMatch, axis: str
) -> float:
    """
    Get the mean squared error (MSE) of peak matches between two spectra.

    Parameters
    ----------
    ssm : spectrum.SpectrumSpectrumMatch
        The match between a query spectrum and a library spectrum.
    axis : str
        Calculate the MSE between the m/z values ("mz") or intensity values
        ("intensity") of the matched peaks.

    Returns
    -------
    float
        The MSE between the m/z or intensity values of the matched peaks.

    Raises
    ------
    ValueError
        If the specified axis is not "mz" or "intensity".
    """
    if axis == "mz":
        query_arr = ssm.query_spectrum.mz
        library_arr = ssm.library_spectrum.mz
    elif axis == "intensity":
        query_arr = ssm.query_spectrum.intensity
        library_arr = ssm.library_spectrum.intensity
    else:
        raise ValueError("Unknown axis specified")
    return (
        (
            query_arr[ssm.peak_matches[:, 0]]
            - library_arr[ssm.peak_matches[:, 1]]
        )
        ** 2
    ).sum() / len(ssm.peak_matches)


def hypergeometric_score(ssm: spectrum.SpectrumSpectrumMatch) -> float:
    """
    Get the hypergeometric score of peak matches between two spectra.

    The hypergeometric score measures the probability of obtaining more than
    the observed number of peak matches by random chance, which follows a
    hypergeometric distribution.

    For the original description, see:
    Dasari, S. et al. Pepitome: Evaluating improved spectral library search for
    identification complementarity and quality assessment. Journal of Proteome
    Research 11, 1686–1695 (2012).

    Parameters
    ----------
    ssm : spectrum.SpectrumSpectrumMatch
        The match between a query spectrum and a library spectrum.

    Returns
    -------
    float
        The hypergeometric score of peak matches.
    """
    n_library_peaks = len(ssm.library_spectrum.mz)
    n_matched_peaks = len(ssm.peak_matches)
    n_peak_bins, _, _ = spectrum.get_dim(
        config.min_mz, config.max_mz, config.bin_size
    )
    return sum(
        [
            (
                scipy.special.comb(n_library_peaks, i)
                * scipy.special.comb(
                    n_peak_bins - n_library_peaks, n_library_peaks - i
                )
            )
            / scipy.special.comb(n_peak_bins, n_library_peaks)
            for i in range(n_matched_peaks + 1, n_library_peaks)
        ]
    )


def kendalltau(ssm: spectrum.SpectrumSpectrumMatch) -> float:
    """
    Get the Kendall-Tau score of peak matches between two spectra.

    The Kendall-Tau score measures the correspondence between the intensity
    ranks of the set of peaks matched between spectra.

    For the original description, see:
    Dasari, S. et al. Pepitome: Evaluating improved spectral library search for
    identification complementarity and quality assessment. Journal of Proteome
    Research 11, 1686–1695 (2012).

    Parameters
    ----------
    ssm : spectrum.SpectrumSpectrumMatch
        The match between a query spectrum and a library spectrum.

    Returns
    -------
    float
        The hypergeometric score of peak matches.
    """
    return scipy.stats.kendalltau(
        ssm.query_spectrum.intensity[ssm.peak_matches[:, 0]],
        ssm.library_spectrum.intensity[ssm.peak_matches[:, 1]],
    )[0]


def bray_curtis_dissimilarity(ssm: spectrum.SpectrumSpectrumMatch) -> float:
    # TODO: Do this using intensity instead?
    return 1 - (2 * len(ssm.peak_matches)) / (
        len(ssm.library_spectrum.mz) + len(ssm.query_spectrum.mz)
    )
