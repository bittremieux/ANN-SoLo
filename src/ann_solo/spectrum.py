import functools
import math
from typing import List, Optional

import mmh3
import numba as nb
import numpy as np
import scipy.special
import scipy.stats as stats
from spectrum_utils.spectrum import MsmsSpectrum
from sklearn.metrics import mean_squared_error

from ann_solo.config import config
from .spectrum_similarity import all_similarity


@nb.njit
def _check_spectrum_valid(spectrum_mz: np.ndarray, min_peaks: int,
                          min_mz_range: float) -> bool:
    """
    Check whether a spectrum is of high enough quality to be used for matching.

    Parameters
    ----------
    spectrum_mz : np.ndarray
        M/z peaks of the sspectrum whose quality is checked.
    min_peaks : int
        The minimum number of peaks for a spectrum to be valid.
    min_mz_range : float
        The minimum mass range (m/z difference between the highest and lowest
        peak) for a spectrum to be valid.

    Returns
    -------
    bool
        True if the spectrum has enough peaks covering a wide enough mass
        range, False otherwise.
    """
    return (len(spectrum_mz) >= min_peaks and
            spectrum_mz[-1] - spectrum_mz[0] >= min_mz_range)


@nb.njit
def _norm_intensity(spectrum_intensity: np.ndarray) -> np.ndarray:
    """
    Normalize spectrum peak intensities.

    Parameters
    ----------
    spectrum_intensity : np.ndarray
        The spectrum peak intensities to be normalized.

    Returns
    -------
    np.ndarray
        The normalized peak intensities.
    """
    return spectrum_intensity / np.linalg.norm(spectrum_intensity)


def process_spectrum(spectrum: MsmsSpectrum, is_library: bool) -> MsmsSpectrum:
    """
    Process the peaks of the MS/MS spectrum according to the config.

    Parameters
    ----------
    spectrum : MsmsSpectrum
        The spectrum that will be processed.
    is_library : bool
        Flag specifying whether the spectrum is a query spectrum or a library
        spectrum.

    Returns
    -------
    MsmsSpectrum
        The processed spectrum. The spectrum is also changed in-place.
    """
    if spectrum.is_processed:
        return spectrum

    min_peaks = config.min_peaks
    min_mz_range = config.min_mz_range

    spectrum = spectrum.set_mz_range(config.min_mz, config.max_mz)
    if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
        spectrum.is_valid = False
        spectrum.is_processed = True
        return spectrum
    if config.resolution is not None:
        spectrum = spectrum.round(config.resolution, 'sum')
        if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
            spectrum.is_valid = False
            spectrum.is_processed = True
            return spectrum

    if config.remove_precursor:
        spectrum = spectrum.remove_precursor_peak(
            config.remove_precursor_tolerance, 'Da', 2)
        if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
            spectrum.is_valid = False
            spectrum.is_processed = True
            return spectrum

    spectrum = spectrum.filter_intensity(
        config.min_intensity, (config.max_peaks_used_library if is_library else
                               config.max_peaks_used))
    if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
        spectrum.is_valid = False
        spectrum.is_processed = True
        return spectrum

    scaling = config.scaling
    if scaling == 'sqrt':
        scaling = 'root'
    if scaling is not None:
        spectrum = spectrum.scale_intensity(
            scaling, max_rank=(config.max_peaks_used_library if is_library else
                               config.max_peaks_used))

    spectrum.intensity = _norm_intensity(spectrum.intensity)

    # Set a flag to indicate that the spectrum has been processed to avoid
    # reprocessing of library spectra for multiple queries.
    spectrum.is_valid = True
    spectrum.is_processed = True

    return spectrum


@functools.lru_cache(maxsize=None)
def get_dim(min_mz, max_mz, bin_size):
    """
    Compute the number of bins over the given mass range for the given bin
    size.

    Args:
        min_mz: The minimum mass in the mass range (inclusive).
        max_mz: The maximum mass in the mass range (inclusive).
        bin_size: The bin size (in Da).

    Returns:
        A tuple containing (i) the number of bins over the given mass range for
        the given bin size, (ii) the highest multiple of bin size lower than
        the minimum mass, (iii) the lowest multiple of the bin size greater
        than the maximum mass. These two final values are the true boundaries
        of the mass range (inclusive min, exclusive max).
    """
    min_mz, max_mz = float(min_mz), float(max_mz)
    start_dim = min_mz - min_mz % bin_size
    end_dim = max_mz + bin_size - max_mz % bin_size
    return round((end_dim - start_dim) / bin_size), start_dim, end_dim


@functools.lru_cache(maxsize=None)
def hash_idx(bin_idx: int, hash_len: int) -> int:
    """
    Hash an integer index to fall between 0 and the given maximum hash index.

    Parameters
    ----------
    bin_idx : int
        The (unbounded) index to be hashed.
    hash_len : int
        The maximum index after hashing.

    Returns
    -------
    int
        The hashed index between 0 and `hash_len`.
    """
    return mmh3.hash(str(bin_idx), 42, signed=False) % hash_len


def spectrum_to_vector(spectrum: MsmsSpectrum, min_mz: float, max_mz: float,
                       bin_size: float, hash_len: int, norm: bool = True,
                       vector: np.ndarray = None) -> np.ndarray:
    """
    Convert a `Spectrum` to a dense NumPy vector.

    Peaks are first discretized in to mass bins of width `bin_size` between
    `min_mz` and `max_mz`, after which they are hashed to random hash bins
    in the final vector.

    Parameters
    ----------
    spectrum : Spectrum
        The `Spectrum` to be converted to a vector.
    min_mz : float
        The minimum m/z to include in the vector.
    max_mz : float
        The maximum m/z to include in the vector.
    bin_size : float
        The bin size in m/z used to divide the m/z range.
    hash_len : int
        The length of the hashed vector, None if no hashing is to be done.
    norm : bool
        Normalize the vector to unit length or not.
    vector : np.ndarray, optional
        A pre-allocated vector.

    Returns
    -------
    np.ndarray
        The hashed spectrum vector with unit length.
    """
    vec_len, min_bound, _ = get_dim(min_mz, max_mz, bin_size)
    if vector is None:
        if hash_len is not None:
            vec_len = hash_len
        vector = np.zeros((vec_len,), np.float32)
        if vec_len != vector.shape[0]:
            raise ValueError('Incorrect vector dimensionality')

    for mz, intensity in zip(spectrum.mz, spectrum.intensity):
        bin_idx = math.floor((mz - min_bound) // bin_size)
        if hash_len is not None:
            bin_idx = hash_idx(bin_idx, hash_len)
        vector[bin_idx] += intensity

    if norm:
        vector /= np.linalg.norm(vector)
    return vector


class SpectrumSpectrumMatch:

    def __init__(
            self,
            query_spectrum: MsmsSpectrum,
            library_spectrum: MsmsSpectrum = None,
            search_engine_score: float = math.nan,
            peak_matches: Optional[List] = None,
            q: float = math.nan,
            group: str = None,
            mode: str = "std",
            num_candidates: int = 0,
    ):
        self.query_spectrum = query_spectrum
        self.library_spectrum = library_spectrum
        self.search_engine_score = search_engine_score
        self.peak_matches = peak_matches
        self.q = q
        self.group = group
        self.mode = mode
        self.std_features = {}
        self.open_features = {}
        self.compute_features()
        self.num_candidates = num_candidates

    def __str__(self):
        return " query_spectrum: " + str(self.query_spectrum) + \
               " library_spectrum: " + str(self.library_spectrum) + \
               " search_engine_score: " + str(self.search_engine_score) + \
               " q: " + str(self.q) + \
               " num_candidates: " + str(self.num_candidates)

    @property
    def sequence(self):
        return (self.library_spectrum.peptide
                if self.library_spectrum is not None else None)

    @property
    def query_identifier(self):
        return self.query_spectrum.identifier

    @property
    def query_index(self):
        return self.query_spectrum.index

    @property
    def library_identifier(self):
        return (self.library_spectrum.identifier
                if self.library_spectrum is not None else None)

    @property
    def retention_time(self):
        return self.query_spectrum.retention_time

    @property
    def charge(self):
        return self.query_spectrum.precursor_charge

    @property
    def exp_mass_to_charge(self):
        return self.query_spectrum.precursor_mz

    @property
    def calc_mass_to_charge(self):
        return (self.library_spectrum.precursor_mz
                if self.library_spectrum is not None else None)

    @property
    def is_decoy(self):
        return (self.library_spectrum.is_decoy
                if self.library_spectrum is not None else None)

    @property
    def n_matched_peaks(self):
        return len(self.peak_matches)

    # FIXME: Similarities shouldn't be defined here.
    def hypergeometric_peak_match_score(self):
        k = self.n_matched_peaks
        m = len(self.library_spectrum.mz)
        N,_,_ = get_dim(config.min_mz,config.max_mz,config.bin_size)
        hgs = 0
        for i in range(k+1,m+1):
            hgs += (scipy.special.comb(m,i,exact=True) * scipy.special.comb(N-m,m-i,exact=True))/scipy.special.comb(N,m,exact=True)
        return hgs

    def fraction_of_matched_query_spectrum_peaks(self):
        return self.n_matched_peaks / len(self.query_spectrum.mz)

    def fraction_of_matched_library_spectrum_peaks(self):
        return self.n_matched_peaks / len(self.library_spectrum.mz)

    def fraction_of_matched_spectrum_peak_intensities(self):
        matched_qs_intensity, matched_ls_intensity = 0, 0
        for qsi, lsi in self.peak_matches:
            matched_qs_intensity += self.query_spectrum.intensity[qsi]
            matched_ls_intensity += self.library_spectrum.intensity[lsi]
        return matched_qs_intensity / sum(self.query_spectrum.intensity) , \
               matched_ls_intensity / sum(self.library_spectrum.intensity)

    def bray_curtis_dissimilarity(self):
        return 1 - (2 * self.n_matched_peaks) / (len(self.library_spectrum.mz) + len(self.query_spectrum.mz))

    def kendalltau(self):
        qs_intesity_len = len(self.query_spectrum.intensity)
        ls_intesity_len = len(self.library_spectrum.intensity)
        max_len = max(qs_intesity_len,ls_intesity_len)
        tau, _ = stats.kendalltau(np.pad(self.query_spectrum.intensity, (0,max_len-qs_intesity_len), 'constant',constant_values=(0, 0)),
                                  np.pad(self.library_spectrum.intensity, (0, max_len-ls_intesity_len), 'constant', constant_values=(0, 0)))
        return tau

    def mse_matched_spec_peak(self,axis):
        qs_list = []
        ls_list = []
        for qsi,lsi in self.peak_matches:
            if axis == 'intensity':
                qs_list.append(self.query_spectrum.intensity[qsi])
                ls_list.append(self.library_spectrum.intensity[lsi])
            else:
                qs_list.append(self.query_spectrum.mz[qsi])
                ls_list.append(self.library_spectrum.mz[lsi])
        return 1 if len(self.peak_matches) == 0 else mean_squared_error(qs_list, ls_list)

    def get_spec_2d_representation(self,input_spec):
        spectrum = {}
        for index, mz in enumerate(input_spec.mz):
            if index:
                peak = np.array([[mz, input_spec.mz[index]]], dtype=np.float32)
                spectrum = np.concatenate((spectrum, peak), axis=0)
            else:
                spectrum = np.array([[mz, input_spec.mz[index]]], dtype=np.float32)
        return spectrum

    def compute_all_other_distance_measures(self,ms2_ppm: float = None, ms2_da: float = None):
        if ms2_ppm is None:
            return all_similarity(self.get_spec_2d_representation(self.query_spectrum), self.get_spec_2d_representation(self.library_spectrum), ms2_da=ms2_da)
        else:
            return all_similarity(self.get_spec_2d_representation(self.query_spectrum), self.get_spec_2d_representation(self.library_spectrum), ms2_ppm=ms2_ppm)

    def compute_features(self):
        frc_matched_qspec_peak_intsty,frc_matched_lspec_peak_intsty = \
            self.fraction_of_matched_spectrum_peak_intensities()
        common_features = {'query_precursor_mass':self.get_precursor_mass(self.query_spectrum),
                           'library_precursor_mass':self.get_precursor_mass(self.library_spectrum),
                           'query_precursor_charge':self.get_precursor_charge(self.query_spectrum),
                           'library_precursor_charge':self.get_precursor_charge(self.library_spectrum),
                           'shifted_dot_product':self.search_engine_score,
                           'hypergeometric_score':self.hypergeometric_peak_match_score(),
                           'frc_matched_qspec_peaks':self.fraction_of_matched_query_spectrum_peaks(),
                           'frc_matched_lspec_peaks':self.fraction_of_matched_library_spectrum_peaks(),
                           'mse_matched_spec_peak_intensity':self.mse_matched_spec_peak("intensity"),
                           'mse_matched_spec_peak_m/z':self.mse_matched_spec_peak("M/Z"),
                           'frc_matched_qspec_peak_intensities':frc_matched_qspec_peak_intsty,
                           'frc_matched_lspec_peak_intensities':frc_matched_lspec_peak_intsty,
                           'bray_curtis_dissimilarity':self.bray_curtis_dissimilarity(),
                           'kendalltau':self.kendalltau()}

        if config.precursor_tolerance_mass is not None and config.precursor_tolerance_mode is not None and self.mode=='std':
            distance_measures = self.compute_all_other_distance_measures(
                ms2_ppm=config.precursor_tolerance_mass if config.precursor_tolerance_mode == 'ppm' else None,
                ms2_da=config.precursor_tolerance_mass if config.precursor_tolerance_mode == 'Da' else None)
            distance_measures["cosine_dot_product"] = distance_measures.pop("dot_product")
            distance_measures["cosine_distance"] = distance_measures.pop("cosine")

            self.std_features = {**common_features, **distance_measures}

        elif config.precursor_tolerance_mass_open is not None and config.precursor_tolerance_mode_open is not None and self.mode=='open':
            distance_measures = self.compute_all_other_distance_measures(
                ms2_ppm=config.precursor_tolerance_mass_open if config.precursor_tolerance_mode_open == 'ppm' else None,
                ms2_da=config.precursor_tolerance_mass_open if config.precursor_tolerance_mode_open == 'Da' else None)
            distance_measures["cosine_dot_product"] = distance_measures.pop("dot_product")
            distance_measures["cosine_distance"] = distance_measures.pop("cosine")

            self.open_features = {**common_features, **distance_measures}
