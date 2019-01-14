import functools
import math

import mmh3
import numpy as np

from ann_solo.config import config


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


class Spectrum:
    """
    Tandem mass spectrum.

    A tandem mass spectrum with associated peaks and optionally an
    identification (for library spectra).
    """

    def __init__(self, identifier, precursor_mz, precursor_charge,
                 retention_time=None, peptide=None, is_decoy=False):
        """
        Create a new Spectrum with the specified information.

        Args:
            identifier: The unique identifier of the Spectrum (within the query
                file or the spectral library file).
            precursor_mz: The spectrum's precursor mass.
            precursor_charge: The spectrum's precursor charge.
            retention_time: The spectrum's retention time.
            peptide: The peptide sequence (if known).
            is_decoy: True if the Spectrum is a decoy spectrum, False if not
                (target or query).
        """
        # unique identifier
        self.identifier = identifier

        # precursor and peptide identification
        self.precursor_mz = precursor_mz
        self.precursor_charge = precursor_charge
        self.retention_time = retention_time
        self.peptide = peptide

        # decoy or not?
        self.is_decoy = is_decoy

        # peak values
        self.masses = None
        self.intensities = None
        self.annotations = None
        self._is_processed = False

    def is_valid(self):
        """
        Verify if this is a valid and high-quality Spectrum.

        Returns:
            True if the Spectrum's peaks have been processed and are of high
            quality, False if not.
        """
        return self._is_processed

    def set_peaks(self, masses, intensities, annotations=None):
        """
        Assigns peaks to the spectrum. Note: no quality checking or processing
        of the spectrum is performed.

        Args:
            masses: The masses at which the peaks are detected.
            intensities: The intensities of the peaks at their corresponding
                masses.
            annotations: (Optionally) the annotations of the peaks at their
                corresponding masses.
        """
        self.masses = masses.astype(np.float32)
        self.intensities = intensities.astype(np.float32)
        self.annotations = (annotations if annotations is not None
                            else np.empty(len(masses), object))

    def process_peaks(self, resolution=None, min_mz=None, max_mz=None,
                      remove_precursor=None, remove_precursor_tolerance=None,
                      min_peaks=None, max_peaks_used=None,
                      min_intensity=None, min_mz_range=None, scaling=None):
        """
        Check that the spectrum is of sufficient quality and process the peaks.

        Args:
            resolution: Spectral library resolution; masses will be rounded to
                the given number of decimals.
            min_mz: Minimum m/z value (inclusive). Peaks at lower m/z values
                will be discarded.
            max_mz: Maximum m/z value (inclusive). Peaks at higher m/z values
                will be discarded.
            remove_precursor: If True, remove peaks around the precursor mass.
            remove_precursor_tolerance: If remove_precursor, this specifies the
                window (in Da) around the precursor mass to remove peaks.
            min_peaks: Discard spectra with less peaks.
            max_peaks_used: Only retain this many of the most intense peaks.
            min_intensity: Remove peaks with a lower intensity relative to the
                maximum intensity.
            min_mz_range: Discard spectra with a smaller mass range.
            scaling: Manner in which to scale the intensities ('sqrt' for
                square root scaling, 'rank' for rank scaling).
        """
        if resolution is None:
            resolution = config.resolution
        if min_mz is None:
            min_mz = config.min_mz
        if max_mz is None:
            max_mz = config.max_mz
        if remove_precursor is None:
            remove_precursor = config.remove_precursor
        if remove_precursor_tolerance is None:
            remove_precursor_tolerance = config.remove_precursor_tolerance
        if min_peaks is None:
            min_peaks = config.min_peaks
        if max_peaks_used is None:
            max_peaks_used = config.max_peaks_used
        if min_intensity is None:
            min_intensity = config.min_intensity
        if min_mz_range is None:
            min_mz_range = config.min_mz_range
        if scaling is None:
            scaling = config.scaling

        if self._is_processed:
            return self

        low_quality = False

        masses = self.masses
        intensities = self.intensities
        annotations = self.annotations

        # round masses based on the spectral library resolution
        if resolution is not None:
            # peaks get sorted here
            masses, indices, inverse = np.unique(
                np.around(masses, resolution), True, True)
            if len(masses) != len(intensities):
                # some peaks got merged, so sum their intensities
                intensities_merged = intensities[indices]
                merged_indices = np.setdiff1d(
                    np.arange(len(intensities)), indices, True)
                intensities_merged[inverse[merged_indices]] +=\
                    intensities[merged_indices]

                intensities = intensities_merged
                # TODO: select the most likely annotation
                annotations = annotations[indices]
        else:
            # make sure the peaks are always sorted by mass
            order = np.argsort(masses)
            masses = masses[order]
            intensities = intensities[order]
            annotations = annotations[order]

        # restrict to range [min_mz ; max_mz]
        filter_range = np.where(
            np.logical_and(min_mz <= masses, masses <= max_mz))[0]

        # remove peak(s) close to the precursor mass
        filter_peaks = filter_range
        if (remove_precursor and self.precursor_mz is not None and
                self.precursor_charge is not None):
            pep_mass = (self.precursor_mz * self.precursor_charge
                        if self.precursor_charge is not None
                        else self.precursor_mz)
            max_charge = (self.precursor_charge + 1
                          if self.precursor_charge is not None
                          else 2)  # exclusive
            filter_precursor = []
            for charge in range(1, max_charge):
                for isotope in range(3):
                    filter_precursor.append(np.where(np.logical_and(
                        (pep_mass + isotope) / charge
                        - remove_precursor_tolerance <= masses,
                        masses <= (pep_mass + isotope) / charge
                        + remove_precursor_tolerance))[0])
            filter_peaks = np.setdiff1d(
                filter_peaks, np.concatenate(filter_precursor), True)

        # check if sufficient peaks remain
        if len(filter_peaks) < min_peaks:
            # return self
            low_quality = True
        if len(filter_peaks) == 0:
            return self

        # apply mass range filter
        filtered_masses = masses[filter_peaks]
        filtered_intensities = intensities[filter_peaks]
        filtered_annotations = annotations[filter_peaks]

        # only use the specified number of most intense peaks
        filter_number = np.argsort(filtered_intensities)[::-1][:max_peaks_used]
        # discard low-intensity noise peaks
        max_intensity = filtered_intensities[filter_number][0]
        filter_noise = np.where(
            filtered_intensities >= min_intensity * max_intensity)[0]

        # apply intensity filters
        filter_intensity = np.intersect1d(filter_number, filter_noise, True)
        filtered_masses = filtered_masses[filter_intensity]
        filtered_intensities = filtered_intensities[filter_intensity]
        filtered_annotations = filtered_annotations[filter_intensity]

        # check if the peaks cover a sufficient m/z range
        if filtered_masses[-1] - filtered_masses[0] < min_mz_range:
            # return self
            low_quality = True

        # scale the intensities by their root
        # to reduce the effect of extremely high intensity peaks
        if scaling == 'sqrt':
            scaled_intensities = np.sqrt(filtered_intensities)
        elif scaling == 'rank':
            scaled_intensities = max_peaks_used - np.argsort(np.argsort(filtered_intensities)[::-1])
        else:
            scaled_intensities = filtered_intensities
        # normalize the intensities to get a unit vector
        norm_intensities = (
            scaled_intensities / np.linalg.norm(scaled_intensities)).astype(
                np.float32)

        self.masses = filtered_masses
        self.intensities = norm_intensities
        self.annotations = filtered_annotations
        self._is_processed = True and not low_quality

        return self


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


def spectrum_to_vector(spectrum: Spectrum, min_mz: float, max_mz: float,
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
        The length of the hashed vector.
    norm : bool
        Normalize the vector to unit length or not.
    vector : np.ndarray, optional
        A pre-allocated vector.

    Returns
    -------
    np.ndarray
        The hashed spectrum vector with unit length.
    """
    if vector is None:
        vector = np.zeros((hash_len,), np.float32)
    if hash_len != vector.shape[0]:
        raise ValueError('Incorrect vector dimensionality')

    _, min_bound, max_bound = get_dim(min_mz, max_mz, bin_size)
    for mz, intensity in zip(spectrum.masses, spectrum.intensities):
        bin_idx = hash_idx(math.floor((mz - min_bound) // bin_size), hash_len)
        vector[bin_idx] += intensity

    if norm:
        vector /= np.linalg.norm(vector)
    return vector


class SpectrumSpectrumMatch:

    def __init__(self, query_spectrum: Spectrum,
                 library_spectrum: Spectrum = None,
                 search_engine_score: float = math.nan,
                 q: float = math.nan,
                 num_candidates: int = 0):
        self.query_spectrum = query_spectrum
        self.library_spectrum = library_spectrum
        self.search_engine_score = search_engine_score
        self.q = q
        self.num_candidates = num_candidates

    @property
    def sequence(self):
        if self.library_spectrum is not None:
            return self.library_spectrum.peptide
        else:
            return None

    @property
    def identifier(self):
        return self.query_spectrum.identifier

    @property
    def accession(self):
        if self.library_spectrum is not None:
            return self.library_spectrum.identifier
        else:
            return None

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
        if self.library_spectrum is not None:
            return self.library_spectrum.precursor_mz
        else:
            return None

    @property
    def is_decoy(self):
        if self.library_spectrum is not None:
            return self.library_spectrum.is_decoy
        else:
            return None
