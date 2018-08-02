import math

import numpy as np

from ann_solo.config import config


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
            return self

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
        self._is_processed = True
        
        return self

    def get_vector(self, min_mz=None, max_mz=None, bin_size=None):
        """
        Convert the Spectrum to a vector by binning the peaks.

        Args:
            min_mz: The minimum mass.
            max_mz: The maximum mass.
            bin_size: The size (in Da) of the mass bins.

        Returns:
            None if the Spectrum doesn't have any peaks; otherwise a vector
            with each peak of the Spectrum assigned to its mass bin. For
            multiple peaks assigned to the same mass bin the intensities are
            summed. The final vector is normalized to have unit length.
        """
        if min_mz is None:
            min_mz = config.min_mz
        if max_mz is None:
            max_mz = config.max_mz
        if bin_size is None:
            bin_size = config.bin_size

        if self.is_valid():
            vec_length, min_bound, max_bound = get_dim(
                    min_mz, max_mz, bin_size)
            peaks = np.zeros((vec_length,), dtype=np.float32)
            # add each mass and intensity to their low-dimensionality bin
            for mass, intensity in zip(self.masses, self.intensities):
                mass_bin = math.floor((mass - min_bound) // bin_size)
                peaks[mass_bin] += intensity

            # normalize
            return peaks / np.linalg.norm(peaks)
        else:
            return None


class SpectrumMatch:

    def __init__(self, query_spectrum,
                 library_spectrum=None, search_engine_score=0.0, q=math.nan):
        # query information
        self.query_id = query_spectrum.identifier
        self.retention_time = query_spectrum.retention_time
        self.charge = query_spectrum.precursor_charge
        self.exp_mass_to_charge = query_spectrum.precursor_mz

        # identification information
        if library_spectrum is not None:
            self.library_id = library_spectrum.identifier
            self.sequence = library_spectrum.peptide
            self.calc_mass_to_charge = library_spectrum.precursor_mz
            self.is_decoy = library_spectrum.is_decoy
        else:
            self.library_id = self.sequence = self.calc_mass_to_charge = self.is_decoy = None

        self.search_engine_score = search_engine_score
        self.q = q

        # performance information
        self.num_candidates = self.time_candidates = self.time_match = self.time_total = 0
