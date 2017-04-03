# distutils: language = c++
# distutils: sources = dot.cpp

cimport cython
import numpy as np
cimport numpy as np
from libcpp.algorithm cimport sort
from libcpp.pair cimport pair
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector

from config import config



def get_best_match(query, candidates, fragment_mz_tolerance=None, allow_shift=None):
    """
    Find the best matching candidate spectrum compared to the given query spectrum.

    Spectrum similarity is evaluated based on the dot product. Either a standard element-wise dot product (while taking
    the fragment mass tolerance into account), or an element-wise dot product with allowed peak shifts based on the
    precursor mass difference between the spectra is calculated. This shifted dot product can result in a more
    appropriate score when PTM(s) cause some peaks in the spectrum to be shifted to another mass value. When peak shifts
    are allowed, annotated peaks from the candidate spectra (i.e. known library spectra) will be taken into account,
    whereas no annotations are taken into account for the peaks from the query spectrum.

    Args:
        query: The query spectrum for which the most similar candidate spectrum is determined.
        candidates: All candidate spectra that are compared to the query spectrum.
        fragment_mz_tolerance: Mass tolerance indicating the window around the mass peaks used for matching two peaks.
        allow_shift: Allow peaks to be shifted according to the precursor mass difference or not.

    Returns:
        The candidate with the highest similarity compared to the query spectrum, and their similarity score. None if no
        matching candidate was found.
    """
    # query spectrum
    cdef np.ndarray[np.float32_t] query_masses = query.masses
    cdef np.ndarray[np.float32_t] query_intensities = query.intensities
    cdef vector[SpectrumPeak] query_unshifted_peaks, query_shifted_peaks
    cdef double query_precursor_mz = query.precursor_mz
    # candidate spectrum
    cdef np.ndarray[np.float32_t] candidate_masses, candidate_intensities
    cdef np.ndarray[np.uint8_t] candidate_annotation_charges
    cdef double candidate_precursor_mz
    cdef unsigned int candidate_precursor_charge
    cdef double mass_dif, score
    cdef double max_score = 0
    # internal variables
    cdef unsigned int nr_candidates = len(candidates)
    cdef unsigned int index
    cdef int max_candidate_index = -1

    if fragment_mz_tolerance is None:
        fragment_mz_tolerance = config.fragment_mz_tolerance
    if allow_shift is None:
        allow_shift = config.allow_peak_shifts

    # generate a peak list for the query spectrum
    query_unshifted_peaks = get_unshifted_peak_list(query_masses, query_intensities)

    # find the best matching candidate spectrum
    for index in range(nr_candidates):
        # convert the candidate spectrum to Cython-style objects
        candidate = candidates[index]
        candidate_precursor_mz = candidate.precursor_mz
        candidate_precursor_charge = candidate.precursor_charge
        candidate_masses = candidate.masses
        candidate_intensities = candidate.intensities
        candidate_annotation_charges = np.zeros(len(candidate.annotations), dtype=np.uint8)
        for i, annotation in enumerate(candidate.annotations):
            if annotation is not None:
                candidate_annotation_charges[i] = annotation[1]

        # add possible shifted peaks for the query spectrum
        if allow_shift and abs(query_precursor_mz - candidate_precursor_mz) > fragment_mz_tolerance:
            mass_dif = (query_precursor_mz - candidate_precursor_mz) * candidate_precursor_charge
            query_shifted_peaks = get_shifted_peak_list(query_masses, query_intensities, candidate_precursor_charge, mass_dif)

        # compute the matching score
        match = dot(candidate_masses, candidate_intensities, candidate_annotation_charges,
                    query_unshifted_peaks, query_shifted_peaks, fragment_mz_tolerance)

        if match.first > max_score:
            max_score = match.first
            peak_matches = match.second
            max_candidate_index = index

    if max_candidate_index != -1:
        return candidates[max_candidate_index], max_score, peak_matches
    else:
        return None, max_score, []


"""
Represent a peak in a spectrum as a 4-tuple of its mass, its intensity, the charge of the peak (0 if unknown),
and the peak index it was derived from (if shifted peaks are allowed).
"""
cdef struct SpectrumPeak:
    double mass
    double intensity
    unsigned int charge
    unsigned int index


cdef bint compare_spectrum_peak(SpectrumPeak a, SpectrumPeak b):
    """
    Comparator to sort `SpectrumPeak`s on their mass.

    Args:
        a: The first `SpectrumPeak` to be compared.
        b: The second `SpectrumPeak` to be compared.

    Returns:
        `1` if the mass of `a` is smaller than the mass of `b`, else `0`.
    """
    if a.mass < b.mass:
        return 1
    else:
        return 0


"""
Represent a peak match between two spectra by the product of the intensities of both matched peaks, and their
indices in the first and second spectrum respectively.
"""
cdef struct PeakMatch:
    double peak_multiplication
    unsigned int spectrum1_index
    unsigned int spectrum2_index


cdef bint compare_peak_match(PeakMatch a, PeakMatch b):
    """
    Comparator to sort `PeakMatch`s on their product of the intensities.

    `PeakMatch`s are sorted in reversed order (i.e. the PeakMatch with the highest product of the intensities will be
    the first element).

    Args:
        a: The first `PeakMatch` to be compared.
        b: The second `PeakMatch` to be compared.

    Returns:
        `1` if the product of the intensities of `a` is larger than the product of the intensities of `b`, else `0`.
    """
    if a.peak_multiplication > b.peak_multiplication:
        return 1
    else:
        return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef vector[SpectrumPeak] get_unshifted_peak_list(np.ndarray[np.float32_t] masses, np.ndarray[np.float32_t] intensities):
    """
    Convert a spectrum to a vector of `SpectrumPeak` structs.

    Args:
        masses: The masses of the spectrum.
        intensities: The intensities of the spectrum associated with each mass.

    Returns:
         A vector of `SpectrumPeak` structs.
    """
    # internal variables
    cdef unsigned int spectrum_n = masses.shape[0]
    cdef unsigned int index
    cdef SpectrumPeak peak
    # return value
    cdef vector[SpectrumPeak] spectrum_peaks

    # converted all unshifted peaks to a vector of SpectrumPeak structs
    for index in range(spectrum_n):
        peak = SpectrumPeak(mass=masses[index], intensity=intensities[index], charge=0, index=index)
        spectrum_peaks.push_back(peak)

    return spectrum_peaks


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef vector[SpectrumPeak] get_shifted_peak_list(np.ndarray[np.float32_t] masses, np.ndarray[np.float32_t] intensities,
                           unsigned int precursor_charge, double mass_dif):
    """
    Generate a vector of all possible shifted peaks of a spectrum based on the specified mass difference at all
    possible charges up to the specified charge.

    Peaks will be shifted by a mass corresponding to the given mass difference divided by all charges up to the given
    charge.

    Args:
        masses: The masses of the spectrum.
        intensities: The intensities of the spectrum associated with each mass.
        precursor_charge: The maximum charge each peak can correspond to.
        mass_dif: The mass difference used to shift peaks.

    Returns:
         A vector of `SpectrumPeak` structs of all possible shifted peaks.
    """
    # internal variables
    cdef unsigned int spectrum_n = masses.shape[0]
    cdef unsigned int index, charge
    cdef double mass
    cdef SpectrumPeak peak
    # return value
    cdef vector[SpectrumPeak] spectrum_peaks

    # enumerate all possible shifted peaks based on the mass difference at a specific charge
    for index in range(spectrum_n):
        mass = masses[index]
        for charge in range(1, precursor_charge + 1):
            peak = SpectrumPeak(mass=mass - mass_dif / charge, intensity=intensities[index], charge=charge, index=index)
            spectrum_peaks.push_back(peak)

    return spectrum_peaks


@cython.boundscheck(False)
@cython.wraparound(False)
cdef pair[double, vector[pair[uint, uint]]] dot(np.ndarray[np.float32_t] masses1,
                                                np.ndarray[np.float32_t] intensities1,
                                                np.ndarray[np.uint8_t] annotation_charges1,
                                                vector[SpectrumPeak] spectrum2_unshifted_peaks,
                                                vector[SpectrumPeak] spectrum2_shifted_peaks,
                                                double fragment_mz_tolerance):
    """
    Calculate the dot product between two spectra.

    Either a standard element-wise dot product (while taking the fragment mass tolerance into account), or an element-
    wise dot product with allowed peak shifts based on the precursor mass difference between both spectra is
    calculated. This shifted dot product can result in a better score when PTM(s) cause some peaks in the spectrum to be
    shifted to another mass value. When peak shifts are allowed, annotated peaks from `spectrum1` will be taken into
    account, whereas no annotations are taken into account for the peaks from `spectrum2`.

    Args:
        masses1: The masses of the first (known) spectrum.
        intensities1: The intensities of the first (known) spectrum associated with each mass.
        annotation_charges1: The annotations of the first (known) spectrum associated with each peak.
        spectrum2_unshifted_peaks: A vector of `SpectrumPeak` structs of all possible unshifted peaks for the second
                                   (unknown) spectrum.
        spectrum2_shifted_peaks: A vector of `SpectrumPeak` structs of all possible shifted peaks for the second
                                 (unknown) spectrum. Empty if no shifted peaks are allowed.
        fragment_mz_tolerance: Mass tolerance indicating the window around the mass peaks used for matching two peaks.

    Returns:
        A 2-tuple of the dot product between the first spectrum and the second spectrum and a list of the peaks matched
        to compute this dot product.
    """
    # internal variables
    cdef vector[SpectrumPeak] spectrum2_peaks
    cdef unsigned int spectrum1_n = masses1.shape[0]
    cdef unsigned int index, spectrum1_index, spectrum2_index, spectrum1_annotation_charge
    cdef double spectrum1_mass, spectrum1_intensity
    cdef bint can_match
    cdef PeakMatch match
    cdef vector[PeakMatch] peak_matches
    cdef pair[unsigned int, unsigned int] best_matches
    cdef unordered_set[unsigned int] spectrum1_used_peaks, spectrum2_used_peaks
    # return value
    cdef pair[double, vector[pair[uint, uint]]] dot_product

    # if necessary, merge the unshifted and shifted peaks for spectrum2 and sort the peaks based on the mass
    if spectrum2_shifted_peaks.size() > 0:
        spectrum2_shifted_peaks.insert(spectrum2_shifted_peaks.end(),
                                       spectrum2_unshifted_peaks.begin(), spectrum2_unshifted_peaks.end())
        sort(spectrum2_shifted_peaks.begin(), spectrum2_shifted_peaks.end(), compare_spectrum_peak)
        spectrum2_peaks = spectrum2_shifted_peaks
    else:
        spectrum2_peaks = spectrum2_unshifted_peaks

    # find the matching peaks between the two spectra
    spectrum2_index = 0
    for spectrum1_index in range(spectrum1_n):
        spectrum1_mass = masses1[spectrum1_index]
        spectrum1_intensity = intensities1[spectrum1_index]
        spectrum1_annotation_charge = annotation_charges1[spectrum1_index]
        # advance while the mass is too small to match
        while spectrum2_index < spectrum2_peaks.size() - 1 and\
              spectrum1_mass - fragment_mz_tolerance > spectrum2_peaks[spectrum2_index].mass:
            spectrum2_index += 1

        # match the peaks within the fragment mass window if possible
        index = 0
        while spectrum2_index + index < spectrum2_peaks.size() and\
              abs(spectrum1_mass - spectrum2_peaks[spectrum2_index + index].mass) <= fragment_mz_tolerance:
            can_match = 0
            if spectrum2_peaks[spectrum2_index + index].charge == 0:
                # unshifted peak
                can_match = 1
            elif spectrum1_annotation_charge == spectrum2_peaks[spectrum2_index + index].charge != 0:
                # annotated peak shifted according to the peak's charge
                can_match = 1
            elif spectrum1_annotation_charge == 0:
                # non-annotated peak shifted according to some charge
                can_match = 1

            if can_match:
                match = PeakMatch(peak_multiplication=spectrum1_intensity * spectrum2_peaks[spectrum2_index + index].intensity,
                                  spectrum1_index=spectrum1_index,
                                  spectrum2_index=spectrum2_peaks[spectrum2_index + index].index)
                peak_matches.push_back(match)

            index += 1

    # use the most prominent matches to compute the score
    sort(peak_matches.begin(), peak_matches.end(), compare_peak_match)
    for match in peak_matches:
        if spectrum1_used_peaks.count(match.spectrum1_index) == spectrum2_used_peaks.count(match.spectrum2_index) == 0:
            dot_product.first += match.peak_multiplication

            # save the matched peaks
            best_matches.first = match.spectrum1_index
            best_matches.second = match.spectrum2_index
            dot_product.second.push_back(best_matches)

            # make sure these peaks are not used anymore
            spectrum1_used_peaks.insert(match.spectrum1_index)
            spectrum2_used_peaks.insert(match.spectrum2_index)

    return dot_product
