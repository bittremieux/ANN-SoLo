# distutils: language = c++
# distutils: sources = SpectrumMatch.cpp

import numpy as np
cimport numpy as np
from libcpp.utility cimport pair
from libcpp.vector cimport vector

from config import config


cdef extern from 'SpectrumMatch.h' namespace 'ann_solo':
    cdef cppclass Spectrum:
        Spectrum(double, unsigned int, vector[float], vector[float], vector[np.uint8_t]) except +

    cdef cppclass SpectrumSpectrumMatch:
        SpectrumSpectrumMatch(unsigned int) except +
        unsigned int getCandidateIndex()
        double getScore()
        vector[pair[uint, uint]]* getPeakMatches()

    cdef cppclass SpectrumMatcher:
        SpectrumMatcher() except +
        SpectrumSpectrumMatch* dot(Spectrum*, vector[Spectrum*], double, boolean)


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
        The candidate with the highest similarity compared to the query spectrum, their similarity score, and a list of
        tuples `(query_peak_id, candidate_peak_id)` of the matching peaks between the query spectrum and the optimal
        candidate spectrum.
    """
    if fragment_mz_tolerance is None:
        fragment_mz_tolerance = config.fragment_mz_tolerance
    if allow_shift is None:
        allow_shift = config.allow_peak_shifts

    cdef vector[Spectrum*] candidates_vec
    try:
        # convert the candidates
        for candidate in candidates:
            candidate_charges = np.zeros(len(candidate.annotations), dtype=np.uint8)
            for index, annotation in enumerate(candidate.annotations):
                if annotation is not None:
                    candidate_charges[index] = annotation[1]
            candidates_vec.push_back(new Spectrum(candidate.precursor_mz, candidate.precursor_charge,
                                                  candidate.masses, candidate.intensities, candidate_charges))

        query_spec = new Spectrum(query.precursor_mz, query.precursor_charge, query.masses, query.intensities, query.annotations)

        query_matcher = new SpectrumMatcher()
        result = query_matcher.dot(query_spec, candidates_vec, fragment_mz_tolerance, allow_shift)

        return candidates[result.getCandidateIndex()], result.getScore(), result.getPeakMatches()[0]
    finally:
        for i in range(candidates_vec.size()):
            del candidates_vec[i]
        candidates_vec.clear()
        del query_spec
        del query_matcher
        del result
