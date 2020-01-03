import collections
from typing import List, Tuple

import numba as nb
import numpy as np
import spectrum_utils.spectrum as sus


SpectrumTuple = collections.namedtuple(
    'SpectrumTuple', ['precursor_mz', 'precursor_charge', 'mz', 'intensity',
                      'charge'])


def get_best_match(query: sus.MsmsSpectrum, candidates: List[sus.MsmsSpectrum],
                   fragment_mz_tolerance: float, allow_shift: bool) \
        -> Tuple[sus.MsmsSpectrum, float, List[Tuple[int, int]]]:
    candidates_arr = []
    for candidate in candidates:
        if not hasattr(candidate, 'charge'):
            candidate.charge = np.asarray(
                [annotation.charge if annotation is not None else 0
                 for annotation in candidate.annotation], np.uint8)
        candidates_arr.append(SpectrumTuple(
            candidate.precursor_mz, candidate.precursor_charge, candidate.mz,
            candidate.intensity, candidate.charge))
    best_candidate_index, best_score, best_peak_matches =\
        _get_best_match_nb(SpectrumTuple(
                query.precursor_mz, query.precursor_charge, query.mz,
                query.intensity, np.zeros_like(query.mz, dtype=np.uint8)),
            candidates_arr, fragment_mz_tolerance, allow_shift)
    return candidates[best_candidate_index], best_score, best_peak_matches


@nb.njit
def _get_best_match_nb(query: SpectrumTuple, candidates: List[SpectrumTuple],
                       fragment_mz_tolerance: float, allow_shift: bool) \
        -> Tuple[int, float, List[Tuple[int, int]]]:
    best_candidate_index, best_score, best_peak_matches = None, 0, None
    # Compute a (shifted) dot product score between the query spectrum and each
    # candidate spectrum.
    for candidate_index, candidate in enumerate(candidates):
        # Candidate peak indices depend on whether we allow shifts
        # (check all shifted peaks as well) or not.
        precursor_mass_diff = ((query.precursor_mz - candidate.precursor_mz)
                               * candidate.precursor_charge)
        # Only take peak shifts into account if the mass difference is
        # relevant.
        num_shifts = 1
        if allow_shift and abs(precursor_mass_diff) >= fragment_mz_tolerance:
            num_shifts += candidate.precursor_charge
        candidate_peak_index = np.zeros(num_shifts, np.uint8)
        mass_diff = np.zeros(num_shifts, np.float32)
        for charge in range(1, num_shifts):
            mass_diff[charge] = precursor_mass_diff / charge

        # Find the matching peaks between the query spectrum and the candidate
        # spectrum.
        peak_match_scores, peak_match_idx = [], []
        for query_peak_index, (query_peak_mz, query_peak_intensity) in \
                enumerate(zip(query.mz, query.intensity)):
            # Advance while there is an excessive mass difference.
            for cpi in range(num_shifts):
                while (candidate_peak_index[cpi] < len(candidate.mz) - 1 and
                       (query_peak_mz - fragment_mz_tolerance >
                        candidate.mz[candidate_peak_index[cpi]]
                        + mass_diff[cpi])):
                    candidate_peak_index[cpi] += 1
            # Match the peaks within the fragment mass window if possible.
            for cpi in range(num_shifts):
                index = 0
                candidate_peak_i = candidate_peak_index[cpi] + index
                while (candidate_peak_i < len(candidate.mz) and
                       abs(query_peak_mz - (candidate.mz[candidate_peak_i]
                           + mass_diff[cpi])) <= fragment_mz_tolerance):
                    # Slightly penalize matching peaks without an annotation.
                    match_multiplier = 0.
                    # Unshifted peaks are matched directly.
                    if cpi == 0:
                        match_multiplier = 1.
                    # Shifted peaks with a known charge (and therefore
                    # annotation) should be shifted with a mass difference
                    # according to this charge.
                    elif candidate.charge[candidate_peak_i] == cpi:
                        match_multiplier = 1.
                    # Shifted peaks without a known charge can be shifted with
                    # a mass difference according to any charge up to the
                    # precursor charge.
                    elif candidate.charge[candidate_peak_i] == 0:
                        match_multiplier = 2. / 3.

                    if match_multiplier > 0.:
                        peak_match_scores.append(
                            match_multiplier * query_peak_intensity
                            * candidate.intensity[candidate_peak_i])
                        peak_match_idx.append((query_peak_index,
                                               candidate_peak_i))

                    index += 1
                    candidate_peak_i = candidate_peak_index[cpi] + index

        candidate_score, candidate_peak_matches = 0., []
        if len(peak_match_scores) > 0:
            # Use the most prominent peak matches to compute the score (sort in
            # descending order).
            peak_match_scores_arr = np.asarray(peak_match_scores)
            peak_match_order = np.argsort(peak_match_scores_arr)[::-1]
            peak_match_scores_arr = peak_match_scores_arr[peak_match_order]
            peak_match_idx_arr = np.asarray(peak_match_idx)[peak_match_order]
            query_peaks_used, candidate_peaks_used = set(), set()
            for peak_match_score, query_peak_i, candidate_peak_i in zip(
                    peak_match_scores_arr, peak_match_idx_arr[:, 0],
                    peak_match_idx_arr[:, 1]):
                if (query_peak_i not in query_peaks_used
                        and candidate_peak_i not in candidate_peaks_used):
                    candidate_score += peak_match_score
                    # Save the matched peaks.
                    candidate_peak_matches.append((query_peak_i,
                                                   candidate_peak_i))
                    # Make sure these peaks are not used anymore.
                    query_peaks_used.add(query_peak_i)
                    candidate_peaks_used.add(candidate_peak_i)

        if best_candidate_index is None or best_score < candidate_score:
            best_candidate_index = candidate_index
            best_score = candidate_score
            best_peak_matches = candidate_peak_matches

    return best_candidate_index, best_score, best_peak_matches
