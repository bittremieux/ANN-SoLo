from typing import Dict, Iterator, List, Union

import mokapot
import numpy as np
import pandas as pd
import scipy.signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from spectrum_utils.utils import mass_diff

from ann_solo import spectrum_similarity as sim
from ann_solo.spectrum import SpectrumSpectrumMatch


def score_ssms(
    ssms: List[SpectrumSpectrumMatch],
    fdr: float,
    grouped: bool = False,
    min_group_size: int = 100,
) -> List[SpectrumSpectrumMatch]:
    """
    Score SSMs using semi-supervised learning with mokapot.

    Parameters
    ----------
    ssms : List[SpectrumSpectrumMatch]
        SSMs to be scored.
    fdr : float
        The minimum FDR threshold to accept target SSMs.
    grouped : bool
        Compute q-values per SSM group or not (default: False).
    min_group_size : int
        The minimum group size (default: 100). SSMs in smaller groups are
        combined into a residual group. Must be specified if `grouped` is True.

    Returns
    -------
    Iterator[SpectrumSpectrumMatch]
        An iterator of the SSMs with assigned scores and q-values.
    """
    # Create a Dataset with SSM features.
    features = pd.DataFrame(_compute_ssm_features(ssms))
    features["group"] = _get_ssm_groups(ssms, min_group_size) if grouped else 0
    dataset = mokapot.dataset.LinearPsmDataset(
        features,
        target_column="is_target",
        spectrum_columns="index",
        peptide_column="sequence",
        group_column="group",
    )
    # Define the mokapot model.
    #   - We use a random forest.
    #   - Features are preprocessed by TODO (remove correlated and zero
    #                                        variance features).
    #   - We perform minimal tuning of TODO hyperparameters.
    hyperparameters = {"max_depth": [None]}
    model = mokapot.Model(
        GridSearchCV(RandomForestClassifier(), param_grid=hyperparameters),
        train_fdr=fdr,
    )
    # Train the mokapot model and combine the SSMs for all groups.
    confidences, _ = mokapot.brew(dataset, model, fdr, max_workers=-1)
    ssm_scores = pd.concat(
        [
            confidences.group_confidence_estimates[group].psms
            for group in confidences.groups
        ],
        ignore_index=True,
    )
    # Assign the scores to the SSM objects.
    for i, score, q in zip(
        ssm_scores["index"],
        ssm_scores["mokapot score"],
        ssm_scores["mokapot q-value"],
    ):
        ssms[i].search_engine_score = score
        ssms[i].q = q
    return ssms


def _get_ssm_groups(
    ssms: List[SpectrumSpectrumMatch], min_group_size: int
) -> pd.Series:
    """
    Get SSM group labels based on the precursor mass differences.

    SSMs are grouped by finding peaks in histograms centered around each
    nominal mass difference and assigning SSMs to the nearest peak.

    Parameters
    ----------
    ssms : Iterator[SpectrumSpectrumMatch]
        SSMs to be grouped.
    min_group_size : int
        The minimum group size. SSMs in smaller groups are combined into a
        residual group.

    Returns
    -------
    pd.Series
        A Series with group labels for all SSMs.
    """
    # Group SSMs in a window around each nominal mass difference.
    mass_diffs = np.asarray(
        [
            (ssm.exp_mass_to_charge - ssm.calc_mass_to_charge) * ssm.charge
            for ssm in ssms
        ]
    )
    order = np.argsort(mass_diffs)
    groups, group = -np.ones(len(ssms), np.int32), 0
    group_md, group_i = np.nan, []
    for counter, (md, i) in enumerate(zip(mass_diffs[order], order)):
        if round(md) != group_md or counter == len(mass_diffs) - 1:
            if round(md) == group_md:
                group_i.append(i)
            if len(group_i) > 0:
                # Assign groups within the current interval.
                # Create a mass difference histogram and find peaks.
                bins = np.linspace(group_md - 0.5, group_md + 0.5, 101)
                hist, _ = np.histogram(mass_diffs[group_i], bins=bins)
                peaks_bin_i, prominences = scipy.signal.find_peaks(
                    hist, prominence=(None, None)
                )
                if len(peaks_bin_i) > 0:
                    # Assign mass differences to their closest peak.
                    for md_j, j in zip(mass_diffs[group_i], group_i):
                        peak_assignment = -1, np.inf
                        for peak_i, peak in enumerate(bins[peaks_bin_i]):
                            distance_to_peak = abs(peak - md_j)
                            if (
                                bins[prominences["left_bases"][peak_i]]
                                < md_j
                                < bins[prominences["right_bases"][peak_i]]
                                and distance_to_peak < peak_assignment[1]
                            ):
                                peak_assignment = peak_i, distance_to_peak
                        if peak_assignment[0] != -1:
                            groups[j] = group + peak_assignment[0]
                group += len(peaks_bin_i)
            # Start a new interval.
            group_i = []
        group_i.append(i)
        group_md = round(md)
    groups = pd.Series(groups)
    # Reassign small groups to a residual group (accurate FDR calculation is
    # not possible in overly small groups).
    group_counts = groups.value_counts()
    groups[groups.isin(group_counts.index[group_counts < min_group_size])] = -1
    return groups


def _compute_ssm_features(
    ssms: Iterator[SpectrumSpectrumMatch],
) -> Dict[str, Union[bool, float, int, str]]:
    """
    Compute spectrum-spectrum match features.

    Parameters
    ----------
    ssms : Iterator[SpectrumSpectrumMatch]
        The SSMs whose features are computed.

    Returns
    -------
    Dict[str, Union[bool, float, int, str]]
        A dictionary with features for all SSMs. The dictionary must contain
        "index", "sequence", "is_target", and "group" keys with the
        corresponding SSM metadata. Additional keys contain SSM features.
    """
    features = {
        "index": [],
        "sequence": [],
        "query_precursor_mz": [],
        "query_precursor_charge": [],
        "library_precursor_mz": [],
        "library_precursor_charge": [],
        "mz_diff_ppm": [],
        "dot": [],
        "frac_n_peaks_query": [],
        "frac_n_peaks_library": [],
        "frac_intensity_query": [],
        "frac_intensity_library": [],
        "mean_squared_error_mz": [],
        "mean_squared_error_intensity": [],
        "hypergeometric_score": [],
        "kendalltau": [],
        "ms_for_id_v1": [],
        "ms_for_id_v2": [],
        "entropy_unweighted": [],
        "entropy_weighted": [],
        "manhattan": [],
        "euclidean": [],
        "chebyshev": [],
        "pearson_correlation": [],
        "bray_curtis_distance": [],
        "is_target": [],
        "group": [],
    }
    for i, ssm in enumerate(ssms):
        # Skip low-quality spectrum matches.
        if len(ssm.peak_matches) <= 1:
            continue
        features["index"].append(i)
        features["sequence"].append(ssm.sequence)
        features["query_prec_mz"].append(ssm.query_spectrum.precursor_mz)
        features["query_prec_ch"].append(ssm.query_spectrum.precursor_charge)
        features["lib_prec_mz"].append(ssm.library_spectrum.precursor_mz)
        features["lib_prec_ch"].append(ssm.library_spectrum.precursor_charge)
        features["mz_diff_ppm"].append(
            mass_diff(
                ssm.query_spectrum.precursor_mz,
                ssm.library_spectrum.precursor_mz,
                False,
            )
        )
        # TODO: Explicitly compute the cosine similarity.
        features["cosine"].append(ssm.search_engine_score)
        features["frac_n_peaks_query"].append(sim.frac_n_peaks_query(ssm))
        features["frac_n_peaks_lib"].append(sim.frac_n_peaks_library(ssm))
        features["frac_int_query"].append(sim.frac_intensity_query(ssm))
        features["frac_int_lib"].append(sim.frac_intensity_library(ssm))
        features["mse_mz"].append(sim.mean_squared_error(ssm, "mz"))
        features["mse_int"].append(sim.mean_squared_error(ssm, "intensity"))
        features["hypergeometric_score"].append(sim.hypergeometric_score(ssm))
        features["kendalltau"].append(sim.kendalltau(ssm))
        features["ms_for_id_v1"].append(sim.ms_for_id_v1(ssm))
        features["ms_for_id_v2"].append(sim.ms_for_id_v2(ssm))
        features["entropy_unweighted"].append(sim.entropy(ssm, False))
        features["entropy_weighted"].append(sim.entropy(ssm, True))
        features["manhattan"].append(sim.manhattan(ssm))
        features["euclidean"].append(sim.euclidean(ssm))
        features["chebyshev"].append(sim.chebyshev(ssm))
        features["pearson_correlation"].append(sim.pearson_correlation(ssm))
        features["bray_curtis_distance"].append(sim.bray_curtis_distance(ssm))
        features["is_target"].append(not ssm.is_decoy)
        features["group"].append(ssm.group)
    return features
