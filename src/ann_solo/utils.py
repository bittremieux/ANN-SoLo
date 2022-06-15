from typing import Dict, Iterator, List, Union

import mokapot
import numpy as np
import pandas as pd
import scipy.signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from spectrum_utils.utils import mass_diff

from ann_solo import spectrum_similarity as sim
from ann_solo.spectrum import SpectrumSpectrumMatch


class CorrelationThreshold:

    def __init__(self, threshold=None):
        self.threshold = threshold if threshold is not None else 1.0

    def fit(self, X, y=None):
        corr = np.abs(np.corrcoef(X, rowvar=False))
        self.mask = ~(np.tril(corr, k=-1) > self.threshold).any(axis=1)
        return self

    def transform(self, X, y=None):
        return X[:, self.mask]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)

    def get_support(self, indices=False):
        return self.mask if not indices else np.where(self.mask)[0]


def score_ssms(
    ssms: List[SpectrumSpectrumMatch],
    fdr: float,
    model: str,
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
    model : str
        The type of machine learning model to use. Can be "rf" for a random
        forest classifier, "svm" for a Percolator-like linear SVM, or `None`
        to disable semi-supervised learning.
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
    #   - Choice between a random forest, linear SVM, or no semi-supervised
    #     learning.
    #   - Features are preprocessed by standardizing them, removing
    #     zero-variance features, and removing highly correlated features.
    #   - We perform minimal tuning of TODO hyperparameters.
    if model is None:
        # Calculate q-values based on the cosine similarity.
        confidences = dataset.assign_confidence(features["cosine"], True)
    else:
        scaler = make_pipeline(
            StandardScaler(),
            VarianceThreshold(),
            CorrelationThreshold(0.95),
        )
        if model == "svm":
            clf = mokapot.model.PercolatorModel(scaler, train_fdr=fdr)
        elif model == "rf":
            clf = mokapot.Model(
                GridSearchCV(
                    RandomForestClassifier(n_jobs=-1, random_state=1),
                    param_grid={"max_depth": [None]},
                ),
                scaler,
                train_fdr=fdr,
            )
        else:
            raise ValueError(
                "Unknown semi-supervised machine learning model given"
            )
        # Train the mokapot model and combine the SSMs for all groups.
        confidences, _ = mokapot.brew(dataset, clf, fdr)
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
        "query_prec_mz": [],
        "query_prec_ch": [],
        "lib_prec_mz": [],
        "lib_prec_ch": [],
        "mz_diff_ppm": [],
        "cosine": [],
        "frac_n_peaks_query": [],
        "frac_n_peaks_lib": [],
        "frac_int_query": [],
        "frac_int_lib": [],
        "mse_mz": [],
        "mse_int": [],
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
