import operator
from typing import Dict, Iterator, List, Union

import mokapot
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from spectrum_utils.utils import mass_diff

from ann_solo import spectrum_similarity as sim
from ann_solo.spectrum import SpectrumSpectrumMatch


def score_ssms(
    ssms: List[SpectrumSpectrumMatch],
    fdr: float,
    grouped: bool = False,
    tol_mass: float = None,
    tol_mode: str = None,
    min_group_size: int = None,
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
    tol_mass : float
        The mass tolerance to group SSMs. Must be specified if `grouped` is
        True.
    tol_mode : str
        The unit in which the mass tolerance is specified ("Da" or "ppm").
        Must be specified if `grouped` is True.
    min_group_size : int
        The minimum group size. SSMs in smaller groups are combined into a
        residual group. Must be specified if `grouped` is True.

    Returns
    -------
    Iterator[SpectrumSpectrumMatch]
        An iterator of the SSMs with assigned scores and q-values.
    """
    # Create a Dataset with SSM features.
    features = pd.DataFrame(_compute_ssm_features(ssms))
    if grouped:
        features["group"] = _get_ssm_groups(
            ssms, tol_mass, tol_mode, min_group_size
        )
    else:
        features["group"] = 0
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
    ssms: List[SpectrumSpectrumMatch],
    tol_mass: float,
    tol_mode: str,
    min_group_size: int,
) -> np.ndarray:
    """
    Get SSM group labels based on the precursor mass differences.

    Parameters
    ----------
    ssms : Iterator[SpectrumSpectrumMatch]
        SSMs to be grouped.
    tol_mass : float
        The mass tolerance to group SSMs.
    tol_mode : str
        The unit in which the mass tolerance is specified ("Da" or "ppm").
    min_group_size : int
        The minimum group size. SSMs in smaller groups are combined into a
        residual group.

    Returns
    -------
    np.ndarray
        An array with group labels for all SSMs.
    """
    groups = -np.ones(len(ssms), np.int32)
    ssms_remaining = np.asarray(
        sorted(
            ssms, key=operator.attrgetter("search_engine_score"), reverse=True
        )
    )
    exp_masses = np.asarray([ssm.exp_mass_to_charge for ssm in ssms_remaining])
    mass_diffs = np.asarray(
        [
            (ssm.exp_mass_to_charge - ssm.calc_mass_to_charge) * ssm.charge
            for ssm in ssms_remaining
        ]
    )
    # Start with the highest ranked SSM.
    group = 0
    while ssms_remaining.size > 0:
        # Find all remaining SSMs within the mass difference window.
        if (
            tol_mass is None
            or tol_mode not in ("Da", "ppm")
            or min_group_size is None
        ):
            mask = np.full(len(ssms_remaining), True, dtype=bool)
        elif tol_mode == "Da":
            mask = np.fabs(mass_diffs - mass_diffs[0]) <= tol_mass
        elif tol_mode == "ppm":
            mask = (
                np.fabs(mass_diffs - mass_diffs[0]) / exp_masses * 10**6
                <= tol_mass
            )
        if np.count_nonzero(mask) >= min_group_size:
            for ssm in ssms_remaining[mask]:
                ssm.group = group
            group += 1
        else:
            for ssm in ssms_remaining[mask]:
                ssm.group = 0
        # Exclude the selected SSMs from further selections.
        ssms_remaining = ssms_remaining[~mask]
        exp_masses = exp_masses[~mask]
        mass_diffs = mass_diffs[~mask]
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
