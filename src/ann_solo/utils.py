import os
from typing import Iterator

import mokapot
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from ann_solo import spectrum_similarity
from ann_solo.spectrum import SpectrumSpectrumMatch


def rescore_matches(
    ssms: Iterator[SpectrumSpectrumMatch], fdr: float, mode: str
) -> Iterator[SpectrumSpectrumMatch]:
    """
    Rescore SSMs using semi-supervised learning with Mokapot.

    Parameters
    ----------
        ssms : Iterator[SpectrumSpectrumMatch]
            SSMs to be rescored.

    Returns
    -------
    Iterator[SpectrumSpectrumMatch]
        FIXME
        An iterator of the SSMs with an FDR below the given FDR threshold. Each
        SSM is assigned its q-value in the `q` attribute.
    """
    # Compute all SSM features.
    features = {
        "ssm_id": [],
        "sequence": [],
        "query_precursor_mz": [],
        "query_precursor_charge": [],
        "library_precursor_mz": [],
        "library_precursor_charge": [],
        "shifted_dot": [],
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
        "is_decoy": [],
    }
    for ssm in ssms:
        if len(ssm.peak_matches) <= 1:
            continue
        features["ssm_id"].append(ssm.query_identifier)
        features["sequence"].append(ssm.sequence)
        features["query_precursor_mz"].append(ssm.query_spectrum.precursor_mz)
        features["query_precursor_charge"].append(
            ssm.query_spectrum.precursor_charge
        )
        features["library_precursor_mz"].append(
            ssm.library_spectrum.precursor_mz
        )
        features["library_precursor_charge"].append(
            ssm.library_spectrum.precursor_charge
        )
        features["shifted_dot"].append(ssm.search_engine_score)
        features["frac_n_peaks_query"].append(
            spectrum_similarity.frac_n_peaks_query(ssm)
        )
        features["frac_n_peaks_library"].append(
            spectrum_similarity.frac_n_peaks_library(ssm)
        )
        features["frac_intensity_query"].append(
            spectrum_similarity.frac_intensity_query(ssm)
        )
        features["frac_intensity_library"].append(
            spectrum_similarity.frac_intensity_library(ssm)
        )
        features["mean_squared_error_mz"].append(
            spectrum_similarity.mean_squared_error(ssm, "mz")
        )
        features["mean_squared_error_intensity"].append(
            spectrum_similarity.mean_squared_error(ssm, "intensity")
        )
        features["hypergeometric_score"].append(
            spectrum_similarity.hypergeometric_score(ssm)
        )
        features["kendalltau"].append(spectrum_similarity.kendalltau(ssm))
        features["ms_for_id_v1"].append(spectrum_similarity.ms_for_id_v1(ssm))
        features["ms_for_id_v2"].append(spectrum_similarity.ms_for_id_v2(ssm))
        features["entropy_unweighted"].append(
            spectrum_similarity.entropy(ssm, False)
        )
        features["entropy_weighted"].append(
            spectrum_similarity.entropy(ssm, True)
        )
        features["manhattan"].append(spectrum_similarity.manhattan(ssm))
        features["euclidean"].append(spectrum_similarity.euclidean(ssm))
        features["chebyshev"].append(spectrum_similarity.chebyshev(ssm))
        features["pearson_correlation"].append(
            spectrum_similarity.pearson_correlation(ssm)
        )
        features["bray_curtis_distance"].append(
            spectrum_similarity.bray_curtis_distance(ssm)
        )
        features["is_decoy"].append(ssm.is_decoy)
    pd.DataFrame(features).to_csv("features.csv", index=False)
    features = mokapot.dataset.LinearPsmDataset(
        pd.DataFrame(features),
        target_column="is_decoy",
        spectrum_columns="ssm_id",
        peptide_column="sequence",
        group_column="group" if mode == "open" else None,
    )
    # Define the Mokapot model.
    hyperparameters = {"max_depth": [8, 10, 20, 30, 50]}
    # FIXME: Add feature preprocessing steps (remove correlated and zero
    #   variance features).
    model = mokapot.Model(
        GridSearchCV(RandomForestClassifier(), param_grid=hyperparameters)
    )
    # Train the Mokapot model and rescore the SSMs.
    results, models = mokapot.brew(
        features, model, fdr, folds=10, max_workers=os.cpu_count()
    )
    targets = results.confidence_estimates["psms"]
    decoys = results.decoy_confidence_estimates["psms"]
    ssms_rescored = pd.concat([targets, decoys], ignore_index=True)
    return ssms_rescored
