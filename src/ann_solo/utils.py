import itertools
import operator
import os
from typing import Iterator

import mokapot
import numpy as np
import pandas as pd
import pyteomics.auxiliary
import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from ann_solo.spectrum import SpectrumSpectrumMatch


def rescore_matches(ssms: Iterator[SpectrumSpectrumMatch], fdr: float = 0.01,
                    mode: str='std')\
        -> Iterator[SpectrumSpectrumMatch]:
    """
    Rescore SSMs and fetch best targets meeting the given FDR.

    The following formula is used for FDR calculation: #D / #T.

    Parameters
    ----------
        ssms : Iterator[SpectrumSpectrumMatch]
            An iterator of SSMs to be filtered based on FDR.
        fdr : float
            The minimum FDR threshold for filtering.
        mode : str
            The search mode of the cascade.

    Returns
    -------
    Iterator[SpectrumSpectrumMatch]
        An iterator of the SSMs with an FDR below the given FDR threshold. Each
        SSM is assigned its q-value in the `q` attribute.
    """
    column_names = ['ssm_id','query_precursor_mass', 'library_precursor_mass', 'query_precursor_charge',
                   'library_precursor_charge', 'shifted_dot_product', 'hypergeometric_score',
                   'frc_matched_qspec_peaks', 'frc_matched_lspec_peaks',
                   'mse_matched_spec_peak_intensity','mse_matched_spec_peak_m/z',
                   'frc_matched_qspec_peak_intensities', 'frc_matched_lspec_peak_intensities',
                   'bray_curtis_dissimilarity', 'kendalltau',
                   'entropy', 'unweighted_entropy',
                   'euclidean', 'manhattan', 'chebyshev',
                   'squared_euclidean', 'fidelity', 'matusita',
                   'squared_chord', 'bhattacharya_1',
                   'bhattacharya_2', 'harmonic_mean', 'probabilistic_symmetric_chi_squared',
                   'ruzicka', 'roberts', 'intersection',
                   'motyka', 'canberra', 'baroni_urbani_buser',
                   'penrose_size', 'mean_character', 'lorentzian',
                   'penrose_shape', 'clark', 'hellinger',
                   'whittaker_index_of_association', 'symmetric_chi_squared',
                   'pearson_correlation', 'improved_similarity', 'absolute_value',
                   'cosine_dot_product', 'spectral_contrast_angle',
                   'wave_hedges', 'cosine_distance', 'jaccard',
                   'dice', 'inner_product', 'divergence', 'vicis_symmetric_chi_squared_3',
                   'ms_for_id_v1', 'ms_for_id', 'weighted_dot_product', 'label','peptide']
    full_dataset = pd.DataFrame(columns=column_names)
    psms_dictionary = {}
    with tqdm.tqdm(desc='Spectra-spectra match devset generated', total=len(ssms),
                   leave=False, unit='ssm', smoothing=0.1) as pbar:
        for idx,ssm in enumerate(ssms,1):
            ssm_id = 'ssm_' + str(idx)
            psms_dictionary[ssm_id] = ssm
            features = ssm.std_features if mode == 'std' else ssm.open_features
            ssm_meta = {}
            if mode == 'open':
                ssm_meta = {'ssm_id': ssm_id, 'label': False if ssm.is_decoy else True,
                            'peptide': ssm.sequence, 'group': ssm.group}
            else:
                ssm_meta = {'ssm_id': ssm_id, 'label': False if ssm.is_decoy else True,
                            'peptide': ssm.sequence}
            full_dataset = full_dataset.append({**features, **ssm_meta}, ignore_index = True)
            pbar.update(1)
    #Define ML algorithm and hyperparameter search space
    inducer_details = {
        'RandomForestClassifier': {'parameters': {'n_estimators': [120], 'max_depth': [8, 10, 20, 30, 50],
                                                  'criterion':['gini'],'bootstrap':[True]},
                                   'model': RandomForestClassifier()}}
        #Criterion of split can either use the Gini Index for impurity or Entropy for information gain.
        #Computationally, Entropy is more complex (i.e. will increase runtime) since it makes use of logarithms
        #and consequently, we opted for the Gini Index as it is much faster
    parameters = inducer_details['RandomForestClassifier']['parameters']
    ml_algo = inducer_details['RandomForestClassifier']['model']
    estimator = GridSearchCV(ml_algo, param_grid=parameters)
    model = mokapot.Model(estimator)
    #Create LinearPsmDataset from the pandas table
    if mode == 'open':
        psms = mokapot.dataset.LinearPsmDataset(assert_columns_dtypes(full_dataset,mode), target_column='label',
                                                spectrum_columns='ssm_id',
                                                peptide_column='peptide',
                                                group_column='group')
    else:
        psms = mokapot.dataset.LinearPsmDataset(assert_columns_dtypes(full_dataset,mode), target_column='label',
                                                spectrum_columns='ssm_id',
                                                peptide_column='peptide')


    #Train models
    results, models = mokapot.brew(psms, model=model, test_fdr = fdr, folds=10, max_workers=os.cpu_count())

    ##Compute q-value of the re-scored SSMs
	best_targets = None
    if mode == 'open':
        ##Get full list of re-scored PSMs
        rescored_pd = None
        for key, value in results.group_confidence_estimates.items():
            targets = value.confidence_estimates['psms']
            decoys = value.decoy_confidence_estimates['psms']
            rescored_pd = pd.concat([rescored_pd, targets, decoys], ignore_index=True)
        #Sort table by group and score
        rescored_pd.sort_values(['group', 'mokapot score'], ascending=False, inplace=True, ignore_index=True)
        #Get a unique sorted list of group IDs
        sorted_groups_list_rescored = sorted(list(set(rescored_pd['group'])), reverse=True)
        #Compute the q-values and assign them to each group
        full_q_value_list_rescored = []
        for grp in sorted_groups_list_rescored:
            full_q_value_list_rescored.extend(compute_q_value(rescored_pd[rescored_pd['group'] == grp]['label']))
        rescored_pd['q_value'] = full_q_value_list_rescored
        #Get targets table meeting the FDR threshold
        best_targets = rescored_pd[(rescored_pd['q_value'] <= fdr) & (rescored_pd['label'] == True)]
        #targets = results.confidence_estimates['psms']
        #best_targets = targets[targets['mokapot q-value'] <= 0.01]
    else:
        targets = results.confidence_estimates['psms']
        decoys = results.decoy_confidence_estimates['psms']
        rescored_pd = pd.concat([targets, decoys], ignore_index=True)
        ##Compute q-value
        # Sort rows by score
        rescored_pd.sort_values(['mokapot score'], ascending=False, inplace=True, ignore_index=True)
        # Compute the q-values and assign them to each group
        rescored_pd['q_value'] = compute_q_value(rescored_pd['label'])
        # Get targets table meeting the FDR threshold
        best_targets = rescored_pd[(rescored_pd['q_value'] <= fdr) & (rescored_pd['label'] == True)]

    print("Number of matches found: {}".format(best_targets.shape[0]))
    #Retreive final list of PSMs
    for key in best_targets['ssm_id']:
        psms_dictionary[key].search_engine_score = best_targets.loc[best_targets['ssm_id']== key,'mokapot score'].values[0]
        psms_dictionary[key].q = best_targets.loc[best_targets['ssm_id'] == key, 'q_value'].values[0]
        yield psms_dictionary[key]

def compute_q_value(rankedSSMS: list)\
        -> list:
    """
    Compute FDR/q-value of a ranked list of SSMs

    The following formula is used for FDR calculation: #D / #T.

    Parameters
    ----------
        rankedSSMS : list
            A list of floats corresponding to the scores.

    Returns
    -------
    list
        A list of floats corresponding to the q-values.
    """
    running_decoy_count = 0
    running_target_count = 0
    q_values = []
    for count,label in enumerate(rankedSSMS,1):
        running_decoy_count += 0 if label == True else 1
        running_target_count += 1 if label == True else 0
        q_values.append(running_decoy_count/max(1,running_target_count))
    return q_values

def assert_columns_dtypes(df: pd, mode: str='std')\
        -> pd:
    """
    Sets the correct datatype of the columns as they are expected in mokapot.

    Parameters
    ----------
        df : pd
            A pandas dataframe of the PSMs.
        mode : str
            The search mode of the cascade.

    Returns
    -------
    pd
        A pandas dataframe of the PSMs with correct column datatypes.
    """
    non_feature_cols = ['ssm_id','label','peptide']
    if mode == 'open':
        df['group'] = df['group'].astype('str')
        non_feature_cols.append('group')
    df['label'] = df['label'].astype('bool')
    df['peptide'] = df['peptide'].astype('str')
    df['ssm_id'] = df['ssm_id'].astype('str')
    for col in df.columns:
        if col not in non_feature_cols:
            df[col] = pd.to_numeric(df[col])
    return df



def filter_fdr(ssms: Iterator[SpectrumSpectrumMatch], fdr: float = 0.01)\
        -> Iterator[SpectrumSpectrumMatch]:
    """
    Filter SSMs exceeding the given FDR.

    The following formula is used for FDR calculation: #D / #T.

    Parameters
    ----------
        ssms : Iterator[SpectrumSpectrumMatch]
            An iterator of SSMs to be filtered based on FDR.
        fdr : float
            The minimum FDR threshold for filtering.

    Returns
    -------
    Iterator[SpectrumSpectrumMatch]
        An iterator of the SSMs with an FDR below the given FDR threshold. Each
        SSM is assigned its q-value in the `q` attribute.
    """

    for _, _, q, ssm in pyteomics.auxiliary.qvalues(
            ssms, key=operator.attrgetter('search_engine_score'), reverse=True,
            is_decoy=operator.attrgetter('is_decoy'), remove_decoy=False,
            formula=1, correction=0, full_output=True):
        ssm.q = q
        if q <= fdr:
            yield ssm
        else:
            break


def group_rescore(ssms: Iterator[SpectrumSpectrumMatch], fdr: float = 0.01,
                     tol_mass: float = None, tol_mode: str = None,
                     min_group_size: int = None)\
        -> Iterator[SpectrumSpectrumMatch]:
    """
    Filter SSMs exceeding the given FDR.

    Prior to FDR filtering SSMs are grouped based on their precursor mass
    difference and rescored. FDR filtering is applied separately to each common SSM group
    and combined to all uncommon SSM groups.

    Args:
        ssms : Iterator[SpectrumSpectrumMatch]
            An iterator of `SSMs to be filtered based on FDR.
        fdr : float
            The minimum FDR threshold for filtering.
        tol_mass : float, optional
            The mass range to group SSMs. If None no grouping is performed.
        tol_mode : str, optional
            The unit in which the mass range is specified ('Da' or 'ppm'). If
            None no grouping is performed.
        min_group_size : int, optional
            The minimum number of SSMs that should be present in a group for it
            to be considered common. If None no grouping is performed.

    Returns:
    Iterator[SpectrumSpectrumMatch]
        An iterator of the SSMs with an FDR below the given FDR threshold. Each
        SSM is assigned its q-value in the `q` variable.
    """
    ssms_remaining = np.asarray(sorted(
        ssms, key=operator.attrgetter('search_engine_score'), reverse=True))
    exp_masses = np.asarray([ssm.exp_mass_to_charge for ssm in ssms_remaining])
    mass_diffs = np.asarray([(ssm.exp_mass_to_charge - ssm.calc_mass_to_charge)
                             * ssm.charge for ssm in ssms_remaining])

    # Start with the highest ranked SSM.
    groups_common, groups_uncommon = [], []
    while ssms_remaining.size > 0:
        # Find all remaining PSMs within the mass difference window.
        if (tol_mass is None or tol_mode not in ('Da', 'ppm') or
                min_group_size is None):
            mask = np.full(len(ssms_remaining), True, dtype=bool)
        elif tol_mode == 'Da':
            mask = np.fabs(mass_diffs - mass_diffs[0]) <= tol_mass
        elif tol_mode == 'ppm':
            mask = (np.fabs(mass_diffs - mass_diffs[0]) / exp_masses * 10 ** 6
                    <= tol_mass)
        if np.count_nonzero(mask) >= min_group_size:
            groups_common.append(ssms_remaining[mask])
        else:
            groups_uncommon.extend(ssms_remaining[mask])
        # Exclude the selected SSMs from further selections.
        ssms_remaining = ssms_remaining[~mask]
        exp_masses = exp_masses[~mask]
        mass_diffs = mass_diffs[~mask]

    #Set group id
    all_groups = []
    for ssm in groups_uncommon:
        ssm.group = 'Unc'
        all_groups.append(ssm)
    for idx, group in enumerate(groups_common,1):
        for ssm in group:
            ssm.group = 'C'+str(idx)
            all_groups.append(ssm)
    for ssm in rescore_matches(all_groups, fdr, mode='open'):
        yield ssm

def filter_group_fdr(ssms: Iterator[SpectrumSpectrumMatch], fdr: float = 0.01,
                     tol_mass: float = None, tol_mode: str = None,
                     min_group_size: int = None)\
        -> Iterator[SpectrumSpectrumMatch]:
    """
    Filter SSMs exceeding the given FDR.

    Prior to FDR filtering SSMs are grouped based on their precursor mass
    difference. FDR filtering is applied separately to each common SSM group
    and combined to all uncommon SSM groups.

    Args:
        ssms : Iterator[SpectrumSpectrumMatch]
            An iterator of `SSMs to be filtered based on FDR.
        fdr : float
            The minimum FDR threshold for filtering.
        tol_mass : float, optional
            The mass range to group SSMs. If None no grouping is performed.
        tol_mode : str, optional
            The unit in which the mass range is specified ('Da' or 'ppm'). If
            None no grouping is performed.
        min_group_size : int, optional
            The minimum number of SSMs that should be present in a group for it
            to be considered common. If None no grouping is performed.

    Returns:
    Iterator[SpectrumSpectrumMatch]
        An iterator of the SSMs with an FDR below the given FDR threshold. Each
        SSM is assigned its q-value in the `q` variable.
    """
    ssms_remaining = np.asarray(sorted(
        ssms, key=operator.attrgetter('search_engine_score'), reverse=True))
    exp_masses = np.asarray([ssm.exp_mass_to_charge for ssm in ssms_remaining])
    mass_diffs = np.asarray([(ssm.exp_mass_to_charge - ssm.calc_mass_to_charge)
                             * ssm.charge for ssm in ssms_remaining])

    # Start with the highest ranked SSM.
    groups_common, groups_uncommon = [], []
    while ssms_remaining.size > 0:
        # Find all remaining PSMs within the mass difference window.
        if (tol_mass is None or tol_mode not in ('Da', 'ppm') or
                min_group_size is None):
            mask = np.full(len(ssms_remaining), True, dtype=bool)
        elif tol_mode == 'Da':
            mask = np.fabs(mass_diffs - mass_diffs[0]) <= tol_mass
        elif tol_mode == 'ppm':
            mask = (np.fabs(mass_diffs - mass_diffs[0]) / exp_masses * 10 ** 6
                    <= tol_mass)
        if np.count_nonzero(mask) >= min_group_size:
            groups_common.append(ssms_remaining[mask])
        else:
            groups_uncommon.extend(ssms_remaining[mask])
        # Exclude the selected SSMs from further selections.
        ssms_remaining = ssms_remaining[~mask]
        exp_masses = exp_masses[~mask]
        mass_diffs = mass_diffs[~mask]

    # Calculate the FDR combined for all uncommon mass difference groups
    # and separately for each common mass difference group.
    for ssm in itertools.chain(
            filter_fdr(groups_uncommon, fdr),
            *[filter_fdr(group, fdr) for group in groups_common]):
        yield ssm