import collections

import numpy as np
import pandas as pd
import pyteomics.auxiliary
import sklearn.cluster


def filter_fdr(psms, fdr=0.01):
    """
    Filter PSMs exceeding the given FDR.

    Args:
        psms: A DataFrame of PSMs to be filtered based on FDR. The search
            engine score to rank the PSMS should be listed in the
            `search_engine_score[1]` column and the
            `opt_ms_run[1]_cv_MS:1002217_decoy_peptide` column should be a
            boolean column denoting whether the PSM is a decoy match (True) or
            not (False).
        fdr: The minimum FDR threshold for filtering.
        
    Returns:
        A DataFrame of the PSMs with an FDR lower than the given FDR threshold.
        The FDR is available in the `q` column.
    """
    psms_filtered = pyteomics.auxiliary.filter(
            psms, fdr=fdr, key='search_engine_score[1]', reverse=True,
            is_decoy='opt_ms_run[1]_cv_MS:1002217_decoy_peptide')
    
    if hasattr(psms, 'df_name'):
        psms_filtered.df_name = psms.df_name
    
    return psms_filtered


def group_psms_massdif(psms, bandwidth=None, min_group_size=5):
    """
    Group PSMs based on their precursor mass difference.

    Args:
        psms: A DataFrame of PSMs to be grouped PSMs based on their precursor
            mass difference.
        bandwidth: The bandwidth for the Gaussian kernel to cluster the
            precursor mass differences. If None it is estimated from the PSMs
            without a precursor mass difference.
        min_group_size: The minimum number of PSMs that should be present in
            each group after clustering on precursor mass difference. All PSMs
            not belonging to a group based on a specific precursor mass
            difference are grouped together.
        
    Returns:
        A dictionary with as keys (arbitrary) group IDs and as values the
        indices in the given `psms` DataFrame of the PSMs belonging to each
        group.
    """
    # calculate mass differences
    mass_dif = (
        (psms.exp_mass_to_charge - psms.calc_mass_to_charge) * psms.charge
    )
    
    # estimate the bandwidth based on PSMs without a mass difference
    border = 0.75
    if bandwidth is None:
        bandwidth = sklearn.cluster.estimate_bandwidth(
                mass_dif[
                    (mass_dif > -border) & (mass_dif < border)
                ].values.reshape(-1, 1),
                n_jobs=-1)
    
    # find mass difference clusters
    mean_shift = sklearn.cluster.MeanShift(bandwidth,
                                           min_bin_freq=min_group_size,
                                           cluster_all=False, n_jobs=-1)
    clusters = mean_shift.fit_predict(
        mass_dif.values.reshape(-1, 1))
    # discard clusters with too few elements
    cluster_assignments = collections.defaultdict(set)
    cluster_labels, cluster_sizes = np.unique(clusters, return_counts=True)
    for label, size in zip(cluster_labels, cluster_sizes):
        cluster_assignments[-1 if size < min_group_size else label].update(
                np.where(clusters == label)[0].tolist())
    
    return cluster_assignments


def filter_group_fdr(psms, fdr=0.01, bandwidth=None, min_group_size=5):
    """
    Filter PSMs exceeding the given FDR.
    
    PSMs are first grouped based on their precursor mass difference and
    subsequently filtered for each group independently.

    Args:
        psms: A DataFrame of PSMs to be filtered based on FDR. The search
            engine score to rank the PSMS should be listed in the
            `search_engine_score[1]` column and the
            `opt_ms_run[1]_cv_MS:1002217_decoy_peptide` column should be a
            boolean column denoting whether the PSM is a decoy match (True) or
            not (False).
        fdr: The minimum FDR threshold for filtering.
        bandwidth: The bandwidth for the Gaussian kernel to cluster the
            precursor mass differences. If None it is estimated from the PSMs
            without a precursor mass difference.
        min_group_size: The minimum number of PSMs that should be present in
            each group after clustering on precursor mass difference. All PSMs
            not belonging to a group based on a specific precursor mass
            difference are grouped together.
        
    Returns:
        A DataFrame of the PSMs with an FDR lower than the given FDR threshold.
        The FDR is available in the `q` column.
    """
    psm_groups = group_psms_massdif(psms, bandwidth, min_group_size)
    
    # compute the q-value per individual cluster and
    # sort the filtered PSMs by q-value
    filtered_psms = pd.concat([filter_fdr(psms.iloc[list(psm_ids)], fdr=fdr)
                               for psm_ids in psm_groups.values()])
    filtered_psms.sort_values('q', inplace=True)
    
    if hasattr(psms, 'df_name'):
        filtered_psms.df_name = psms.df_name
    
    return filtered_psms
