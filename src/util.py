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
    clusters = mean_shift.fit_predict(mass_dif.values.reshape(-1, 1))
    # discard clusters with too few elements
    labels, sizes = np.unique(clusters, return_counts=True)
    small_cluster_labels = labels[np.where(sizes < min_group_size)[0]]
    clusters[np.isin(clusters, small_cluster_labels)] = -1
    # compute the q-value per individual cluster and
    # sort the filtered PSMs by q-value
    filtered_psms = psms.groupby(clusters).apply(filter_fdr, (fdr,))
    filtered_psms.sort_values('q', inplace=True)
    
    if hasattr(psms, 'df_name'):
        filtered_psms.df_name = psms.df_name
    
    return filtered_psms
