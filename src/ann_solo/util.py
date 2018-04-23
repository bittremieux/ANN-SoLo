import pandas as pd
import pyteomics.auxiliary


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


def filter_group_fdr(psms, fdr=0.01, tol_mass=None, tol_mode=None,
                     min_group_size=5):
    """
    Filter PSMs exceeding the given FDR.

    PSMs are first grouped based on their precursor mass difference and
    subsequently filtered for each group independently.
    PSM groups are formed by combining all PSMs whose precursor mass difference
    is within the given tolerance of a selected PSM, with the remaining PSMs
    selected by descending score.

    Args:
        psms: A DataFrame of PSMs to be filtered based on FDR. The search
            engine score to rank the PSMS should be listed in the
            `search_engine_score[1]` column and the
            `opt_ms_run[1]_cv_MS:1002217_decoy_peptide` column should be a
            boolean column denoting whether the PSM is a decoy match (True) or
            not (False).
        fdr: The minimum FDR threshold for filtering.
        tol_mass: The mass difference tolerance to combine PSMs. If None no
            grouping is performed.
        tol_mode: The unit in which the mass difference tolerance is specified
            ('Da' or 'ppm'). If None no grouping is performed.
        min_group_size: The minimum number of PSMs that should be present in
            each group. All other PSMs not belonging to a group are grouped
            together.

    Returns:
        A DataFrame of the PSMs with an FDR lower than the given FDR threshold.
        The FDR is available in the `q` column.
    """
    filtered_psms = []
    psms_remaining = psms.sort_values('search_engine_score[1]',
                                      ascending=False)
    psms_remaining['mass_diff'] = (
        (psms_remaining['exp_mass_to_charge']
         - psms_remaining['calc_mass_to_charge']
         ) * psms_remaining['charge']
    )
    # start with the highest ranked PSM
    psms_rest = []
    while len(psms_remaining) > 0:
        # find all remaining PSMs within the precursor mass window
        mass_diff = psms_remaining['mass_diff'].iloc[0]
        if tol_mass is None:
            psms_selected = psms_remaining
        elif tol_mode == 'Da':
            psms_selected =\
                psms_remaining[abs(mass_diff - psms_remaining['mass_diff'])
                               <= tol_mass]
        elif tol_mode == 'ppm':
            psms_selected =\
                psms_remaining[abs(mass_diff - psms_remaining['mass_diff'])
                               / psms_remaining['exp_mass_to_charge'] * 10**6
                               <= tol_mass]
        else:
            psms_selected = psms_remaining
        # exclude the selected PSMs from further selections
        psms_remaining = psms_remaining.drop(psms_selected.index)
        # compute the FDR for the selected PSMs
        if len(psms_selected) > min_group_size:
            filtered_psms.append(filter_fdr(psms_selected, fdr))
        else:
            psms_rest.append(psms_selected)

    # compute the FDR for the rest PSMs
    if len(psms_rest) > 0:
        filtered_psms.append(filter_fdr(pd.concat(psms_rest), fdr))
    
    # combine all filtered PSMs
    filtered_psms = pd.concat(filtered_psms).sort_values('q')
    
    if hasattr(psms, 'df_name'):
        filtered_psms.df_name = psms.df_name
    
    return filtered_psms
