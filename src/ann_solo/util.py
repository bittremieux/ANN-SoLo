import itertools

import numpy as np
import pyteomics.auxiliary


def filter_fdr(psms, fdr=0.01):
    """
    Filter PSMs exceeding the given FDR.

    The following formula is used for FDR calculation: #D / #T.

    Args:
        psms: An iterable of `SpectrumMatch` PSMs to be filtered based on FDR.
        fdr: The minimum FDR threshold for filtering.

    Returns:
        A generator of the `SpectrumMatch` PSMs with an FDR below the given FDR
        threshold. Each PSM is assigned its q-value in the `q` variable.
    """
    for _, _, q, psm in pyteomics.auxiliary.qvalues(
            psms, key=lambda x: x.search_engine_score, reverse=True,
            is_decoy=lambda x: x.is_decoy, remove_decoy=True,
            formula=1, correction=0, full_output=True):
        psm.q = q
        if q <= fdr:
            yield psm
        else:
            break


def filter_group_fdr(psms, fdr=0.01, tol_mass=0., tol_mode=None,
                     min_group_size=5):
    """
    Filter PSMs exceeding the given FDR.

    Prior to FDR filtering PSMs are grouped based on their precursor mass
    difference. FDR filtering is applied separately to each common PSM group
    and combined to all uncommon PSM groups.

    Args:
        psms: An iterable of `SpectrumMatch` PSMs to be filtered based on FDR.
        fdr: The minimum FDR threshold for filtering.
        tol_mass: The mass range to group PSMs. If None no grouping is
            performed.
        tol_mode: The unit in which the mass range is specified ('Da' or
            'ppm'). If None no grouping is performed.
        min_group_size: The minimum number of PSMs that should be present in
            a group for it to be considered common.

    Returns:
        A generator of the `SpectrumMatch` PSMs with an FDR below the given FDR
        threshold. Each PSM is assigned its q-value in the `q` variable.
    """
    psms_remaining = np.asarray(sorted(
            psms, key=lambda psm: psm.search_engine_score, reverse=True))
    exp_masses = np.asarray([psm.exp_mass_to_charge for psm in psms_remaining])
    mass_diffs = np.asarray([(psm.exp_mass_to_charge - psm.calc_mass_to_charge)
                             * psm.charge for psm in psms_remaining])

    # start with the highest ranked PSM
    groups_common, groups_uncommon = [], []
    while len(psms_remaining) > 0:
        # find all remaining PSMs within the mass difference window
        if tol_mass is None or tol_mode not in ('Da', 'ppm'):
            mask = np.full(len(psms_remaining), True, dtype=bool)
        elif tol_mode == 'Da':
            mask = np.fabs(mass_diffs - mass_diffs[0]) <= tol_mass
        elif tol_mode == 'ppm':
            mask = (np.fabs(mass_diffs - mass_diffs[0]) / exp_masses * 10 ** 6 <= tol_mass)
        if np.count_nonzero(mask) >= min_group_size:
            groups_common.append(psms_remaining[mask])
        else:
            groups_uncommon.extend(psms_remaining[mask])
        # exclude the selected PSMs from further selections
        psms_remaining = psms_remaining[~mask]
        exp_masses = exp_masses[~mask]
        mass_diffs = mass_diffs[~mask]
        
    # calculate the FDR combined for all uncommon mass difference groups
    # and separately for each common mass difference group
    for psm in itertools.chain(
            filter_fdr(groups_uncommon, fdr),
            *[filter_fdr(group, fdr) for group in groups_common]):
        yield psm
