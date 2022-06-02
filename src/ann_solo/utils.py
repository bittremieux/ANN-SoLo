import itertools
import operator
from typing import Iterator

import numpy as np
import pyteomics.auxiliary

from ann_solo.spectrum import SpectrumSpectrumMatch


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
            is_decoy=operator.attrgetter('is_decoy'), remove_decoy=True,
            formula=1, correction=0, full_output=True):
        ssm.q = q
        if q <= fdr:
            yield ssm
        else:
            break


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
