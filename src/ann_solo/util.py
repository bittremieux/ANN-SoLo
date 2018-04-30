import collections
import itertools

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


def filter_group_fdr(psms, fdr=0.01, tol_mass=0, tol_mode=None,
                     min_group_size=5):
    """
    Filter PSMs exceeding the given FDR.
    
    PSMs are binned on their precursor mass difference prior to FDR filtering.
    FDR filtering is applied separately to each common PSM bin and combined to
    all uncommon PSM bins.

    Args:
        psms: An iterable of `SpectrumMatch` PSMs to be filtered based on FDR.
        fdr: The minimum FDR threshold for filtering.
        tol_mass: The bin width to bin PSMs. If None no binning is performed.
        tol_mode: The unit in which the bin width is specified ('Da' or 'ppm').
        If None no grouping is performed.
        min_group_size: The minimum number of PSMs that should be present in
            a bin for it to be considered common.

    Returns:
        A generator of the `SpectrumMatch` PSMs with an FDR below the given FDR
        threshold. Each PSM is assigned its q-value in the `q` variable.
    """
    if tol_mass is not None:
        if tol_mode == 'Da':
            _get_bin = lambda mass_diff, psm: int(mass_diff // tol_mass)
        elif tol_mode == 'ppm':
            _get_bin = (lambda mass_diff, psm:
                        int((mass_diff / psm.exp_mass_to_charge * 10 ** 6)
                            // tol_mass))
        else:
            _get_bin = lambda mass_diff, psm: -1
    else:
        _get_bin = lambda mass_diff, psm: -1
    
    mass_bins = collections.defaultdict(list)
    mass_diffs = [(psm.exp_mass_to_charge - psm.calc_mass_to_charge)
                  * psm.charge for psm in psms]
    for mass_diff, psm in zip(mass_diffs, psms):
        mass_bins[_get_bin(mass_diff, psm)].append(psm)
    
    groups_common, groups_uncommon = [], []
    for group in mass_bins.values():
        if len(group) >= min_group_size:
            groups_common.append(group)
        else:
            groups_uncommon.extend(group)
    
    # calculate the FDR combined for all uncommon mass bins
    # and separately for each common mass bin
    for psm in itertools.chain(
            filter_fdr(groups_uncommon, fdr),
            *[filter_fdr(group, fdr) for group in groups_common]):
        yield psm
