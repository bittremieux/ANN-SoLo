import datetime
import logging
import os
import pathlib
import re

from . import __version__
from ann_solo.config import config


def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


def write_mztab(identifications, filename, lib_reader):
    # check if the filename contains the mztab extension and add if required
    if os.path.splitext(filename)[1].lower() != '.mztab':
        filename += '.mztab'

    logging.info('Saving identifications to file %s', filename)

    # collect the necessary metadata
    metadata = [
        ('mzTab-version', '1.0.0'),
        ('mzTab-mode', 'Summary'),
        ('mzTab-type', 'Identification'),
        ('mzTab-ID', 'ANN-SoLo_{}'.format(filename)),
        ('title', 'ANN-SoLo identification file "{}"'.format(filename)),
        ('description', 'Identification results of file "{}" against spectral '
                        'library file "{}"'.format(
                            config.query_filename,
                            config.spectral_library_filename)),
        ('software[1]', '[MS, MS:1001456, ANN-SoLo, {}]'.format(__version__)),
        ('psm_search_engine_score[1]', '[MS, MS:1001143, search engine '
                                       'specific score for PSMs,]'),
        ('psm_search_engine_score[2]', '[MS, MS:1002354, PSM-level q-value,]'),
        ('ms_run[1]-location', pathlib.Path(
            os.path.abspath(config.query_filename)).as_uri()),
        ('fixed_mod[1]', '[MS, MS:1002453, No fixed modifications searched,]'),
        ('variable_mod[1]', '[MS, MS:1002454, No variable modifications '
                            'searched,]'),
    ]

    # add relevant configuration settings
    config_keys = [
        'resolution', 'min_mz', 'max_mz', 'remove_precursor',
        'remove_precursor_tolerance', 'min_intensity', 'min_peaks',
        'min_mz_range', 'max_peaks_used', 'scaling',
        'precursor_tolerance_mass', 'precursor_tolerance_mode',
        'precursor_tolerance_mass_open', 'precursor_tolerance_mode_open',
        'fragment_mz_tolerance', 'allow_peak_shifts', 'fdr', 'mode']
    if config.mode == 'ann':
        config_keys.extend(['bin_size', 'hash_len', 'num_candidates',
                            'num_list', 'num_probe'])
    for i, key in enumerate(config_keys):
        metadata.append(('software[1]-setting[{}]'.format(i),
                         '{} = {}'.format(key, config[key])))

    version = lib_reader.get_version()
    database_version = '{} ({} entries)'.format(
        datetime.datetime.strftime(version[0], '%Y-%m-%d'), version[1])\
        if version is not None else 'null'

    with open(filename, 'w') as f_out:
        # metadata section
        for m in metadata:
            f_out.write('\t'.join(['MTD'] + list(m)) + '\n')

        # PSMs
        f_out.write('\t'.join([
            'PSH', 'sequence', 'PSM_ID', 'accession', 'unique', 'database',
            'database_version', 'search_engine', 'search_engine_score[1]',
            'search_engine_score[2]', 'modifications', 'retention_time',
            'charge', 'exp_mass_to_charge', 'calc_mass_to_charge',
            'spectra_ref', 'pre', 'post', 'start', 'end',
            'opt_ms_run[1]_cv_MS:1002217_decoy_peptide',
            'opt_ms_run[1]_num_candidates', 'opt_ms_run[1]_time_total',
            'opt_ms_run[1]_time_candidates', 'opt_ms_run[1]_time_match'])
                    + '\n')
        # SSMs sorted by their query identifier.
        for ssm in sorted(identifications,
                          key=lambda ssm: natural_sort_key(ssm.identifier)):
            f_out.write('\t'.join([
                'PSM',
                ssm.sequence,
                str(ssm.identifier),
                str(ssm.accession),
                'null',
                pathlib.Path(os.path.abspath(
                    config.spectral_library_filename)).as_uri(),
                database_version,
                '[MS, MS:1001456, ANN SoLo,]',
                str(ssm.search_engine_score),
                str(ssm.q),
                'null',
                str(ssm.retention_time),
                str(ssm.charge),
                str(ssm.exp_mass_to_charge),
                str(ssm.calc_mass_to_charge),
                f'ms_run[1]:spectrum={ssm.identifier}',
                'null', 'null', 'null', 'null',
                f'{ssm.is_decoy:d}',
                str(ssm.num_candidates)]) + '\n')

    logging.info('Identifications saved to file %s', filename)
