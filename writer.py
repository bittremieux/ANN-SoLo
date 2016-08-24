import datetime
import logging
import os
import re
try:
    import pathlib
except ImportError:
    import pathlib2 as pathlib

import reader
from config import config


def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]


def write_mztab(identifications, filename):
    # check if the filename contains the mztab extension and add if required
    if os.path.splitext(filename)[1].lower() != '.mztab':
        filename += '.mztab'

    logging.info('Saving identifications to file %s', filename)

    # collect the necessary metadata
    metadata = [
        ('mzTab-version', '1.0.0'),
        ('mzTab-mode', 'Summary'),
        ('mzTab-type', 'Identification'),
        ('description', 'Identification results of file "{}" against spectral library file "{}"'.format(
            config.query_filename, config.spectral_library_filename)),
        ('software[1]', '[MS, MS:1001456, ANN SoLo, 0.0.dev]'),
        ('psm_search_engine_score[1]', '[MS, MS:1001143, search engine specific score for PSMs,]'),
        ('ms_run[1]-location', pathlib.Path(os.path.abspath(config.query_filename)).as_uri()),
        ('fixed_mod[1]', '[MS, MS:1002453, No fixed modifications searched,]'),
        ('variable_mod[1]', '[MS, MS:1002454, No variable modifications searched,]'),
    ]

    # add relevant configuration settings
    config_keys = ['resolution', 'min_mz', 'max_mz', 'remove_precursor', 'remove_precursor_tolerance', 'min_intensity',
                   'min_dynamic_range', 'min_peaks', 'min_mz_range', 'max_peaks_used', 'scaling',
                   'precursor_tolerance_mass', 'precursor_tolerance_mode', 'fragment_mz_tolerance', 'allow_peak_shifts',
                   'mode']
    if config.mode == 'annoy':
        config_keys.extend(['bin_size', 'num_candidates', 'ann_cutoff', 'num_trees', 'search_k'])
    elif config.mode == 'hnsw':
        config_keys.extend(['bin_size', 'num_candidates', 'ann_cutoff', 'M', 'ef'])
    for i, key in enumerate(config_keys):
        metadata.append(('software[1]-setting[{}]'.format(i), '{} = {}'.format(key, config[key])))

    with reader.get_spectral_library_reader(os.path.abspath(config.spectral_library_filename)) as lib_reader:
        version = lib_reader.get_version()
        database_version = '{} ({} entries)'.format(datetime.datetime.strftime(version[0], '%Y-%m-%d'), version[1])\
                           if version is not None else 'null'

    with open(filename, 'w') as f_out:
        # metadata section
        for m in metadata:
            f_out.write('\t'.join(['MTD'] + list(m)) + '\n')

        # PSMs
        f_out.write('\t'.join(['PSH', 'sequence', 'PSM_ID', 'accession', 'unique', 'database', 'database_version',
                               'search_engine', 'search_engine_score[1]', 'modifications', 'retention_time', 'charge',
                               'exp_mass_to_charge', 'calc_mass_to_charge', 'spectra_ref', 'pre', 'post', 'start',
                               'end', 'opt_ms_run[1]_cv_MS:1002217_decoy_peptide', 'opt_ms_run[1]_num_candidates',
                               'opt_ms_run[1]_time_total', 'opt_ms_run[1]_time_candidates', 'opt_ms_run[1]_time_match']) + '\n')
        # PSMs sorted by their query id
        for identification in sorted(identifications, key=lambda i: natural_sort_key(i.query_id)):
            f_out.write('\t'.join(['PSM', identification.sequence, str(identification.query_id),
                                   str(identification.library_id), 'null',
                                   pathlib.Path(os.path.abspath(config.spectral_library_filename)).as_uri(),
                                   database_version, '[MS, MS:1001456, ANN SoLo,]',
                                   str(identification.search_engine_score), 'null', str(identification.retention_time),
                                   str(identification.charge), str(identification.exp_mass_to_charge),
                                   str(identification.calc_mass_to_charge),
                                   'ms_run[1]:spectrum={}'.format(identification.query_id), 'null', 'null', 'null',
                                   'null', str(1 if identification.is_decoy else 0), str(identification.num_candidates),
                                   str(identification.time_total), str(identification.time_candidates),
                                   str(identification.time_match)]) + '\n')

    logging.info('Identifications saved to file %s', filename)
