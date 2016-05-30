import logging
import os
import pathlib
import re

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
        ('software[1]', '[MS, MS:1001456, ANN spectral library, 0.1]'),
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
    if config.mode == 'ann':
        config_keys.extend(['num_trees', 'bin_size', 'num_candidates', 'search_k', 'ann_cutoff'])
    for i, key in enumerate(config_keys):
        metadata.append(('software[1]-setting[{}]'.format(i), '{} = {}'.format(key, config[key])))

    with open(filename, 'w') as f_out:
        # metadata section
        for m in metadata:
            f_out.write('\t'.join(['MTD', *m]) + '\n')

        # PSMs
        f_out.write('\t'.join(['PSH', 'sequence', 'PSM_ID', 'accession', 'unique', 'database', 'database_version',
                               'search_engine', 'search_engine_score[1]', 'modifications', 'retention_time', 'charge',
                               'exp_mass_to_charge', 'calc_mass_to_charge', 'spectra_ref', 'pre', 'post', 'start',
                               'end', 'is_decoy', 'num_candidates', 'time_total', 'time_candidates', 'time_match']) +
                    '\n')
        # PSMs sorted by their query id
        for identification in sorted(identifications, key=lambda i: natural_sort_key(i.query_id)):
            f_out.write('\t'.join(['PSM', identification.sequence, str(identification.query_id), 'null', 'null', 'null',
                                   'null', '[MS, MS:1001456, ANN spectral library,]',
                                   str(identification.search_engine_score), 'null', str(identification.retention_time),
                                   str(identification.charge), str(identification.exp_mass_to_charge),
                                   str(identification.calc_mass_to_charge),
                                   'ms_run[1]:spectrum={}'.format(identification.query_id), 'null', 'null', 'null',
                                   'null', str(identification.is_decoy), str(identification.num_candidates),
                                   str(identification.time_total), str(identification.time_candidates),
                                   str(identification.time_match)]) + '\n')

    logging.info('Identifications saved to file %s', filename)
