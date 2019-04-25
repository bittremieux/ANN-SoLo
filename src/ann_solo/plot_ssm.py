import argparse
import os
import urllib.parse as urlparse

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from spectrum_utils import plot
from spectrum_utils.spectrum import FragmentAnnotation

from ann_solo import reader
from ann_solo import spectrum_match
from ann_solo.config import config
from ann_solo.spectrum import process_spectrum


def set_matching_peaks(library_spectrum, query_spectrum):
    peak_matches = spectrum_match.get_best_match(
        query_spectrum, [library_spectrum],
        config.fragment_mz_tolerance, config.allow_peak_shifts)[2]
    query_spectrum.annotation = np.full_like(query_spectrum.mz, None, object)
    for peak_match in peak_matches:
        library_annotation = library_spectrum.annotation[peak_match[1]]
        if library_annotation is not None:
            query_spectrum.annotation[peak_match[0]] = library_annotation
        else:
            fragment_annotation = FragmentAnnotation('z', 1, 1, 0)
            fragment_annotation.ion_type = 'unknown'
            query_spectrum.annotation[peak_match[0]] =\
                library_spectrum.annotation[peak_match[1]] =\
                fragment_annotation


def main():
    # Load the cmd arguments.
    parser = argparse.ArgumentParser(
        description='Visualize spectrum–spectrum matches from your '
                    'ANN-SoLo identification results')
    parser.add_argument(
        'mztab_filename', help='Identifications in mzTab format')
    parser.add_argument(
        'query_id', help='The identifier of the query to visualize')
    args = parser.parse_args()

    # Read the mzTab file.
    metadata = {}
    with open(args.mztab_filename) as f_mztab:
        for line in f_mztab:
            line_split = line.strip().split('\t')
            if line_split[0] == 'MTD':
                metadata[line_split[1]] = line_split[2]
            else:
                break   # Metadata lines should be on top.
    ssms = reader.read_mztab_ssms(args.mztab_filename)
    # make sure the SSM ids are strings.
    ssms.index = ssms.index.map(str)

    # Recreate the search configuration.
    settings = []
    # Search settings.
    for key in metadata:
        if 'software[1]-setting' in key:
            param = metadata[key][: metadata[key].find(' ')]
            value = metadata[key][metadata[key].rfind(' ') + 1:]
            if value != 'None':
                if value != 'False':
                    settings.append('--{}'.format(param))
                if value not in ('False', 'True'):
                    settings.append(value)
    # File names.
    settings.append('dummy_spectral_library_filename')
    settings.append('dummy_query_filename')
    settings.append('dummy_output_filename')
    config.parse(' '.join(settings))

    # Retrieve information on the requested query.
    query_id = args.query_id
    query_uri = urlparse.urlparse(urlparse.unquote(
        metadata['ms_run[1]-location']))
    query_filename = os.path.abspath(os.path.join(
        query_uri.netloc, query_uri.path))
    ssm = ssms.loc[query_id]
    library_id = ssm['accession']
    library_uri = urlparse.urlparse(urlparse.unquote(ssm['database']))
    library_filename = os.path.abspath(os.path.join(
        library_uri.netloc, library_uri.path))
    score = ssm['search_engine_score[1]']

    # Read library and query spectrum.
    with reader.SpectralLibraryReader(library_filename) as lib_reader:
        library_spectrum = lib_reader.get_spectrum(library_id, True)
    query_spectrum = None
    for spec in reader.read_mgf(query_filename):
        if spec.identifier == query_id:
            query_spectrum = process_spectrum(spec, False)
            # Make sure that the precursor charge is set for query spectra
            # with a undefined precursor charge.
            query_spectrum.precursor_charge = library_spectrum.precursor_charge
            break
    # verify that the query spectrum was found
    if query_spectrum is None:
        raise ValueError('Could not find the specified query spectrum')

    # Set the matching peaks in the query spectrum to correctly color them.
    set_matching_peaks(library_spectrum, query_spectrum)
    # Modify the colors to differentiate non-matching peaks.
    plot.colors[None] = '#757575'

    # Plot the match.
    fig, ax = plt.subplots(figsize=(20, 10))

    # Plot without annotations.
    plot.mirror(query_spectrum, library_spectrum, True, False, ax)
    # Add annotations to the library spectrum.
    max_intensity = library_spectrum.intensity.max()
    for i, annotation in enumerate(library_spectrum.annotation):
        if annotation is not None and annotation.ion_type != 'unknown':
            x = library_spectrum.mz[i]
            y = -library_spectrum.intensity[i] / max_intensity
            ax.text(x, y, str(annotation),
                    color=plot.colors[annotation.ion_type], zorder=5,
                    horizontalalignment='right', verticalalignment='center',
                    rotation=90, rotation_mode='anchor')

    ax.set_ylim(-1.1, 1.05)

    ax.text(0.5, 1.06, f'{library_spectrum.peptide}, Score: {score:.3f}',
            horizontalalignment='center', verticalalignment='bottom',
            fontsize='x-large', fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 1.02, f'File: {os.path.basename(query_filename)}, '
                       f'Scan: {query_spectrum.identifier}, '
                       f'Precursor m/z: {query_spectrum.precursor_mz:.4f}, '
                       f'Library m/z: {library_spectrum.precursor_mz:.4f}, '
                       f'Charge: {query_spectrum.precursor_charge}',
            horizontalalignment='center', verticalalignment='bottom',
            fontsize='large', transform=ax.transAxes)

    plt.savefig(f'{query_id}.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
