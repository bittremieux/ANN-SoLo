import argparse
import os
import urllib.parse as urlparse

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import reader
import spectrum_match
from config import config


sns.set_context('notebook')
sns.set_style('white')


colors = {'a': 'green', 'b': 'blue', 'y': 'red', None: 'black'}


def get_matching_peaks(library_spectrum, query_spectrum):
    _, score, peak_matches = spectrum_match.get_best_match(query_spectrum, [library_spectrum],
                                                           allow_shift=config.allow_peak_shifts)
    library_matches, query_matches = {}, {}
    for peak_match in peak_matches:
        query_matches[peak_match[0]] = library_matches[peak_match[1]] = \
            colors[library_spectrum.annotations[peak_match[1]][0][0]
            if library_spectrum.annotations[peak_match[1]] is not None else None]

    return library_matches, query_matches, score


if __name__ == '__main__':
    # load the cmd arguments
    parser = argparse.ArgumentParser(description='Visualize PSMs')
    parser.add_argument('mztab_filename', help='Identifications in mzTab format')
    parser.add_argument('query_id', help='The identifier of the query to visualize')
    args = parser.parse_args()

    # read the mzTab file
    metadata = {}
    psms = {}
    psm_header = None
    with open(args.mztab_filename) as f_mztab:
        for line in f_mztab:
            line_split = line.strip().split('\t')
            if line_split[0] == 'MTD':
                metadata[line_split[1]] = line_split[2]
            elif line_split[0] == 'PSH':
                psm_header = line_split[1:]
            elif line_split[0] == 'PSM':
                psm = {key: value for key, value in zip(psm_header, line_split[1:])}
                psms[psm['PSM_ID']] = psm

    # recreate the search configuration
    settings = []
    # search settings
    for key in metadata:
        if 'software[1]-setting' in key:
            param = metadata[key][: metadata[key].find(' ')]
            value = metadata[key][metadata[key].rfind(' ') + 1:]
            if value != 'False':
                settings.append('--{}'.format(param))
            if value != 'False' and value != 'True':
                settings.append(value)
    # file names
    settings.append('dummy_spectral_library_filename')
    settings.append('dummy_query_filename')
    settings.append('dummy_output_filename')
    config.parse(' '.join(settings))

    # retrieve information on the requested query
    query_id = args.query_id
    query_uri = urlparse.urlparse(urlparse.unquote(metadata['ms_run[1]-location']))
    query_filename = os.path.abspath(os.path.join(query_uri.netloc, query_uri.path))
    psm = psms[query_id]
    library_id = psm['accession']
    library_uri = urlparse.urlparse(urlparse.unquote(psm['database']))
    library_filename = os.path.abspath(os.path.join(library_uri.netloc, library_uri.path))
    score = psm['search_engine_score[1]']

    # read library and query spectrum
    with reader.get_spectral_library_reader(library_filename) as library_reader:
        library_spectrum = library_reader.get_spectrum(library_id, True)
    query_spectrum = None
    for spec in reader.read_mgf(query_filename):
        if spec.identifier == query_id:
            query_spectrum = spec
            query_spectrum.process_peaks()
            # make sure that the precursor charge is set for query spectra with a undefined precursor charge
            if query_spectrum.precursor_charge is None:
                query_spectrum.precursor_charge = library_spectrum.precursor_charge
            break
    # verify that the query spectrum was found
    if query_spectrum is None:
        raise ValueError('Could not find the specified query spectrum')

    # compute the matching peaks
    library_matches, query_matches, score = get_matching_peaks(library_spectrum, query_spectrum)

    # plot the match
    plt.figure(figsize=(20, 10))

    # query spectrum on top
    for i, (mass, intensity) in enumerate(zip(query_spectrum.masses, query_spectrum.intensities)):
        is_match = i in query_matches
        plt.plot([mass, mass], [0, intensity], color=query_matches[i] if i in query_matches else 'darkgrey')
    # library spectrum mirrored underneath
    for i, (mass, intensity, annotation) in enumerate(
            zip(library_spectrum.masses, library_spectrum.intensities, library_spectrum.annotations)):
        is_match = i in library_matches
        plt.plot([mass, mass], [0, -1 * intensity], color=library_matches[i] if i in library_matches else 'darkgrey')
        if annotation is not None:
            plt.text(mass - 5, -1 * intensity - 0.01, '{}{}'.format(annotation[0], '+' * annotation[1]),
                     color=library_matches[i] if i in library_matches else 'darkgrey', rotation=270)

    # consistent m/z range
    plt.xticks(np.arange(0, config.max_mz, 200))
    plt.xlim(config.min_mz, config.max_mz)

    # horizontal line between the two spectra
    plt.axhline(0, color='black')
    # make sure this is centered vertically
    ylim = np.amax(np.fabs(plt.ylim()))
    plt.ylim(-1.1 * ylim, 1.1 * ylim)
    # hide the y-axis labels
    plt.gca().set_yticklabels([])

    # show major/minor tick lines
    plt.gca().xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    plt.gca().yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    plt.grid(b=True, which='major', color='lightgrey', linestyle='--', linewidth=1.0)
    plt.grid(b=True, which='minor', color='lightgrey', linestyle='--', linewidth=0.5)

    plt.xlabel('m/z')

    plt.text(0.5, 1.06, '{}, Score: {:.3f}'.format(library_spectrum.peptide, score),
             horizontalalignment='center', verticalalignment='bottom', fontsize='x-large', fontweight='bold',
             transform=plt.gca().transAxes)
    plt.text(0.5, 1.02, 'File: {}, Scan: {}, Precursor m/z: {:.4f}, Library m/z: {:.4f}, Charge: {}'.format(
        os.path.basename(query_filename), query_spectrum.identifier, query_spectrum.precursor_mz,
        library_spectrum.precursor_mz, query_spectrum.precursor_charge),
             horizontalalignment='center', verticalalignment='bottom', fontsize='large', transform=plt.gca().transAxes)

    plt.savefig('{}.png'.format(query_id), bbox_inches='tight')
    plt.close()
