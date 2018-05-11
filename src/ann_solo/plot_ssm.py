import argparse
import os
import urllib.parse as urlparse

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

from ann_solo import reader, spectrum_match
from ann_solo.config import config


sns.set_context('notebook')
sns.set_style('white')


colors = {'a': 'green', 'b': 'blue', 'y': 'red',
          'unknown': 'black', None: 'darkgrey'}
zorders = {'a': 2, 'b': 3, 'y': 3, 'unknown': 1, None: 0}


def get_matching_peaks(library_spectrum, query_spectrum):
    _, score, peak_matches = spectrum_match.get_best_match(
            query_spectrum, [library_spectrum],
            allow_shift=config.allow_peak_shifts)
    library_matches, query_matches = {}, {}
    for peak_match in peak_matches:
        query_matches[peak_match[0]] = library_matches[peak_match[1]] = (
            'unknown' if library_spectrum.annotations[peak_match[1]] is None
            else library_spectrum.annotations[peak_match[1]][0][0])

    return library_matches, query_matches, score


def main():
    # load the cmd arguments
    parser = argparse.ArgumentParser(
            description='Visualize spectrum-spectrum matches from your '
                        'ANN-SoLo identification results')
    parser.add_argument(
            'mztab_filename', help='Identifications in mzTab format')
    parser.add_argument(
            'query_id', help='The identifier of the query to visualize')
    args = parser.parse_args()

    # read the mzTab file
    metadata = {}
    with open(args.mztab_filename) as f_mztab:
        for line in f_mztab:
            line_split = line.strip().split('\t')
            if line_split[0] == 'MTD':
                metadata[line_split[1]] = line_split[2]
            else:
                break   # metadata lines should be on top
    psms = reader.read_mztab_psms(args.mztab_filename)
    # make sure the PSM id's are strings
    psms.index = psms.index.map(str)

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
    query_uri = urlparse.urlparse(urlparse.unquote(
            metadata['ms_run[1]-location']))
    query_filename = os.path.abspath(os.path.join(
            query_uri.netloc, query_uri.path))
    psm = psms.loc[query_id]
    library_id = psm['accession']
    library_uri = urlparse.urlparse(urlparse.unquote(psm['database']))
    library_filename = os.path.abspath(os.path.join(
            library_uri.netloc, library_uri.path))
    score = psm['search_engine_score[1]']

    # read library and query spectrum
    with reader.get_spectral_library_reader(library_filename) as lib_reader:
        library_spectrum = lib_reader.get_spectrum(library_id, True)
    query_spectrum = None
    for spec in reader.read_mgf(query_filename):
        if spec.identifier == query_id:
            query_spectrum = spec
            query_spectrum.process_peaks()
            # make sure that the precursor charge is set for query spectra
            # with a undefined precursor charge
            if query_spectrum.precursor_charge is None:
                query_spectrum.precursor_charge =\
                    library_spectrum.precursor_charge
            break
    # verify that the query spectrum was found
    if query_spectrum is None:
        raise ValueError('Could not find the specified query spectrum')

    # compute the matching peaks
    library_matches, query_matches, _ =\
        get_matching_peaks(library_spectrum, query_spectrum)

    # plot the match
    fix, ax = plt.subplots(figsize=(20, 10))

    # query spectrum on top
    max_intensity = np.max(query_spectrum.intensities)
    for i, (mass, intensity) in enumerate(zip(
            query_spectrum.masses, query_spectrum.intensities)):
        color = colors[query_matches.get(i)]
        zorder = zorders[query_matches.get(i)]
        ax.plot([mass, mass], [0, intensity / max_intensity],
                color=color, zorder=zorder)
    # library spectrum mirrored underneath
    max_intensity = np.max(library_spectrum.intensities)
    for i, (mass, intensity, annotation) in enumerate(
            zip(library_spectrum.masses,
                library_spectrum.intensities,
                library_spectrum.annotations)):
        color = colors[library_matches.get(i)]
        zorder = zorders[library_matches.get(i)]
        ax.plot([mass, mass], [0, -intensity / max_intensity],
                color=color, zorder=zorder)
        if annotation is not None:
            ax.text(mass - 5, -intensity / max_intensity - 0.05,
                    '{}{}'.format(annotation[0], '+' * annotation[1]),
                    color=color, rotation=270)

    # horizontal line between the two spectra
    ax.axhline(0, color='black')
    # consistent axes range and labels
    ax.set_xticks(np.arange(0, config.max_mz, 200))
    ax.set_xlim(config.min_mz, config.max_mz)
    y_ticks = np.arange(-1, 1.05, 0.25)
    y_ticklabels = np.arange(-1, 1.05, 0.25)
    y_ticklabels[y_ticklabels < 0] = -y_ticklabels[y_ticklabels < 0]
    y_ticklabels = ['{:.0%}'.format(l) for l in y_ticklabels]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticklabels)
    ax.set_ylim(-1.15, 1.05)

    # show major/minor tick lines
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(b=True, which='major', color='lightgrey',
            linestyle='--', linewidth=1.0)
    ax.grid(b=True, which='minor', color='lightgrey',
            linestyle='--', linewidth=0.5)
    
    # small tick labels
    ax.tick_params(axis='both', which='both', labelsize='small')

    ax.set_xlabel('m/z')
    ax.set_ylabel('Intensity')

    ax.text(0.5, 1.06,
            '{}, Score: {:.3f}'.format(library_spectrum.peptide, score),
            horizontalalignment='center', verticalalignment='bottom',
            fontsize='x-large', fontweight='bold',
            transform=plt.gca().transAxes)
    ax.text(0.5, 1.02,
            'File: {}, Scan: {}, Precursor m/z: {:.4f}, '
            'Library m/z: {:.4f}, Charge: {}'.format(
                    os.path.basename(query_filename),
                    query_spectrum.identifier,
                    query_spectrum.precursor_mz,
                    library_spectrum.precursor_mz,
                    query_spectrum.precursor_charge),
            horizontalalignment='center', verticalalignment='bottom',
            fontsize='large', transform=plt.gca().transAxes)

    plt.savefig('{}.png'.format(query_id), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
