import datetime
import logging
import os
import sqlite3

import configargparse
import tqdm

import reader


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)s/%(processName)s] %(module)s.%(funcName)s : %(message)s',
                        level=logging.DEBUG)

    config_parser = configargparse.ArgParser(description='Convert SpectraST to BiblioSpec')
    config_parser.add_argument('spectrast_filename', help='Input SpectraST spectral library file in sptxt/splib format')
    config_parser.add_argument('spql_filename', help='Output spectral library file in spql format')
    args = config_parser.parse_args()

    # check if the output filename has the spql extension and add if required
    spql_filename = args.spql_filename
    if os.path.splitext(spql_filename)[1].lower() != '.spql':
        spql_filename += '.spql'

    logging.info('Convert {} to {}'.format(args.spectrast_filename, spql_filename))

    major_version = 1
    minor_version = 0

    # create the sqlite file
    conn = sqlite3.connect(spql_filename, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()
    # create the database tables
    cursor.execute('CREATE TABLE LibInfo(libLSID TEXT, createTime TEXT, numSpecs INTEGER, '
                   'majorVersion INTEGER, minorVersion INTEGER)')
    cursor.execute('CREATE TABLE RefSpectra (id INTEGER primary key autoincrement not null, peptideSeq VARCHAR(150), '
                   'precursorMZ REAL, precursorCharge INTEGER, numPeaks INTEGER, isDecoy BOOLEAN)')
    cursor.execute('CREATE TABLE RefSpectraPeaks(RefSpectraID INTEGER, peakMZ array, peakIntensity array, peakAnnotation array)')
    # create the database indices
    cursor.execute('CREATE INDEX idxPeptide ON RefSpectra (peptideSeq, precursorCharge)')
    cursor.execute('CREATE INDEX idxRefIdPeaks ON RefSpectraPeaks (RefSpectraID)')

    # add all spectra
    num_specs = 0
    with reader.get_spectral_library_reader(args.spectrast_filename) as lib_reader:
        for spectrum, _ in tqdm.tqdm(lib_reader._get_all_spectra(), desc='Spectra converted', unit='spectra', smoothing=0):
            num_specs += 1
            cursor.execute('INSERT INTO RefSpectra VALUES (?, ?, ?, ?, ?, ?)',
                           (spectrum.identifier, spectrum.peptide, spectrum.precursor_mz, spectrum.precursor_charge,
                            len(spectrum.masses), 1 if spectrum.is_decoy else 0))
            cursor.execute('INSERT INTO RefSpectraPeaks VALUES (?, ?, ?, ?)',
                           (spectrum.identifier, spectrum.masses, spectrum.intensities, spectrum.annotations))

    # add the global information
    cursor.execute('INSERT INTO LibInfo VALUES (?, ?, ?, ?, ?)',
                   ('urn:lsid:inspector:ann_solo:{}'.format(os.path.basename(spql_filename)),
                    datetime.datetime.strftime(datetime.datetime.utcnow(), '%Y-%m-%d %H:%M:%S'),
                    num_specs, major_version, minor_version))

    conn.commit()
    conn.close()

    logging.info('Finished creating {}'.format(spql_filename))

    logging.shutdown()
