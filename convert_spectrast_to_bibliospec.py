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
    config_parser.add_argument('blib_filename', help='Output BiblioSpec spectral library file in blib format')
    args = config_parser.parse_args()

    # check if the BiblioSpec filename has the blib extension and add if required
    blib_filename = args.blib_filename
    if os.path.splitext(blib_filename)[1].lower() != '.blib':
        blib_filename += '.blib'

    logging.info('Convert {} to {}'.format(args.spectrast_filename, blib_filename))

    major_version = 1
    minor_version = 1

    # create the sqlite file
    conn = sqlite3.connect(blib_filename)
    cursor = conn.cursor()
    # create the database tables
    cursor.execute('CREATE TABLE LibInfo(libLSID TEXT, createTime TEXT, numSpecs INTEGER, '
                   'majorVersion INTEGER, minorVersion INTEGER)')
    cursor.execute('CREATE TABLE Modifications (id INTEGER primary key autoincrement not null, '
                   'RefSpectraID INTEGER, position INTEGER, mass REAL)')
    cursor.execute('CREATE TABLE RefSpectra (id INTEGER primary key autoincrement not null, peptideSeq VARCHAR(150), '
                   'precursorMZ REAL, precursorCharge INTEGER, peptideModSeq VARCHAR(200), prevAA CHAR(1), '
                   'nextAA CHAR(1), copies INTEGER, numPeaks INTEGER, ionMobilityValue REAL, ionMobilityType INTEGER, '
                   'retentionTime REAL, fileID INTEGER, SpecIDinFile VARCHAR(256), score REAL, scoreType TINYINT)')
    cursor.execute('CREATE TABLE RefSpectraPeaks(RefSpectraID INTEGER, peakMZ BLOB, peakIntensity BLOB)')
    cursor.execute('CREATE TABLE RetentionTimes (RefSpectraID INTEGER, RedundantRefSpectraID INTEGER, '
                   'SpectrumSourceID INTEGER, ionMobilityValue REAL, ionMobilityType INTEGER, retentionTime REAL, '
                   'bestSpectrum INTEGER, FOREIGN KEY(RefSpectraID) REFERENCES RefSpectra(id))')
    cursor.execute('CREATE TABLE ScoreTypes (id INTEGER PRIMARY KEY, scoreType VARCHAR(128))')
    cursor.execute('CREATE TABLE SpectrumSourceFiles (id INTEGER PRIMARY KEY autoincrement not null, '
                   'fileName VARCHAR(512))')
    # create the database indices
    cursor.execute('CREATE INDEX idxPeptide ON RefSpectra (peptideSeq, precursorCharge)')
    cursor.execute('CREATE INDEX idxPeptideMod ON RefSpectra (peptideModSeq, precursorCharge)')
    cursor.execute('CREATE INDEX idxRefIdPeaks ON RefSpectraPeaks (RefSpectraID)')

    # add all spectra
    num_specs = 0
    with reader.get_spectral_library_reader(args.spectrast_filename) as lib_reader:
        for spectrum, _ in tqdm.tqdm(lib_reader.get_all_spectra(), desc='Spectra converted', unit='spectra', smoothing=0):
            num_specs += 1
            cursor.execute('INSERT INTO RefSpectra VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                           (spectrum.identifier, spectrum.peptide,
                            spectrum.precursor_mz, spectrum.precursor_charge, spectrum.peptide,
                            '-', '-', 1,                # prevAA, nextAA, copies
                            len(spectrum.masses),       # numPeaks
                            0.0, 0, 0.0,                # ionMobilityValue, ionMobilityType, retentionTime
                            1, spectrum.identifier,     # fileID, SpecIDInFile
                            0.0, 0))                    # score, scoreType
            cursor.execute('INSERT INTO RefSpectraPeaks VALUES (?, ?, ?)',
                           (spectrum.identifier, spectrum.masses, spectrum.intensities))

    # add the global information
    cursor.execute('INSERT INTO LibInfo VALUES (?, ?, ?, ?, ?)',
                   ('urn:lsid:inspector:ann_solo:bibliospec:nr:{}'.format(os.path.basename(blib_filename)),
                    datetime.datetime.strftime(datetime.datetime.utcnow(), '%Y-%m-%d %H:%M:%S%z'),
                    num_specs, major_version, minor_version))
    cursor.execute('INSERT INTO ScoreTypes VALUES (?, ?)', (0, 'UNKNOWN'))
    cursor.execute('INSERT INTO SpectrumSourceFiles VALUES (?, ?)', (0, os.path.abspath(args.spectrast_filename)))

    conn.commit()
    conn.close()

    logging.info('Finished creating {}'.format(blib_filename))

    logging.shutdown()
