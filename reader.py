import abc
import collections
import io
import logging
import mmap
import os
import six
import sqlite3
import struct
try:
    from functools import lru_cache
except ImportError:
    from functools32 import lru_cache

import joblib
import numpy as np
import pandas as pd
import tqdm
from pyteomics import mgf

import spectrum
from config import config


# define FileNotFoundError for Python 2
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


# convert NumPy arrays to/from BLOBs
def adapt_array(arr):
    buf = io.BytesIO()
    joblib.dump(arr, buf, compress=0, protocol=2)
    return sqlite3.Binary(buf.getvalue())


def convert_array(text):
    return joblib.load(io.BytesIO(text))


sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter('array', convert_array)


def get_spectral_library_reader(filename, config_match_keys=None):
    if not os.path.isfile(filename):
        raise FileNotFoundError('Spectral library file {} not found'.format(filename))

    base_filename, ext = os.path.splitext(filename)
    if ext == '.spql':
        return SqliteSpecReader(filename, config_match_keys)
    elif ext == '.splib' or ext == '.sptxt':
        splib_exists = os.path.isfile(base_filename + '.splib')
        sptxt_exists = os.path.isfile(base_filename + '.sptxt')
        if splib_exists:
            # prefer an splib file because it is faster to read
            return SplibReader(base_filename + '.splib', config_match_keys)
        elif sptxt_exists:
            # fall back to an sptxt file
            return SptxtReader(base_filename + '.sptxt', config_match_keys)
    else:
        raise FileNotFoundError('Unrecognized file format (supported file formats: spql, splib, sptxt)')


def verify_extension(supported_extensions, filename):
    base_name, ext = os.path.splitext(os.path.basename(filename))
    if ext.lower() not in supported_extensions:
        logging.error('Unrecognized file format: {}'.format(filename))
        raise FileNotFoundError('Unrecognized file format: {}'.format(filename))


_annotation_ion_types = frozenset(b'abcxyz')
_ignore_annotations = True


def _parse_annotation(raw):
    if raw[0] in _annotation_ion_types:     # discard peaks that don't correspond to an ion type
        first_annotation = raw.split(b',', 1)[0]
        if b'i' not in first_annotation:    # discard isotope peaks
            ion_sep = first_annotation.find(b'/')
            has_mod = first_annotation.find(b'-', 0, ion_sep) != -1 or first_annotation.find(b'+', 0, ion_sep) != -1
            if not has_mod:     # discard modified peaks
                charge_sep = first_annotation.find(b'^')
                ion_type = first_annotation[:charge_sep if charge_sep != -1 else ion_sep].decode('UTF-8')
                charge = int(first_annotation[charge_sep + 1:ion_sep] if charge_sep != -1 else 1)

                return ion_type, charge

    return None


def is_matching_config(match_keys, config1, config2):
    """
    Check if two configurations match for the specified keys.

    Args:
        match_keys: The keys on which the two configurations need to match.
        config1: The first configuration.
        config2: The second configuration.

    Returns:
        True if both configurations match for the specified keys, False if not.
    """
    filtered_config1 = {key: config1[key] for key in match_keys}
    filtered_config2 = {key: config2[key] for key in match_keys}
    return filtered_config1 == filtered_config2


@six.add_metaclass(abc.ABCMeta)
class SpectralLibraryReader(object):
    """
    Read spectra from a spectral library file.
    """

    _max_cache_size = None

    _supported_extensions = []

    def __init__(self, filename, config_match_keys=None):
        """
        Initialize the spectral library reader. Metadata for future easy access of the individual Spectra is read from
        the corresponding configuration file.

        The configuration file contains minimally for each spectrum in the spectral library its precursor charge and
        precursor mass for quickly filtering the spectra library. Furthermore, it also contains the settings used to
        construct this spectral library to make sure these match the runtime settings.

        Args:
            filename: The file name of the spectral library.
            config_match_keys: Configuration settings that need to match between the runtime configuration and the
                               loaded configuration.

        Raises:
            FileNotFoundError: The given spectral library file wasn't found.
            ValueError: The configuration file wasn't found or its settings don't correspond to the runtime settings.
        """
        self._filename = filename
        do_create = False

        # test if the given spectral library file is in a supported format
        verify_extension(self._supported_extensions, self._filename)

        logging.info('Loading the spectral library configuration')

        # verify that the configuration file corresponding to this spectral library is present
        config_filename = self._filename + '.spcfg'
        if not os.path.isfile(config_filename):
            # if not we should recreate this file prior to using the spectral library
            do_create = True
            logging.warning('Missing configuration file corresponding to this spectral library')
        else:
            # load the configuration file
            self.spec_info, load_config = joblib.load(config_filename)

            # verify that the runtime settings match the loaded settings
            if config_match_keys is not None and\
               not is_matching_config(config_match_keys, config._namespace, load_config):
                # if not we should recreate the configuration file prior to using the spectral library
                do_create = True
                logging.warning('The spectral library search engine was created using non-compatible settings')

        logging.info('Finished loading the spectral library configuration')

        # (re)create the spectral library configuration if it is missing or incorrect
        if do_create:
            self._create()

    @abc.abstractmethod
    def __enter__(self):
        return self

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @abc.abstractmethod
    def _get_all_spectra(self):
        """
        Generates all spectra from the spectral library file.

        For each individual Spectrum a tuple consisting of the Spectrum and some additional information as a nested
        tuple (containing on the type of spectral library file) are returned.

        Returns:
            A generator that yields all spectra from the spectral library file.
        """
        pass


@six.add_metaclass(abc.ABCMeta)
class SpectraSTReader(SpectralLibraryReader):
    """
    Read spectra from a SpectraST spectral library file.
    """

    _max_cache_size = None

    _supported_extensions = []

    def __enter__(self):
        self._file = open(self._filename, 'rb')
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._mm.close()
        self._file.close()

    def _create(self):
        """
        Create a new configuration file for the spectral library.

        The configuration file contains for each spectrum in the spectral library its offset for quick random-access
        reading, and its precursor mass for filtering using a precursor mass window. Finally, it also contains the
        settings used to construct this spectral library to make sure these match the runtime settings.
        """
        logging.info('Creating the spectral library configuration for file %s', self._filename)

        # read all the spectra in the spectral library
        temp_info = collections.defaultdict(lambda: {'id': [], 'precursor_mass': []})
        offsets = {}
        with self as lib_reader:
            for spec, offset in tqdm.tqdm(lib_reader._get_all_spectra(), desc='Library spectra read', unit='spectra'):
                # store the spectrum information for easy retrieval
                info_charge = temp_info[spec.precursor_charge]
                info_charge['id'].append(spec.identifier)
                info_charge['precursor_mass'].append(spec.precursor_mz)
                offsets[spec.identifier] = offset
        self.spec_info = {charge: {'id': np.asarray(charge_info['id'], np.uint32),
                                   'precursor_mass': np.asarray(charge_info['precursor_mass'], np.float32)}
                          for charge, charge_info in six.iteritems(temp_info)}
        self.spec_info['offset'] = offsets

        # store the configuration
        config_filename = self._filename + '.spcfg'
        logging.debug('Saving the spectral library configuration to file %s', config_filename)
        joblib.dump((self.spec_info, config._namespace), config_filename, compress=9, protocol=2)

        logging.info('Finished creating the spectral library configuration')

    @lru_cache(maxsize=_max_cache_size)
    def get_spectrum(self, spec_id, process_peaks=False):
        """
        Read the `Spectrum` with the specified identifier from the spectral library file.

        Args:
            spec_id: The identifier of the `Spectrum` in the spectral library file.
            process_peaks: Flag whether to process the `Spectrum`'s peaks or not.

        Returns:
            The `Spectrum` from the spectral library file with the specified identifier.
        """
        self._mm.seek(self.spec_info['offset'][spec_id])

        read_spectrum = self._read_spectrum()[0]
        if process_peaks:
            read_spectrum.process_peaks()

        return read_spectrum


class SptxtReader(SpectraSTReader):
    """
    Read spectra from a SpectraST spectral library .sptxt file.
    """

    _supported_extensions = ['.sptxt']

    def _get_all_spectra(self):
        try:
            while True:
                yield self._read_spectrum()
        except StopIteration:
            pass

    def _read_spectrum(self):
        # find the next spectrum in the file
        file_offset = self._mm.tell()
        line = self._read_line()
        while b'Name: ' not in line:
            file_offset = self._mm.tell()
            line = self._read_line()

        # read the spectrum
        # identification information
        name = line.strip()[6:]
        sep_idx = name.find(b'/')
        peptide = name[:sep_idx].decode(encoding='UTF-8')
        precursor_charge = int(name[sep_idx + 1:])
        identifier = int(self._read_line().strip()[7:])
        self._skip_line()   # mw = float(self._read_line().strip()[4:])
        precursor_mz = float(self._read_line().strip()[13:])
        self._skip_line()  # status = self._read_line().strip()
        self._skip_line()  # full_name = self._read_line().strip()
        comment = self._read_line().strip()
        is_decoy = b' Remark=DECOY_' in comment

        read_spectrum = spectrum.Spectrum(identifier, precursor_mz, precursor_charge, None, peptide, is_decoy)

        # read the peaks of the spectrum
        num_peaks = int(self._read_line().strip()[10:])
        masses = np.empty((num_peaks,), np.float32)
        intensities = np.empty((num_peaks,), np.float32)
        annotations = np.empty((num_peaks,), object)
        for i in range(num_peaks):
            peak = self._read_line().strip().split(b'\t')
            masses[i] = np.float32(peak[0])
            intensities[i] = np.float32(peak[1])
            if not _ignore_annotations:
                annotations[i] = _parse_annotation(peak[2])

        read_spectrum.set_peaks(masses, intensities, annotations)

        return read_spectrum, file_offset

    def _read_line(self):
        line = self._mm.readline()
        if not line:
            raise StopIteration
        else:
            return line

    def _skip_line(self):
        self._mm.readline()


class SplibReader(SpectraSTReader):
    """
    Read spectra from a SpectraST spectral library .splib file.
    """

    _supported_extensions = ['.splib']

    def _get_all_spectra(self):
        try:
            # read splib preamble
            # SpectraST version used to create the splib file
            version = struct.unpack('i', self._mm.read(4))[0]
            sub_version = struct.unpack('i', self._mm.read(4))[0]
            # splib information
            filename = self._mm.readline()
            num_lines = struct.unpack('i', self._mm.read(4))[0]
            for i in range(num_lines):
                self._mm.readline()

            # read all spectra
            while True:
                yield self._read_spectrum()
        except StopIteration:
            pass

    def _read_spectrum(self):
        file_offset = self._mm.tell()

        # libId (int): 4 bytes
        read_bytes = self._mm.read(4)
        if not read_bytes:  # EOF: no more spectra to be read
            raise StopIteration
        identifier = struct.unpack('i', read_bytes)[0]
        # fullName: \n terminated string
        name = self._mm.readline().strip()
        peptide = name[name.find(b'.') + 1: name.rfind(b'.')].decode(encoding='UTF-8')
        precursor_charge = int(name[name.rfind(b'/') + 1:])
        # precursor m/z (double): 8 bytes
        precursor_mz = struct.unpack('d', self._mm.read(8))[0]
        # status: \n terminated string
        status = self._mm.readline().strip()
        # numPeaks (int): 4 bytes
        num_peaks = struct.unpack('i', self._mm.read(4))[0]
        # read all peaks
        masses = np.empty((num_peaks,), np.float64)
        intensities = np.empty((num_peaks,), np.float64)
        annotations = np.empty((num_peaks,), object)
        for i in range(num_peaks):
            # m/z (double): 8 bytes
            masses[i] = struct.unpack('d', self._mm.read(8))[0]
            # intensity (double): 8 bytes
            intensities[i] = struct.unpack('d', self._mm.read(8))[0]
            # annotation: \n terminated string
            annotation_str = self._mm.readline().strip()
            if not _ignore_annotations:
                annotations[i] = _parse_annotation(annotation_str)
            # info: \n terminated string
            info = self._mm.readline()
        # comment: \n terminated string
        comment = self._mm.readline()
        is_decoy = b' Remark=DECOY_' in comment

        read_spectrum = spectrum.Spectrum(identifier, precursor_mz, precursor_charge, None, peptide, is_decoy)
        read_spectrum.set_peaks(masses, intensities, annotations)

        return read_spectrum, file_offset


class SqliteSpecReader(SpectralLibraryReader):
    """
    Read spectra from a custom SQLite spectral library file.
    """

    _max_cache_size = None

    _supported_extensions = ['.spql']

    def __enter__(self):
        self._conn = sqlite3.connect(self._filename, detect_types=sqlite3.PARSE_DECLTYPES)

        cursor = self._conn.cursor()
        cursor.execute('PRAGMA locking_mode = EXCLUSIVE')
        self._conn.commit()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._conn.close()

    def _create(self):
        """
        Create a new configuration file for the spectral library.
        """
        logging.info('Creating the spectral library configuration for file %s', self._filename)

        # collect summary information about the spectra for easy retrieval
        conn = sqlite3.connect(self._filename, detect_types=sqlite3.PARSE_DECLTYPES)
        cursor = conn.cursor()

        # retrieve from the database and convert to a consistent DataFrame
        cursor.execute('SELECT id, precursorCharge, precursorMZ FROM RefSpectra')
        ids, precursor_charges, precursor_masses = zip(*cursor.fetchall())
        conn.close()

        temp_info = collections.defaultdict(lambda: {'id': [], 'precursor_mass': []})
        for i, charge, mass in zip(ids, precursor_charges, precursor_masses):
            info_charge = temp_info[charge]
            info_charge['id'].append(i)
            info_charge['precursor_mass'].append(mass)
        self.spec_info = {charge: {'id': np.asarray(charge_info['id'], np.uint32),
                                   'precursor_mass': np.asarray(charge_info['precursor_mass'], np.float32)}
                          for charge, charge_info in six.iteritems(temp_info)}

        # store the configuration
        config_filename = self._filename + '.spcfg'
        logging.debug('Saving the spectral library configuration to file %s', config_filename)
        joblib.dump((self.spec_info, config._namespace), config_filename, compress=9, protocol=2)

        logging.info('Finished creating the spectral library configuration')

    def _get_all_spectra(self):
        """
        Generates all spectra from the spectral library file.

        Returns:
            A generator that yields all spectra from the spectral library file.
        """
        cursor = self._conn.cursor()

        # retrieve the specified spectrum
        cursor.execute('SELECT id, peptideSeq, precursorMZ, precursorCharge, isDecoy, peakMZ, peakIntensity '
                       'FROM RefSpectra, RefSpectraPeaks WHERE RefSpectra.id == RefSpectraPeaks.RefSpectraID')
        while True:
            result = cursor.fetchone()
            if result is None:
                break

            spec_id, peptide, precursor_mz, precursor_charge, is_decoy, masses, intensities = result
            read_spectrum = spectrum.Spectrum(spec_id, precursor_mz, precursor_charge, None, peptide, is_decoy == 1)
            read_spectrum.set_peaks(masses, intensities)

            yield read_spectrum, ()

    @lru_cache(maxsize=_max_cache_size)
    def get_spectrum(self, spec_id, process_peaks=False):
        """
        Read the `Spectrum` with the specified identifier from the spectral library file.

        Args:
            spec_id: The identifier of the `Spectrum` in the spectral library file.
            process_peaks: Flag whether to process the `Spectrum`'s peaks or not.

        Returns:
            The `Spectrum` from the spectral library file with the specified identifier.
        """
        cursor = self._conn.cursor()

        # retrieve the specified spectrum
        cursor.execute('SELECT peptideSeq, precursorMZ, precursorCharge, isDecoy, peakMZ, peakIntensity '
                       'FROM RefSpectra, RefSpectraPeaks '
                       'WHERE RefSpectra.id == ? AND RefSpectra.id == RefSpectraPeaks.RefSpectraID', (int(spec_id),))
        peptide, precursor_mz, precursor_charge, is_decoy, masses, intensities = cursor.fetchone()

        read_spectrum = spectrum.Spectrum(spec_id, precursor_mz, precursor_charge, None, peptide, is_decoy == 1)
        read_spectrum.set_peaks(masses, intensities)

        if process_peaks:
            read_spectrum.process_peaks()

        return read_spectrum


def read_mgf(filename):
    """
    Read all spectra from the given mgf file.

    Args:
        filename: The mgf filename from which to read the spectra.

    Returns:
        A tuple of a `Spectrum` (containing the spectrum's information), an array of masses, and an array of
        intensities.
    """
    # test if the given file is an mzML file
    verify_extension(['.mgf'], filename)

    # get all query spectra
    for mgf_spectrum in mgf.read(filename):
        # create query spectrum
        identifier = mgf_spectrum['params']['title']
        precursor_mz = float(mgf_spectrum['params']['pepmass'][0])
        retention_time = float(mgf_spectrum['params']['rtinseconds'])
        if 'charge' in mgf_spectrum['params']:
            precursor_charge = int(mgf_spectrum['params']['charge'][0])
        else:
            precursor_charge = None

        read_spectrum = spectrum.Spectrum(identifier, precursor_mz, precursor_charge, retention_time)
        read_spectrum.set_peaks(mgf_spectrum['m/z array'], mgf_spectrum['intensity array'])

        yield read_spectrum


def read_mztab_psms(filename):
    """
    Read the PSM section of the given mzTab file.

    Args:
        filename: The mzTab filename from which to read the PSMs.

    Returns:
        A Pandas `DataFrame` consisting of the PSMs.
    """
    verify_extension(['.mztab'], filename)

    skiplines = 0
    with open(filename) as f_in:
        line = next(f_in)
        while 'PSH' != line.split('\t', 1)[0]:
            line = next(f_in)
            skiplines += 1

    psms = pd.read_csv(filename, sep='\t', header=skiplines, index_col='PSM_ID')
    psms.drop('PSH', 1, inplace=True)

    psms.df_name = os.path.splitext(os.path.basename(filename))[0]

    return psms
