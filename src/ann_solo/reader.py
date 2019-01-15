import abc
import collections
import logging
import mmap
import os
import pickle
import struct
from functools import lru_cache
from typing import Iterator
from typing import List
from typing import Tuple
from typing import Union

import joblib
import numpy as np
import pandas as pd
import tqdm
from pyteomics import mgf

from ann_solo.spectrum import Spectrum


class SpectralLibraryReader(metaclass=abc.ABCMeta):
    """
    Read spectra from a spectral library file.
    """

    _max_cache_size = None

    _supported_extensions = []

    is_recreated = False

    def __init__(self, filename: str, config_hash: str = None) -> None:
        """
        Initialize the spectral library reader. Metadata for future easy access
        of the individual spectra is read from the corresponding configuration
        file.

        The configuration file contains minimally for each spectrum in the
        spectral library its precursor charge and precursor mass to quickly
        filter the spectra library. Furthermore, it also contains the settings
        used to construct this spectral library to make sure these match the
        runtime settings.

        Parameters
        ----------
        filename : str
            The file name of the spectral library.
        config_hash : str, optional
            The hash representing the current spectral library configuration.

        Raises
        ------
        FileNotFoundError
            The given spectral library file wasn't found.
        ValueError
            The configuration file wasn't found or its settings don't
            correspond to the runtime settings.
        """
        self._filename = filename
        self._config_hash = config_hash
        do_create = False

        # Test if the given spectral library file is in a supported format.
        verify_extension(self._supported_extensions, self._filename)

        logging.debug('Load the spectral library configuration')

        # Verify that the configuration file
        # corresponding to this spectral library is present.
        config_filename = self._get_config_filename()
        if not os.path.isfile(config_filename):
            # If not we should recreate this file
            # prior to using the spectral library.
            do_create = True
            logging.warning('Missing spectral library configuration file')
        else:
            # Load the configuration file.
            config_lib_filename, self.spec_info, load_hash =\
                joblib.load(config_filename)

            # Check that the same spectral library file format is used.
            if config_lib_filename != os.path.basename(self._filename):
                do_create = True
                logging.warning('The configuration corresponds to a different '
                                'file format of this spectral library')
            # Verify that the runtime settings match the loaded settings.
            if self._config_hash != load_hash:
                do_create = True
                logging.warning('The spectral library search engine was '
                                'created using non-compatible settings')

        # (Re)create the spectral library configuration
        # if it is missing or invalid.
        if do_create:
            self._create_config()

    def _get_config_filename(self) -> str:
        """
        Gets the configuration file name for the spectral library with the
        current configuration.

        Returns
        -------
        str
            The configuration file name (.spcfg file).
        """
        if self._config_hash is not None:
            return (f'{os.path.splitext(self._filename)[0]}_'
                    f'{self._config_hash[:7]}.spcfg')
        else:
            return f'{os.path.splitext(self._filename)[0]}.spcfg'

    @abc.abstractmethod
    def _create_config(self) -> None:
        """
        Create a new configuration file for the spectral library.

        The configuration file contains for each spectrum in the spectral
        library its offset for quick random-access reading, and its precursor
        m/z for filtering using a precursor mass window. Finally, it also
        contains the settings used to construct this spectral library to make
        sure these match the runtime settings.
        """
        self.is_recreated = True

    @abc.abstractmethod
    def open(self) -> None:
        pass

    @abc.abstractmethod
    def close(self) -> None:
        pass

    @abc.abstractmethod
    def __enter__(self) -> 'SpectralLibraryReader':
        return self

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass

    @abc.abstractmethod
    def get_spectrum(self, spec_id: int, process_peaks: bool = False)\
            -> Spectrum:
        """
        Read the spectrum with the specified identifier from the spectral
        library file.

        Parameters
        ----------
        spec_id : int
            The identifier of the spectrum in the spectral library file.
        process_peaks : bool, optional
            Flag whether to process the spectrum's peaks or not
            (the default is false to not process the spectrum's peaks).

        Returns
        -------
        Spectrum
            The spectrum from the spectral library file with the specified
            identifier.
        """
        pass

    @abc.abstractmethod
    def get_all_spectra(self) -> Iterator[Tuple[Spectrum, int]]:
        """
        Generates all spectra from the spectral library file.

        For each individual spectrum a tuple consisting of the spectrum and
        some additional information as a nested tuple (containing on the type
        of spectral library file) are returned.

        Returns
        -------
        Iterator[Tuple[Spectrum, int]]
            An iterator of all spectra along with their offset in the spectral
            library file.
        """
        pass

    def get_version(self) -> str:
        """
        Gives the spectral library version.

        Returns
        -------
        str
            A string representation of the spectral library version.
        """
        return 'null'


class SpectraSTReader(SpectralLibraryReader, metaclass=abc.ABCMeta):
    """
    Read spectra from a SpectraST spectral library file.
    """

    _max_cache_size = None

    _supported_extensions = []

    def __init__(self, filename: str, config_hash: str = None) -> None:
        super().__init__(filename, config_hash)

        self._file = None
        self._mm = None

    def open(self) -> None:
        self._file = open(self._filename, 'rb')
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def close(self) -> None:
        if self._mm is not None:
            self._mm.close()
        if self._file is not None:
            self._file.close()

    def __enter__(self) -> SpectralLibraryReader:
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _create_config(self) -> None:
        """
        Create a new configuration file for the spectral library.

        The configuration file contains for each spectrum in the spectral
        library its offset for quick random-access reading, and its precursor
        m/z for filtering using a precursor mass window. Finally, it also
        contains the settings used to construct this spectral library to make
        sure these match the runtime settings.
        """
        super()._create_config()

        logging.info('Create the spectral library configuration for file %s',
                     self._filename)

        # Read all the spectra in the spectral library.
        temp_info = collections.defaultdict(
            lambda: {'id': [], 'precursor_mz': []})
        offsets = {}
        with self as lib_reader:
            for spectrum, offset in tqdm.tqdm(lib_reader.get_all_spectra(),
                                              desc='Library spectra read',
                                              unit='spectra'):
                # Store the spectrum information for easy retrieval.
                info_charge = temp_info[spectrum.precursor_charge]
                info_charge['id'].append(spectrum.identifier)
                info_charge['precursor_mz'].append(spectrum.precursor_mz)
                offsets[spectrum.identifier] = offset
        self.spec_info = {
            'charge': {
                charge: {
                    'id': np.asarray(charge_info['id'], np.uint32),
                    'precursor_mz': np.asarray(charge_info['precursor_mz'],
                                               np.float32)
                } for charge, charge_info in temp_info.items()},
            'offset': offsets}

        # Store the configuration.
        config_filename = self._get_config_filename()
        logging.debug('Save the spectral library configuration to file %s',
                      config_filename)
        joblib.dump(
            (os.path.basename(self._filename), self.spec_info,
             self._config_hash),
            config_filename, compress=9, protocol=pickle.DEFAULT_PROTOCOL)

    @lru_cache(maxsize=_max_cache_size)
    def get_spectrum(self, spec_id: int, process_peaks: bool = False)\
            -> Spectrum:
        """
        Read the spectrum with the specified identifier from the spectral
        library file.

        Parameters
        ----------
        spec_id : int
            The identifier of the spectrum in the spectral library file.
        process_peaks : bool, optional
            Flag whether to process the spectrum's peaks or not
            (the default is false to not process the spectrum's peaks).

        Returns
        -------
        Spectrum
            The spectrum from the spectral library file with the specified
            identifier.
        """
        self._mm.seek(self.spec_info['offset'][spec_id])

        spectrum = self._read_spectrum()[0]
        if process_peaks:
            spectrum.process_peaks()

        return spectrum

    @abc.abstractmethod
    def _read_spectrum(self) -> Tuple[Spectrum, int]:
        pass


class SptxtReader(SpectraSTReader):
    """
    Read spectra from a SpectraST spectral library .sptxt file.
    """

    _supported_extensions = ['.sptxt']

    def get_all_spectra(self) -> Iterator[Tuple[Spectrum, int]]:
        """
        Generates all spectra from the spectral library file.

        For each individual spectrum a tuple consisting of the spectrum and
        some additional information as a nested tuple (containing on the type
        of spectral library file) are returned.

        Returns
        -------
        Iterator[Tuple[Spectrum, int]]
            An iterator of all spectra along with their offset in the spectral
            library file.
        """
        self._mm.seek(0)
        try:
            while True:
                yield self._read_spectrum()
        except StopIteration:
            return

    def _read_spectrum(self) -> Tuple[Spectrum, int]:
        # Find the next spectrum in the file.
        file_offset = self._mm.tell()
        line = self._read_line()
        while b'Name: ' not in line:
            file_offset = self._mm.tell()
            line = self._read_line()

        # Read the spectrum.
        # Identification information.
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

        spectrum = Spectrum(identifier, precursor_mz, precursor_charge,
                            None, peptide, is_decoy)

        # Read the peaks of the spectrum.
        num_peaks = int(self._read_line().strip()[10:])
        mz = np.empty((num_peaks,), np.float32)
        intensity = np.empty((num_peaks,), np.float32)
        annotation = np.empty((num_peaks,), object)
        for i in range(num_peaks):
            peak = self._read_line().strip().split(b'\t')
            mz[i] = np.float32(peak[0])
            intensity[i] = np.float32(peak[1])
            if not _ignore_annotations:
                annotation[i] = _parse_annotation(peak[2])
        spectrum.set_peaks(mz, intensity, annotation)

        return spectrum, file_offset

    def _read_line(self) -> bytes:
        """
        Read the next line from the spectral library file.

        Returns
        -------
        bytes
            The next line in the spectral library file.

        Raises
        ------
        StopIteration
            If we are at the end of the file.
        """
        line = self._mm.readline()
        if line is not None:
            return line
        else:
            raise StopIteration

    def _skip_line(self) -> None:
        """
        Skip the next line in the spectral library file.
        """
        self._mm.readline()


class SplibReader(SpectraSTReader):
    """
    Read spectra from a SpectraST spectral library .splib file.
    """

    _supported_extensions = ['.splib']

    def get_all_spectra(self) -> Iterator[Tuple[Spectrum, int]]:
        """
        Generates all spectra from the spectral library file.

        For each individual spectrum a tuple consisting of the spectrum and
        some additional information as a nested tuple (containing on the type
        of spectral library file) are returned.

        Returns
        -------
        Iterator[Tuple[Spectrum, int]]
            An iterator of all spectra along with their offset in the spectral
            library file.
        """
        self._mm.seek(0)
        try:
            # Read splib preamble.
            # SpectraST version used to create the splib file.
            version = struct.unpack('i', self._mm.read(4))[0]
            sub_version = struct.unpack('i', self._mm.read(4))[0]
            # splib information.
            filename = self._mm.readline()
            num_lines = struct.unpack('i', self._mm.read(4))[0]
            for _ in range(num_lines):
                self._mm.readline()

            # Read all spectra.
            while True:
                yield self._read_spectrum()
        except StopIteration:
            return

    def _read_spectrum(self) -> Tuple[Spectrum, int]:
        file_offset = self._mm.tell()

        # libId (int): 4 bytes
        read_bytes = self._mm.read(4)
        if not read_bytes:  # EOF: no more spectra to be read
            raise StopIteration
        identifier = struct.unpack('i', read_bytes)[0]
        # fullName: \n terminated string
        name = self._mm.readline().strip()
        peptide = name[name.find(b'.') + 1: name.rfind(b'.')].decode(
            encoding='UTF-8')
        precursor_charge =\
            int(name[name.rfind(b'/') + 1: name.rfind(b'/') + 2])
        # precursor m/z (double): 8 bytes
        precursor_mz = struct.unpack('d', self._mm.read(8))[0]
        # status: \n terminated string
        status = self._mm.readline().strip()
        # numPeaks (int): 4 bytes
        num_peaks = struct.unpack('i', self._mm.read(4))[0]
        # Read all peaks.
        mz = np.empty((num_peaks,), np.float32)
        intensity = np.empty((num_peaks,), np.float32)
        annotation = np.empty((num_peaks,), object)
        for i in range(num_peaks):
            # m/z (double): 8 bytes
            mz[i] = np.float32(struct.unpack('d', self._mm.read(8))[0])
            # intensity (double): 8 bytes
            intensity[i] = np.float32(struct.unpack('d', self._mm.read(8))[0])
            # annotation: \n terminated string
            annotation_str = self._mm.readline().strip()
            if not _ignore_annotations:
                annotation[i] = _parse_annotation(annotation_str)
            # info: \n terminated string
            info = self._mm.readline()
        # comment: \n terminated string
        comment = self._mm.readline()
        is_decoy = b' Remark=DECOY_' in comment

        spectrum = Spectrum(identifier, precursor_mz, precursor_charge, None,
                            peptide, is_decoy)
        spectrum.set_peaks(mz, intensity, annotation)

        return spectrum, file_offset


def get_spectral_library_reader(filename: str, config_hash: str = None)\
        -> SpectralLibraryReader:
    """
    Get a spectral library reader instance based on the given spectral library
    file and ANN configuration code.

    If both an sptxt and splib SpectraST spectral library are available the
    splib file is preferred.

    Parameters
    ----------
    filename : str
        The spectral library file name.
    config_hash : str, optional
        The ANN configuration code pertaining to specific ANN hyperparameters.

    Returns
    -------
    SpectralLibraryReader
        A reader suitable for the given spectral library.

    Raises
    ------
    FileNotFoundError
        - If the file name does not exist.
        - If the file name does not have one of the supported extensions.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f'Spectral library file {filename} not found')
    verify_extension(['.splib', '.sptxt'], filename)

    base_filename, ext = os.path.splitext(filename)
    splib_exists = os.path.isfile(base_filename + '.splib')
    sptxt_exists = os.path.isfile(base_filename + '.sptxt')
    if splib_exists:
        # Prefer an splib file because it is faster to read.
        return SplibReader(base_filename + '.splib', config_hash)
    elif sptxt_exists:
        # Fall back to an sptxt file.
        return SptxtReader(base_filename + '.sptxt', config_hash)


def verify_extension(supported_extensions: List[str], filename: str) -> None:
    """
    Check that the given file name has a supported extension.

    Parameters
    ----------
    supported_extensions : List[str]
        A list of supported file extensions.
    filename : str
        The file name to be checked.

    Raises
    ------
    FileNotFoundError
        If the file name does not have one of the supported extensions.
    """
    base_name, ext = os.path.splitext(os.path.basename(filename))
    if ext.lower() not in supported_extensions:
        logging.error('Unrecognized file format: %s', filename)
        raise FileNotFoundError(f'Unrecognized file format (supported file '
                                f'formats: {", ".join(supported_extensions)})')


# Possible peak annotations that will be parsed.
_annotation_ion_types = frozenset(b'abcxyz')
# Whether or not to parse the peak annotations.
_ignore_annotations = False


# TODO: Make a PeakParser class out of this?
def _parse_annotation(raw: bytes) -> Union[Tuple[str, int], None]:
    # Discard peaks that don't correspond to a supported ion type.
    if raw[0] in _annotation_ion_types:
        # Take the first possible annotation.
        first_annotation = raw.split(b',', 1)[0]
        # Discard isotope peaks.
        if b'i' not in first_annotation:
            ion_sep = first_annotation.find(b'/')
            if ion_sep == -1:
                ion_sep = len(first_annotation)
            first_annotation_substring = first_annotation[:ion_sep]
            has_mod = (b'-' in first_annotation_substring or
                       b'+' in first_annotation_substring)
            # Discard modified peaks.
            if not has_mod:
                charge_sep = first_annotation.find(b'^')
                if charge_sep != -1:
                    ion_type = first_annotation[:charge_sep].decode('UTF-8')
                    charge = int(first_annotation[charge_sep + 1: ion_sep])
                else:
                    ion_type = first_annotation[:ion_sep].decode('UTF-8')
                    charge = 1

                return ion_type, charge

    return None


def read_mgf(filename: str) -> Iterator[Spectrum]:
    """
    Read all spectra from the given mgf file.

    Parameters
    ----------
    filename: str
        The mgf file name from which to read the spectra.

    Returns
    -------
    Iterator[Spectrum]
        An iterator of spectra in the given mgf file.
    """
    # Test if the given file is an mzML file.
    verify_extension(['.mgf'], filename)

    # Get all query spectra.
    for mgf_spectrum in mgf.read(filename):
        # Create spectrum.
        identifier = mgf_spectrum['params']['title']
        precursor_mz = float(mgf_spectrum['params']['pepmass'][0])
        retention_time = float(mgf_spectrum['params']['rtinseconds'])
        if 'charge' in mgf_spectrum['params']:
            precursor_charge = int(mgf_spectrum['params']['charge'][0])
        else:
            precursor_charge = None

        spectrum = Spectrum(identifier, precursor_mz, precursor_charge,
                            retention_time)
        spectrum.set_peaks(mgf_spectrum['m/z array'],
                           mgf_spectrum['intensity array'])

        yield spectrum


def read_mztab_ssms(filename: str) -> pd.DataFrame:
    """
    Read SSMs from the given mzTab file.

    Parameters
    ----------
    filename: str
        The mzTab file name from which to read the SSMs.

    Returns
    -------
    pd.DataFrame
        A data frame containing the SSM information from the mzTab file.
    """
    verify_extension(['.mztab'], filename)

    # Skip the header lines.
    skiplines = 0
    with open(filename) as f_in:
        line = next(f_in)
        while line.split('\t', 1)[0] != 'PSH':
            line = next(f_in)
            skiplines += 1

    ssms = pd.read_csv(filename, sep='\t', header=skiplines,
                       index_col='PSM_ID')
    ssms.drop('PSH', 1, inplace=True)

    ssms['opt_ms_run[1]_cv_MS:1002217_decoy_peptide'] =\
        ssms['opt_ms_run[1]_cv_MS:1002217_decoy_peptide'].astype(bool)

    ssms.df_name = os.path.splitext(os.path.basename(filename))[0]

    return ssms
