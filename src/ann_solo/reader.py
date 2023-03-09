import collections
import logging
import os
import pickle
from functools import lru_cache
from typing import List, Tuple, Dict, IO, Iterator, Union


import h5py
import joblib
import numpy as np
import pandas as pd
import tqdm
from lxml.etree import LxmlError
from pyteomics import mgf, mzml, mzxml
from spectrum_utils.spectrum import MsmsSpectrum

from ann_solo.parsers import SplibParser
from ann_solo.spectrum import process_spectrum
from ann_solo.config import config



class SpectralLibraryReader:
    """
    Read spectra from a SpectraST spectral library file.
    """

    _supported_extensions = ['.splib']

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
        self._parser = None
        self._spectra_library_store = None
        do_create = False

        # Test if the given spectral library file is in a supported format.
        verify_extension(self._supported_extensions, self._filename)

        logging.debug('Load the spectral library configuration')

        # Verify that the configuration file
        # corresponding to this spectral library is present.
        config_filename = self._get_config_filename()
        store_filename = self._get_store_filename()

        if not os.path.isfile(config_filename) or not os.path.isfile(store_filename):
            # If not we should recreate this file
            # prior to using the spectral library.
            do_create = True
            logging.warning('Missing spectral library store or configuration '
                            'file')
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

        # Open the Spectra Library Store
        self._spectra_library_store = SpectraLibraryStore(self._get_store_filename())
        self._spectra_library_store.open_store_to_read()

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

    def _get_store_filename(self) -> str:
        """
        Gets the spectra library store file name for the spectral library
        with the current configuration.

        Returns
        -------
        str
            The configuration file name (.spcfg file).
        """
        if self._config_hash is not None:
            return (f'{os.path.splitext(self._filename)[0]}_'
                    f'{self._config_hash[:7]}.hdf5')
        else:
            return f'{os.path.splitext(self._filename)[0]}.hdf5'

    def _create_config(self) -> None:
        """
        Create a new configuration file for the spectral library.

        The configuration file contains for each spectrum in the spectral
        library its offset for quick random-access reading, and its precursor
        m/z for filtering using a precursor mass window. Finally, it also
        contains the settings used to construct this spectral library to make
        sure these match the runtime settings.
        """
        logging.info('Create the spectral library configuration for file %s',
                     self._filename)

        self.is_recreated = True

        # Read all the spectra in the spectral library.
        temp_info = collections.defaultdict(
            lambda: {'id': [], 'precursor_mz': []})
        #offsets = {}
        with self as lib_reader:
            with SpectraLibraryStore(self._get_store_filename()) as spectraStore:
                for spectrum in tqdm.tqdm(
                        lib_reader.read_library_file(),
                        desc='Library spectra read', unit='spectra'):
                    # Store the spectrum information for easy retrieval.
                    info_charge = temp_info[spectrum.precursor_charge]
                    info_charge['id'].append(spectrum.identifier)
                    info_charge['precursor_mz'].append(spectrum.precursor_mz)
                    #offsets[spectrum.identifier] = offset
                    spectraStore.write_spectrum_to_library(spectrum)
        self.spec_info = {
            'charge': {
                charge: {
                    'id': np.asarray(charge_info['id'], np.uint32),
                    'precursor_mz': np.asarray(charge_info['precursor_mz'],
                                               np.float32)
                } for charge, charge_info in temp_info.items()}
        }

        # Store the configuration.
        config_filename = self._get_config_filename()
        logging.debug('Save the spectral library configuration to file %s',
                      config_filename)
        joblib.dump(
            (os.path.basename(self._filename), self.spec_info,
             self._config_hash),
            config_filename, compress=9, protocol=pickle.DEFAULT_PROTOCOL)

    def open(self) -> None:
        self._parser = SplibParser(self._filename.encode())

    def close(self) -> None:
        if self._parser is not None:
            del self._parser

    def __enter__(self) -> 'SpectralLibraryReader':
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @lru_cache(maxsize=None)
    def read_spectrum(self, spec_id: int, process_peaks: bool = False)\
            -> MsmsSpectrum:
        """
        Read the spectrum with the specified identifier from the spectral
        library store.

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
            The spectrum from the spectral library store with the specified
            identifier.
        """
        spectrum = self._spectra_library_store.read_spectrum_from_library(
                                                spec_id)
        spectrum.is_processed = False
        if process_peaks:
            return process_spectrum(spectrum, True)

        return spectrum


    def read_all_spectra(self) -> Iterator[Tuple[MsmsSpectrum, int]]:
        """
        Traverse all spectra from the spectral library store.

        Returns
        -------
        Iterator[Tuple[Spectrum, int]]
            An iterator of all spectra in the spectral library hdf5 store.
        """

        for spec_id in self._spectra_library_store.get_all_spectra_ids():
            yield self.read_spectrum(spec_id)

    def read_library_file(self) -> Iterator[MsmsSpectrum]:
        """
        Read all spectra from the splib library file.

        Returns
        -------
        Iterator[Spectrum]
            An iterator of spectra in the given library file.
        """
        verify_extension(['.splib'],self._filename)

        _, ext = os.path.splitext(os.path.basename(self._filename))

        if ext == '.splib':
            self._parser.seek_first_spectrum()
            try:
                while True:
                    spectrum, _ = self._parser.read_spectrum()
                    spectrum.is_processed = False
                    yield spectrum
            except StopIteration:
                return
        elif ext == '.sptxt':
            return None
        elif ext == '.mgf':
            return None

    def get_version(self) -> str:
        """
        Gives the spectral library version.

        Returns
        -------
        str
            A string representation of the spectral library version.
        """
        return 'null'



class SpectraLibraryStore:
    """
        Class for efficient store retrieved spectra library file.
    """
    def __init__(self, file_path) -> None:
        self.file_path = file_path
        self.hdf5_store = None

    def open_store_to_read(self) -> None:
        self.hdf5_store = h5py.File(self.file_path, 'r')

    def open_store_to_write(self) -> None:
        self.hdf5_store = h5py.File(self.file_path, 'w')

    def close_store(self) -> None:
        if self.hdf5_store is not None:
            self.hdf5_store.close()
            self.hdf5_store = None

    def get_all_spectra_ids(self):
        for spec_id in self.hdf5_store.keys():
            yield spec_id

    def write_spectrum_to_library(self, spectrum):
        # Create a new group under the same key
        group = self.hdf5_store.create_group(f'{spectrum.identifier}')

        group.create_dataset('peptide', data=spectrum.peptide)
        group.create_dataset('precursor_charge',
                             data=spectrum.precursor_charge)
        group.create_dataset('precursor_mz', data=spectrum.precursor_mz)
        group.create_dataset('is_decoy', data=spectrum.is_decoy)
        group.create_dataset('mz', data=spectrum.mz)
        group.create_dataset('intensity', data=spectrum.intensity)

        self.hdf5_store.flush()

    def read_spectrum_from_library(self, spec_id):
        spectrum_specs = self.hdf5_store[f'{spec_id}']

        spectrum = MsmsSpectrum(spec_id,
                            spectrum_specs['precursor_mz'][()],
                            spectrum_specs['precursor_charge'][()],
                            spectrum_specs['mz'][()],
                            spectrum_specs['intensity'][()],
                            is_decoy=spectrum_specs['is_decoy'][()])
        spectrum.peptide = spectrum_specs['peptide'][()].decode('utf-8')

        return spectrum

    def __enter__(self) -> 'SpectraLibraryStore':
        self.open_store_to_write()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close_store()


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
    _, ext = os.path.splitext(os.path.basename(filename))
    if ext.lower() not in supported_extensions:
        logging.error('Unrecognized file format: %s', filename)
        raise FileNotFoundError(f'Unrecognized file format (supported file '
                                f'formats: {", ".join(supported_extensions)})')
    elif not os.path.isfile(filename):
        logging.error('File not found: %s', filename)
        raise FileNotFoundError(f'File {filename} does not exist')




def read_mzml(source: Union[IO, str]) -> Iterator[MsmsSpectrum]:
    """
    Get the MS/MS spectra from the given mzML file.

    Parameters
    ----------
    source : Union[IO, str]
        The mzML source (file name or open file object) from which the spectra
        are read.

    Returns
    -------
    Iterator[MsmsSpectrum]
        An iterator over the requested spectra in the given file.
    """
    with mzml.MzML(source) as f_in:
        try:
            for i, spectrum in enumerate(f_in):
                if int(spectrum.get('ms level', -1)) == 2:
                    try:
                        parsed_spectrum = _parse_spectrum_mzml(spectrum)
                        parsed_spectrum.index = i
                        parsed_spectrum.is_processed = False
                        yield parsed_spectrum
                    except ValueError as e:
                        logger.warning(f'Failed to read spectrum %s: %s',
                                       spectrum['id'], e)
        except LxmlError as e:
            logger.warning('Failed to read file %s: %s', source, e)


def _parse_spectrum_mzml(spectrum_dict: Dict) -> MsmsSpectrum:
    """
    Parse the Pyteomics spectrum dict.

    Parameters
    ----------
    spectrum_dict : Dict
        The Pyteomics spectrum dict to be parsed.

    Returns
    -------
    MsmsSpectrum
        The parsed spectrum.

    Raises
    ------
    ValueError: The spectrum can't be parsed correctly:
        - Unknown scan number.
        - Not an MS/MS spectrum.
        - Unknown precursor charge.
    """

    spectrum_id = spectrum_dict['id']

    if 'scan=' in spectrum_id:
        scan_nr = int(spectrum_id[spectrum_id.find('scan=') + len('scan='):])
    elif 'index=' in spectrum_id:
        scan_nr = int(spectrum_id[spectrum_id.find('index=') + len('index='):])
    else:
        raise ValueError(f'Failed to parse scan/index number')

    if int(spectrum_dict.get('ms level', -1)) != 2:
        raise ValueError(f'Unsupported MS level {spectrum_dict["ms level"]}')


    mz_array = spectrum_dict['m/z array']
    intensity_array = spectrum_dict['intensity array']
    retention_time = spectrum_dict['scanList']['scan'][0]['scan start time']

    precursor = spectrum_dict['precursorList']['precursor'][0]
    precursor_ion = precursor['selectedIonList']['selectedIon'][0]
    precursor_mz = precursor_ion['selected ion m/z']
    if 'charge state' in precursor_ion:
        precursor_charge = int(precursor_ion['charge state'])
    elif 'possible charge state' in precursor_ion:
        precursor_charge = int(precursor_ion['possible charge state'])
    else:
        precursor_charge = None
    spectrum = MsmsSpectrum(str(scan_nr), precursor_mz, precursor_charge,
                            mz_array, intensity_array, None, retention_time)

    return spectrum

def read_mzxml(source: Union[IO, str]) -> Iterator[MsmsSpectrum]:
    """
    Get the MS/MS spectra from the given mzXML file.

    Parameters
    ----------
    source : Union[IO, str]
        The mzXML source (file name or open file object) from which the spectra
        are read.

    Returns
    -------
    Iterator[MsmsSpectrum]
        An iterator over the requested spectra in the given file.
    """
    with mzxml.MzXML(source) as f_in:
        try:
            for i, spectrum in enumerate(f_in):
                if int(spectrum.get('msLevel', -1)) == 2:
                    try:
                        parsed_spectrum = _parse_spectrum_mzxml(spectrum)
                        parsed_spectrum.index = i
                        parsed_spectrum.is_processed = False
                        yield parsed_spectrum
                    except ValueError as e:
                        logger.warning(f'Failed to read spectrum %s: %s',
                                       spectrum['id'], e)
        except LxmlError as e:
            logger.warning('Failed to read file %s: %s', source, e)


def _parse_spectrum_mzxml(spectrum_dict: Dict) -> MsmsSpectrum:
    """
    Parse the Pyteomics spectrum dict.

    Parameters
    ----------
    spectrum_dict : Dict
        The Pyteomics spectrum dict to be parsed.

    Returns
    -------
    MsmsSpectrum
        The parsed spectrum.

    Raises
    ------
    ValueError: The spectrum can't be parsed correctly:
        - Not an MS/MS spectrum.
        - Unknown precursor charge.
    """
    scan_nr = int(spectrum_dict['id'])

    if int(spectrum_dict.get('msLevel', -1)) != 2:
        raise ValueError(f'Unsupported MS level {spectrum_dict["msLevel"]}')

    mz_array = spectrum_dict['m/z array']
    intensity_array = spectrum_dict['intensity array']
    retention_time = spectrum_dict['retentionTime']

    precursor_mz = spectrum_dict['precursorMz'][0]['precursorMz']
    if 'precursorCharge' in spectrum_dict['precursorMz'][0]:
        precursor_charge = spectrum_dict['precursorMz'][0]['precursorCharge']
    else:
        precursor_charge = None

    spectrum = MsmsSpectrum(str(scan_nr), precursor_mz, precursor_charge,
                            mz_array, intensity_array, None, retention_time)

    return spectrum

def read_mgf(filename: str) -> Iterator[MsmsSpectrum]:
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

    # Get all query spectra.
    for i, mgf_spectrum in enumerate(mgf.read(filename)):
        # Create spectrum.
        identifier = mgf_spectrum['params']['title']
        precursor_mz = float(mgf_spectrum['params']['pepmass'][0])
        retention_time = float(mgf_spectrum['params']['rtinseconds'])
        if 'charge' in mgf_spectrum['params']:
            precursor_charge = int(mgf_spectrum['params']['charge'][0])
        else:
            precursor_charge = None

        spectrum = MsmsSpectrum(identifier, precursor_mz, precursor_charge,
                                mgf_spectrum['m/z array'],
                                mgf_spectrum['intensity array'],
                                retention_time=retention_time)
        spectrum.index = i
        spectrum.is_processed = False

        yield spectrum


def read_query_file(filename: str) -> Iterator[MsmsSpectrum]:
    """
    Read all spectra from the given mgf, mzml, or mzxml file.

    Parameters
    ----------
    filename: str
        The peak file name from which to read the spectra.

    Returns
    -------
    Iterator[Spectrum]
        An iterator of spectra in the given mgf file.
    """
    verify_extension(['.mgf', '.mzml', '.mzxml'],
                     filename)

    _, ext = os.path.splitext(os.path.basename(filename))

    if ext == '.mgf':
        return read_mgf(filename)
    elif ext == '.mzml':
        return read_mzml(filename)
    elif ext == '.mzxml':
        return read_mzxml(filename)

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
