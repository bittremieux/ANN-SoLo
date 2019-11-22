import collections
import logging
import os
import pickle
from functools import lru_cache
from typing import Iterator
from typing import List
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import tqdm
from pyteomics import mgf, mzml, mzxml
from spectrum_utils.spectrum import MsmsSpectrum

from ann_solo.parsers import SplibParser
from ann_solo.spectrum import process_spectrum


logger = logging.getLogger('ann_solo')


class SpectralLibraryReader:
    """
    Read spectra from a SpectraST spectral library .splib file.
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
        do_create = False

        # Test if the given spectral library file is in a supported format.
        verify_extension(self._supported_extensions, self._filename)

        logger.debug('Load the spectral library configuration')

        # Verify that the configuration file
        # corresponding to this spectral library is present.
        config_filename = self._get_config_filename()
        if not os.path.isfile(config_filename):
            # If not we should recreate this file
            # prior to using the spectral library.
            do_create = True
            logger.warning('Missing spectral library configuration file')
        else:
            # Load the configuration file.
            config_lib_filename, self.spec_info, load_hash =\
                joblib.load(config_filename)

            # Check that the same spectral library file format is used.
            if config_lib_filename != os.path.basename(self._filename):
                do_create = True
                logger.warning('The configuration corresponds to a different '
                               'file format of this spectral library')
            # Verify that the runtime settings match the loaded settings.
            if self._config_hash != load_hash:
                do_create = True
                logger.warning('The spectral library search engine was '
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

    def _create_config(self) -> None:
        """
        Create a new configuration file for the spectral library.

        The configuration file contains for each spectrum in the spectral
        library its offset for quick random-access reading, and its precursor
        m/z for filtering using a precursor mass window. Finally, it also
        contains the settings used to construct this spectral library to make
        sure these match the runtime settings.
        """
        logger.info('Create the spectral library configuration for file %s',
                    self._filename)

        self.is_recreated = True

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
        logger.debug('Save the spectral library configuration to file %s',
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
    def get_spectrum(self, spec_id: int, process_peaks: bool = False)\
            -> MsmsSpectrum:
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
        spectrum = self._parser.read_spectrum(
            self.spec_info['offset'][spec_id])[0]
        spectrum.is_processed = False
        if process_peaks:
            process_spectrum(spectrum, True)

        return spectrum

    def get_all_spectra(self) -> Iterator[Tuple[MsmsSpectrum, int]]:
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
        self._parser.seek_first_spectrum()
        try:
            while True:
                spectrum, offset = self._parser.read_spectrum()
                spectrum.is_processed = False
                yield spectrum, offset
        except StopIteration:
            return

    def get_version(self) -> str:
        """
        Gives the spectral library version.

        Returns
        -------
        str
            A string representation of the spectral library version.
        """
        return 'null'


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
        logger.error('Unrecognized file format: %s', filename)
        raise FileNotFoundError(f'Unrecognized file format (supported file '
                                f'formats: {", ".join(supported_extensions)})')
    elif not os.path.isfile(filename):
        logger.error('File not found: %s', filename)
        raise FileNotFoundError(f'File {filename} does not exist')


def read_spectra(filename: str) -> Iterator[MsmsSpectrum]:
    """
    Read all MS/MS spectra from the given file.

    Supported file types are MGF, mzML, and mzXML.

    Parameters
    ----------
    filename : str
        The file name from which to read the spectra.

    Returns
    -------
    Iterator[Spectrum]
        An iterator of MS/MS spectra in the given file.
    """
    verify_extension(['.mgf', '.mzml', '.mzxml'], filename)
    _, ext = os.path.splitext(os.path.basename(filename.lower()))
    if ext == '.mgf':
        read_func = read_mgf
    elif ext == '.mzml':
        read_func = read_mzml
    elif ext == '.mzxml':
        read_func = read_mzxml
    else:
        raise ValueError(f'Unrecognized file format')
    yield from read_func(filename)


def read_mgf(filename: str) -> Iterator[MsmsSpectrum]:
    """
    Read all spectra from the given MGF file.

    Parameters
    ----------
    filename: str
        The MGF file name from which to read the spectra.

    Returns
    -------
    Iterator[Spectrum]
        An iterator of spectra in the given MGF file.
    """
    # Test if the given file is an MGF file.
    verify_extension(['.mgf'], filename)

    # Get all query spectra.
    for i, spectrum_dict in enumerate(mgf.read(filename)):
        # Create spectrum.
        identifier = spectrum_dict['params']['title']
        precursor_mz = float(spectrum_dict['params']['pepmass'][0])
        retention_time = float(spectrum_dict['params']['rtinseconds'])
        if 'charge' in spectrum_dict['params']:
            precursor_charge = int(spectrum_dict['params']['charge'][0])
        else:
            precursor_charge = None

        spectrum = MsmsSpectrum(identifier, precursor_mz, precursor_charge,
                                spectrum_dict['m/z array'],
                                spectrum_dict['intensity array'],
                                retention_time=retention_time)
        spectrum.index = i
        spectrum.is_processed = False

        yield spectrum


def read_mzml(filename: str) -> Iterator[MsmsSpectrum]:
    """
    Read all MS/MS spectra from the given mzML file.

    Parameters
    ----------
    filename: str
        The mzML file name from which to read the spectra.

    Returns
    -------
    Iterator[Spectrum]
        An iterator of MS/MS spectra in the given mzML file.
    """
    # Test if the given file is an mzML file.
    verify_extension(['.mzml'], filename)

    # Get all query spectra.
    for i, spectrum_dict in enumerate(mzml.read(filename)):
        if int(spectrum_dict.get('ms level', -1)) == 2:
            # Create spectrum.
            identifier = spectrum_dict['id']
            precursor = spectrum_dict['precursorList']['precursor'][0]
            precursor_ion = precursor['selectedIonList']['selectedIon'][0]
            precursor_mz = precursor_ion['selected ion m/z']
            retention_time = (spectrum_dict['scanList']['scan'][0]
                              ['scan start time'])
            if 'charge state' in precursor_ion:
                precursor_charge = int(precursor_ion['charge state'])
            elif 'possible charge state' in precursor_ion:
                precursor_charge = int(precursor_ion['possible charge state'])
            else:
                precursor_charge = None

            spectrum = MsmsSpectrum(identifier, precursor_mz, precursor_charge,
                                    spectrum_dict['m/z array'],
                                    spectrum_dict['intensity array'],
                                    retention_time=retention_time)
            spectrum.index = i
            spectrum.is_processed = False

            yield spectrum


def read_mzxml(filename: str) -> Iterator[MsmsSpectrum]:
    """
    Read all MS/MS spectra from the given mzXML file.

    Parameters
    ----------
    filename: str
        The mzXML file name from which to read the spectra.

    Returns
    -------
    Iterator[Spectrum]
        An iterator of MS/MS spectra in the given mzXML file.
    """
    # Test if the given file is an mzXML file.
    verify_extension(['.mzxml'], filename)

    # Get all query spectra.
    for i, spectrum_dict in enumerate(mzxml.read(filename)):
        if int(spectrum_dict.get('msLevel', -1)) == 2:
            # Create spectrum.
            identifier = spectrum_dict['id']
            precursor_mz = spectrum_dict['precursorMz'][0]['precursorMz']
            retention_time = spectrum_dict['retentionTime']
            if 'precursorCharge' in spectrum_dict['precursorMz'][0]:
                precursor_charge = (spectrum_dict['precursorMz'][0]
                                    ['precursorCharge'])
            else:
                precursor_charge = None

            spectrum = MsmsSpectrum(identifier, precursor_mz, precursor_charge,
                                    spectrum_dict['m/z array'],
                                    spectrum_dict['intensity array'],
                                    retention_time=retention_time)
            spectrum.index = i
            spectrum.is_processed = False

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
