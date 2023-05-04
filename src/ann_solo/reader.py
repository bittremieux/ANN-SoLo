import collections
import io
import mmap
import logging
import os
import pickle
import re
from functools import lru_cache
from typing import Dict, IO, Iterator, List, Tuple, Union


import h5py
import joblib
import numpy as np
import pandas as pd
import tqdm
from lxml.etree import LxmlError
from pyteomics import mgf, mzml, mzxml
from spectrum_utils.spectrum import MsmsSpectrum
from spectrum_utils.fragment_annotation import FragmentAnnotation

from ann_solo.parsers import SplibParser
from ann_solo.spectrum import process_spectrum



class SpectralLibraryReader:
    """
    Read spectra from a spectral library file.
    """

    _supported_extensions = ['.splib','.sptxt','.mgf']

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
        self._spectral_library_store = None
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

        # Open the Spectral Library Store
        self._spectral_library_store = SpectralLibraryStore(
            self._get_store_filename())
        self._spectral_library_store.open_store('r')

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
            The spectral library file name (.hdf5 file).
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
        with self as lib_reader:
            with SpectralLibraryStore(self._get_store_filename()) as spectraStore:
                for spectrum in tqdm.tqdm(
                        lib_reader.read_library_file(),
                        desc='Library spectra read', unit='spectra'):

                    # Store the spectrum information for easy retrieval.
                    info_charge = temp_info[spectrum.precursor_charge]
                    info_charge['id'].append(spectrum.identifier)
                    info_charge['precursor_mz'].append(spectrum.precursor_mz)
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
        verify_extension(['.splib','.sptxt','.mgf'], self._filename)
        _, ext = os.path.splitext(os.path.basename(self._filename))

        if ext == '.splib':
            self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        verify_extension(['.splib', '.sptxt', '.mgf'], self._filename)
        _, ext = os.path.splitext(os.path.basename(self._filename))

        if ext == '.splib':
            self.close()

    @lru_cache(maxsize=None)
    def read_spectrum(self, spec_id: str, process_peaks: bool = False)\
            -> MsmsSpectrum:
        """
        Read the spectrum with the specified identifier from the spectral
        library store.

        Parameters
        ----------
        spec_id : string
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
        spectrum = self._spectral_library_store.read_spectrum_from_library(
                                                spec_id)
        spectrum.is_processed = False
        if process_peaks:
            annotation = spectrum.annotation
            spectrum = process_spectrum(spectrum, True)
            spectrum._annotation = annotation
        return spectrum


    def read_all_spectra(self) -> Iterator[MsmsSpectrum]:
        """
        Traverse all spectra from the spectral library store.

        Returns
        -------
        Iterator[MsmsSpectrum]
            An iterator of all spectra in the spectral library hdf5 store.
        """

        for spec_id in self._spectral_library_store.get_all_spectra_ids():
            yield self.read_spectrum(spec_id)


    def read_library_file(self) -> Iterator[MsmsSpectrum]:
        """
        Read all spectra from the splib library file.

        Returns
        -------
        Iterator[Spectrum]
            An iterator of spectra in the given library file.
        """
        verify_extension(['.splib', '.sptxt', '.mgf'], self._filename)

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
            for spectrum in self.read_sptxt():
                yield spectrum
        elif ext == '.mgf':
            for spectrum in read_mgf(self._filename):
                yield spectrum

    def get_version(self) -> str:
        """
        Gives the spectral library version.

        Returns
        -------
        str
            A string representation of the spectral library version.
        """
        return 'null'

    def _parse_fragment_annotation(self, annotation: str) -> \
            FragmentAnnotation:
        """
        Takes an ion peak anotaion line and parse to retrieve: ion_type,
        ion_index, and charge.

        Parameters
        ----------
        annotation : str
            Raw annotation line.

        Returns
        -------
        FragmentAnnotation
            An FragmentAnnotation object.
        """
        ion_type = annotation[0]
        if ion_type in 'aby':
            index_charge = annotation[1:].split('/', 1)[0].split('^')
            ion_index = re.search(r'^\d+', index_charge[0])
            if len(index_charge) == 1:
                charge, ion_index = (
                1 if ion_index.group(0) == index_charge[0] else -1,
                int(ion_index.group(0)))
            else:
                charge = re.search(r'^\d+', index_charge[1])
                charge, ion_index = int(
                    charge.group(0)) if charge else -1, int(
                    ion_index.group(0)) if ion_index else 1
            return FragmentAnnotation(str(ion_type)+str(ion_index),
                                      charge=abs(charge))
        else:
            return None

    def _peptide_to_proforma(self, peptide: str, modifications: List[str]) \
            -> str:
        """
        Takes a peptide and a list of modifications to return a modified
        peptide in its ProForma format.

        Parameters
        ----------
        peptide : str
            Peptide sequence in its non-modified format.
        modifications: List[str]
            A list of modifications.

        Returns
        -------
        str
            Modified peptide in its ProForma format.
        """
        peptide = parser.parse(peptide)
        for shift, modification in enumerate(modifications):
            idx, aa, modification_name = modification.split(',')
            peptide = peptide[:int(idx) + shift + 1] + \
                      ['['+modification_name+']'] + \
                      peptide[int(idx) + shift + 1:]
        return ''.join(peptide)

    def _parse_sptxt_spectrum(self, identifier: int, raw_spectrum: str)\
            -> MsmsSpectrum:
        """
        Takes a raw spectrum data retrieved from an sptxt file and
        parses it to a structured object of type MsmsSpectrum.

        Parameters
        ----------
        identifier : int
            Incremented identifier of the spectrum in the library.
        raw_spectrum : string
            The spectrum in a raw format.

        Returns
        -------
        MsmsSpectrum
            An MsmsSpectrum object.
        """
        # Split raw spectrum in two chunks: metadata & spectrum
        raw_spectrum_tokens = re.split('Num\s?Peaks:\s?[0-9]+\n',
                                       raw_spectrum.strip(),
                                       flags=re.IGNORECASE)
        spectrum_metadata = raw_spectrum_tokens[0]
        spectrum = raw_spectrum_tokens[1]
        # Check if decoy
        decoy = True if re.search('decoy', spectrum_metadata,
                                  re.IGNORECASE) else False
        # Retrieve peptide & charge
        peptide_Charge = spectrum_metadata.split('\n', 1)[0].split('/')
        peptide = peptide_Charge[0].split(' ')[-1].strip()
        charge = int(peptide_Charge[1].strip())
        # Retrieve precurssor mass
        precursor_mz = re.search('PrecursorMZ:\s?[0-9]+.[0-9]+', spectrum_metadata,
                          re.IGNORECASE)
        if precursor_mz:
            precursor_mz = re.search('[0-9]+.[0-9]+', precursor_mz.group(0))
        else:
            precursor_mz = re.search('Parent=\s?[0-9]+.[0-9]+', spectrum_metadata,
                              re.IGNORECASE)
            precursor_mz = re.search('[0-9]+.[0-9]+', precursor_mz.group(0))
        # Retrieve modifications
        modifications = re.search('Mods=.+?(?=[\s\n])',
                                 spectrum_metadata,
                                 re.IGNORECASE)
        if modifications:
            modifications = str(modifications.group(0)).split('/')[1:]
        else:
            modifications = None
        # Retrieve MZ & Intensities
        file = io.StringIO(spectrum)
        mz_intensity_annotation = pd.read_csv(file, sep="\t", header=None)

        if mz_intensity_annotation.shape[1] > 2:
            annotation = [self._parse_fragment_annotation(annotation)
                          for mz, annotation in
                          zip(mz_intensity_annotation[0], mz_intensity_annotation[2])]
        else:
            annotation = [None] * len(mz_intensity_annotation[0])
        spectrum = MsmsSpectrum(str(identifier), float(precursor_mz.group(0)),
                                charge,
                                mz_intensity_annotation[0].to_numpy(copy=True),
                                mz_intensity_annotation[1].to_numpy(copy=True))

        spectrum.peptide = self._peptide_to_proforma(peptide,modifications)
        spectrum.is_decoy = decoy
        spectrum._annotation = annotation

        return spectrum

    def _parse_sptxt(self) -> Iterator[Tuple[int,str]]:
        """
        Open the sptxt spectra library file and parses it
        to read all spectra.

        Returns
        -------
        Iterator[Tuple[int,str]]
            An iterator of tuples of (id, spectrum) in the given library file,
            where spectrum is in its raw text format.

        """
        with open(self._filename, 'rb') as file:
            mmapped_file = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
            for id, raw_spectrum in tqdm.tqdm(enumerate(re.finditer(
                            b'(?<![a-zA-Z])Name:\s?(?:(?!((?<![a-zA-Z])Name:\s?)).|\n)*',
                            mmapped_file.read(),
                            re.IGNORECASE),
                        1), desc='SpectraST file parse',
                        unit='spectra'):
                    yield (id,'\n'.join(raw_spectrum.group(0).decode(
                        'utf-8').splitlines()))

    def read_sptxt(self) -> Iterator[MsmsSpectrum]:
        """
        Open read spectra from SpectraST spectra library.

        Returns
        -------
        Iterator[MsmsSpectrum]
            An iterator of spectra in the given library file.

        """
        # TODO: Use all logical units in the system (-1)
        for spectrum in joblib.Parallel(n_jobs=1,
                                        backend='multiprocessing')(
                joblib.delayed(
                    self._parse_sptxt_spectrum
                )(id, raw_spectrum) for id, raw_spectrum in
                self._parse_sptxt()):
            yield spectrum



class SpectralLibraryStore:
    """
        Class to efficiently store and retrieve spectra from a library file.
    """
    def __init__(self, file_path: str) -> None:
        """
        Initialize the spectral library store.

        Parameters
        ----------
        filepath : str
            The file path of the spectral library store.

        """
        self.file_path = file_path
        self.hdf5_store = None

    def open_store(self,mode: str) -> None:
        """
        Open the hdf5 spectral library store for read/write purposes.

        Parameters
        ----------
        mode : str
            The mode (r/w) by which to open the  spectral library store.
        """
        self.hdf5_store = h5py.File(self.file_path, mode)

    def close_store(self) -> None:
        """
        Close the hdf5 spectral library store after read/write is done.
        """
        if self.hdf5_store is not None:
            self.hdf5_store.close()
            self.hdf5_store = None

    def get_all_spectra_ids(self) -> Iterator[str]:
        """
        Close the hdf5 spectral library store after write/read is done.

        Returns
        -------
        Iterator[str]
            An iterator of the keys/ids of spectra in the library store.

        """
        for spec_id in self.hdf5_store.keys():
            yield spec_id

    def write_spectrum_to_library(self, spectrum : MsmsSpectrum) -> None:
        """
        Gets an Msmsspectrum object, reads attributes, and stores it in the
        spectral library store for future retrieval.

        Parameters
        ----------
        spectrum : MsmsSpectrum
            The MsmsSpectrum object.

        """
        # Create a new group under the same key
        group = self.hdf5_store.create_group(str(spectrum.identifier))

        group.create_dataset('peptide', data=spectrum.peptide)
        group.create_dataset('precursor_charge',
                             data=spectrum.precursor_charge)
        group.create_dataset('precursor_mz', data=spectrum.precursor_mz)
        group.create_dataset('is_decoy', data=spectrum.is_decoy)
        group.create_dataset('mz', data=spectrum.mz)
        group.create_dataset('intensity', data=spectrum.intensity)
        # Encode annotation
        annotation = np.array([[_encode_ion_type(annotation.ion_type[0]),
                                int(annotation.ion_type[1:]),
                                annotation.charge]
                             if annotation is not None
                             else [0, 0, 0]
                             for annotation in spectrum.annotation])

        group.create_dataset('annotation', data=annotation)

        self.hdf5_store.flush()

    def read_spectrum_from_library(self, spec_id : str)-> MsmsSpectrum:
        """
        Gets a library spectrum id and returns the corresponding spectrum as
        an Msmsspectrum object.

        Parameters
        ----------
        spec_id : string
            A library spectrum id.

        Returns
        -------
        MsmsSpectrum
            An MsmsSpectrum object.
        """
        spectrum_specs = self.hdf5_store[str(spec_id)]
        # Decode annotation
        annotation = [FragmentAnnotation(_decode_ion_type(annotation[2])+
                                         str(annotation[1]),
                                         charge=annotation[2].astype(np.int))
                       if np.count_nonzero(annotation)
                       else None
                       for annotation in spectrum_specs['annotation'][()]]
        spectrum = MsmsSpectrum(str(spec_id),
                            spectrum_specs['precursor_mz'][()],
                            spectrum_specs['precursor_charge'][()],
                            spectrum_specs['mz'][()],
                            spectrum_specs['intensity'][()])

        spectrum.peptide = spectrum_specs['peptide'][()].decode('utf-8')
        spectrum._annotation = annotation
        spectrum.is_decoy = spectrum_specs['is_decoy'][()]

        return spectrum

    def __enter__(self) -> 'SpectralLibraryStore':
        self.open_store('w')
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close_store()

def _encode_ion_type(ion: str) -> int:
    """
    Encode the ion type from the peak's annotation in a spectrum.

    Parameters
    ----------
    ion : str
        An ion type: a, b, y or some other character.
    encoding : int
        The encoding digit of the ion type based on the map, else return 0.

    """
    ion_type_map = {'a': 1, 'b': 2, 'y': 3}
    return ion_type_map.get(ion, 0)

def _decode_ion_type(ion_encoding: int) -> str:
    """
    Decode the ion type from the peak's annotation in a spectrum.

    Parameters
    ----------
    ion : str
        An ion type: a, b, y or some other character.
    encoding : int
        The encoding digit of the ion type based on the map, else return 0.

    """
    ion_type_reverse_map = {1: 'a', 2: 'b', 3: 'y'}
    return ion_type_reverse_map.get(ion_encoding, '_')

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

    # Get all spectra.
    with open(filename, 'rb') as file:
        for i, mgf_spectrum in enumerate(mgf.MGF(file)):
            # Create spectrum.
            identifier = mgf_spectrum['params']['title' if 'title' in  mgf_spectrum['params'] else 'scan']

            precursor_mz = float(mgf_spectrum['params']['pepmass'][0])
            retention_time = float(mgf_spectrum['params']['rtinseconds']) if\
                'rtinseconds' in mgf_spectrum['params'] else None
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
            spectrum.is_decoy = True if 'decoy' in mgf_spectrum['params'] \
                else False

            if 'seq' in mgf_spectrum['params']:
                spectrum.peptide = mgf_spectrum['params']['seq']
                spectrum._annotation = [None] * len(mgf_spectrum['m/z array'])

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
