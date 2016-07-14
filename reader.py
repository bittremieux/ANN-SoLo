import abc
import logging
import mmap
import os
import six
import struct
try:
    from functools import lru_cache
except ImportError:
    from functools32 import lru_cache

import numpy as np
import pandas as pd
from pyteomics import mgf

import spectrum


def get_spectral_library_reader(filename):
    base_filename, _ = os.path.splitext(filename)
    splib_exists = os.path.isfile(base_filename + '.splib')
    sptxt_exists = os.path.isfile(base_filename + '.sptxt')
    if splib_exists:
        # prefer an splib file because it is faster to read
        return SplibReader(base_filename + '.splib')
    elif sptxt_exists:
        # fall back to an sptxt file
        return SptxtReader(base_filename + '.sptxt')
    else:
        raise FileNotFoundError('No spectral library file found (required file format: splib or sptxt)')


def verify_extension(supported_extensions, filename):
    base_name, ext = os.path.splitext(os.path.basename(filename))
    if ext.lower() not in supported_extensions:
        logging.error('Unsupported file: {}'.format(filename))
        raise ValueError('Unsupported file: {}'.format(filename))


_annotation_ion_types = frozenset(b'abcxyz')


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


@six.add_metaclass(abc.ABCMeta)
class SpectralLibraryReader(object):
    """
    Read spectra from a spectral library file.

    This abstract class provides some shared functionality to read spectra using a context manager.
    """

    _max_cache_size = None

    _supported_extensions = []

    def __init__(self, filename):
        # test if the given file is in a supported spectral library format
        verify_extension(self._supported_extensions, filename)

        self._filename = filename

    def __enter__(self):
        self._file = open(self._filename, 'rb')
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._mm.close()
        self._file.close()

    @abc.abstractmethod
    def get_all_spectra(self):
        """
        Read all spectra in the spectral library file.

        Returns:
            All spectra in the spectral library file as a tuple of the `Spectrum` and its offset in the file.
        """
        pass

    @lru_cache(maxsize=_max_cache_size)
    def get_single_spectrum(self, offset, process_peaks=False):
        """
        Read a single `Spectrum` at the specified offset in the spectral library file.

        Args:
            offset: The offset of the `Spectrum` in the spectral library file.
            process_peaks: Flag whether to directly process the `Spectrum`'s peaks.

        Returns:
            The `Spectrum` in the spectral library file at the specified offset.
        """
        self._mm.seek(offset)

        read_spectrum = self._read_spectrum()[0]
        if process_peaks:
            read_spectrum.process_peaks()

        return read_spectrum

    def _read_spectrum(self):
        pass


class SptxtReader(SpectralLibraryReader):
    """
    Read spectra from a spectral library .sptxt file.
    """

    _supported_extensions = ['.sptxt']

    def get_all_spectra(self):
        """
        Read all spectra in the .sptxt file.

        Returns:
            All spectra in the .sptxt file as a tuple of the `Spectrum` and its offset in the file.
        """
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


class SplibReader(SpectralLibraryReader):
    """
    Read spectra from a spectral library .splib file.
    """

    _supported_extensions = ['.splib']

    def get_all_spectra(self):
        """
        Read all spectra in the .splib file.

        Returns:
            All spectra in the .splib file as a tuple of the `Spectrum` and its offset in the file.
        """
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
            annotations[i] = _parse_annotation(self._mm.readline().strip())
            # info: \n terminated string
            info = self._mm.readline()
        # comment: \n terminated string
        comment = self._mm.readline()
        is_decoy = b' Remark=DECOY_' in comment

        read_spectrum = spectrum.Spectrum(identifier, precursor_mz, precursor_charge, None, peptide, is_decoy)
        read_spectrum.set_peaks(masses, intensities, annotations)

        return read_spectrum, file_offset


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
