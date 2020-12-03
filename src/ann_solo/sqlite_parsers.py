"""
This module provides support for common spectral library formats that
rely on SQLite3 databases.

Currently these include:
  - ELIB
  - DLIB

BLIB should be easy to add.
"""
import re
import sqlite3
import zlib
from typing import Tuple, Dict, Iterator

import numpy as np
from spectrum_utils.spectrum import MsmsSpectrum


class ElibParser:
    """Parse an ELIB or DLIB spectral library"""
    def __init__(self, filename: str) -> None:
        """
        Initialize an ELIB/DLIB spectral library parser.

        The ELIB and DLIB formats are described here:
        https://bitbucket.org/searleb/encyclopedia/wiki/EncyclopeDIA%20File%20Formats

        These files should have either '.blib' or '.dlib' extensions.

        Parameters
        ----------
        filename : str
            The file name of the DLIB or BLIB spectral library.
        """
        self._conn = sqlite3.connect(filename)
        self._cursor = self._conn.cursor()
        self._size = (self._cursor
                      .execute('SELECT COUNT(rowid) FROM entries')
                      .fetchone()[0])

        self._pos = 0

        decoy_map = self._cursor.execute(
            'SELECT PeptideSeq, isDecoy FROM peptidetoprotein'
        )
        self._is_decoy = {k: v for k, v in decoy_map}

    def _get_row(self, offset: int) -> Tuple:
        """
        Read a row at the offset

        Parameters
        ----------
        offset : int
            The index of the row to read.

        Returns
        -------
        A tuple containing the values of each column in the table.
        """
        vals = self._cursor.execute(
            'SELECT * FROM entries LIMIT 1 OFFSET ?', (str(offset),)
        )
        return vals.fetchone()

    def _parse_spectrum(self, row: Tuple, identifier: int) -> MsmsSpectrum:
        """
        Parse a single spectrum given one row of the table

        Parameters
        ----------
        row : Tuple
            One row of the 'entries' table.
        identifier : int
            The identifier for a spectrum.

        Returns
        -------
        MsmsSpectrum object
        """
        precursor_mz = row[0]
        precursor_charge = row[1]
        mods = _parse_mods(row[2])
        seq = row[3]
        mz_array = _decode(row[8], dtype='>d')
        int_array = _decode(row[10], dtype='>f')

        spectrum = MsmsSpectrum(identifier, precursor_mz, precursor_charge,
                                mz_array, int_array,
                                is_decoy=self._is_decoy[seq],
                                peptide=seq, modifications=mods)

        spectrum.annotate_peptide_fragments(10, "ppm")
        return spectrum

    def seek_first_spectrum(self):
        """Needed for for compatibility"""
        self._pos = 0

    def read_spectrum(self, offset: int = None) -> MsmsSpectrum:
        """
        Read a spectrum from the library.

        Parameters
        ----------
        offset : int
            The row to start reading from.
        """
        if offset is not None and offset >= 0:
            self._pos = offset

        spectrum_offset = self._pos
        if self._pos >= self._size:
            print(self._pos, self._size)
            raise StopIteration

        row = self._get_row(spectrum_offset)
        spectrum = self._parse_spectrum(row, spectrum_offset)
        self._pos += 1
        return spectrum, spectrum_offset

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
        rows = self._cursor.execute('SELECT * FROM entries')
        for spectrum_offset, row in enumerate(rows):
            spectrum = self._parse_spectrum(row, spectrum_offset)
            spectrum.is_processed = False
            yield spectrum, spectrum_offset

def _decode(array: bytes, dtype: str) -> np.ndarray:
    """
    Decode a zlib-compressed array

    Parameters
    ----------
    array : bytestring
        The zlib-compressed array.
    dtype : str
        The data type to be read by numpy. '>f' is for Big Endian
        floats, '>d' is for Big Endian doubles.
    """
    return np.frombuffer(zlib.decompress(array), dtype=dtype)


def _parse_mods(peptide: str) -> Dict[int, float]:
    """
    Parse a modified peptide string.

    Parameters
    ----------
    peptide : str
        The peptide string with modification indicated in square
        brackets.

    Returns
    -------
    Dict[int, float]
        The modifications for spectrum_utils.
    """
    mods = re.finditer("\[(.+?)\]", peptide)
    mod_dict = {}
    offset = 0
    for mod in mods:
        position = mod.start() - offset
        mod_dict[position] = float(mod.groups(1)[0])
        offset += mod.end() - mod.start()

    return mod_dict
