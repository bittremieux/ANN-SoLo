# distutils: language=c++

import numpy as np
cimport numpy as np
from cython.operator cimport dereference as deref
from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdint cimport uint32_t
from libcpp.string cimport string
from libcpp.vector cimport vector
from posix.fcntl cimport open
from posix.fcntl cimport O_RDONLY
from posix.unistd cimport off_t
from spectrum_utils.spectrum import MsmsSpectrum
from spectrum_utils.spectrum import PeptideFragmentAnnotation


cdef extern from 'sys/mman.h' nogil:
    void *mmap(void *addr, size_t length, int prot, int flags, int fildes,
               off_t off)
    int munmap(void *addr, size_t length)
    enum: PROT_READ
    enum: MAP_FILE
    enum: MAP_SHARED

cdef extern from 'sys/stat.h' nogil:
    cdef struct stat:
        off_t st_size
    cdef int fstat(int fd, stat *buf)

cdef extern from 'string.h' nogil:
    void *memchr(const void *s, int c, size_t n)

cdef extern from '<string>' namespace 'std' nogil:
    int stoi(const string &stri, size_t *idx, int base)
    string to_string(int val)


cdef class SplibParser:

    cdef char *_mmap
    cdef size_t _size
    cdef size_t _pos

    def __cinit__(self, char *filename):
        fd = open(filename, O_RDONLY)
        # Get total file size.
        cdef stat statbuf
        fstat(fd, &statbuf)
        self._size = statbuf.st_size
        # Memory map the spectral library file.
        self._mmap = <char*>mmap(NULL, self._size, PROT_READ, MAP_SHARED, fd,
                                 0)
        self._pos = 0

    def __dealloc__(self):
        munmap(self._mmap, self._size)

    cdef uint32_t _read_int(self) nogil:
        self._pos += 4
        return deref(<uint32_t*>(self._mmap + self._pos - 4))

    cdef double _read_double(self) nogil:
        self._pos += 8
        return deref(<double*>(self._mmap + self._pos - 8))

    cdef string _read_line(self) nogil:
        cdef size_t offset = 0
        while self._pos + offset < self._size:
            if self._mmap[self._pos + offset] == b'\n':
                offset += 1
                break
            offset += 1
        cdef string result = string(self._mmap + self._pos, offset)
        self._pos += offset
        return result

    cdef void _skip_line(self) nogil:
        cdef size_t offset = 0
        while self._pos + offset < self._size:
            if self._mmap[self._pos + offset] == b'\n':
                offset += 1
                break
            offset += 1
        self._pos += offset

    def seek_first_spectrum(self):
        with nogil:
            # Go to the start of the file and skip non-spectrum data.
            self._pos = 8
            self._skip_line()
            for _ in range(self._read_int()):
                self._skip_line()

    def read_spectrum(self, offset: int = None):
        cdef uint32_t num_peaks
        cdef float *mz
        cdef float *intensity
        cdef vector[(string, int, int)] annotation
        cdef bint is_decoy
        cdef size_t peptide_pos, peptide_len, charge_pos, charge_len

        if offset is not None and offset >= 0:
            self._pos = offset

        with nogil:
            if self._pos >= self._size:
                raise StopIteration

            spectrum_offset = self._pos
            # Identifier.
            identifier = self._read_int()
            # Peptide sequence.
            name = self._read_line()
            peptide_pos = name.find(b'.') + 1
            peptide_len = name.find(b'.', peptide_pos) - peptide_pos
            peptide = name.substr(peptide_pos, peptide_len)
            charge_pos = name.find(b'/', peptide_pos + peptide_len) + 1
            charge_len = name.find(b' ', charge_pos)
            precursor_charge = stoi(name.substr(charge_pos, charge_len), NULL,
                                    10)
            # Precursor m/z.
            precursor_mz = self._read_double()
            # Status.
            self._skip_line()
            # Number of peaks.
            num_peaks = self._read_int()
            # Read all peaks.
            mz = <float*>malloc(num_peaks * sizeof(float))
            intensity = <float*>malloc(num_peaks * sizeof(float))
            for i in range(num_peaks):
                mz[i] = <float>self._read_double()
                intensity[i] = <float>self._read_double()
                annotation.push_back(parse_annotation(self._read_line()))
                self._skip_line()
            is_decoy = self._read_line().find(b' Remark=DECOY_') != <size_t>-1

        annotation_p = np.full((num_peaks,), None, object)
        for i in range(num_peaks):
            ion_type, ion_index, charge = annotation[i]
            if charge != -1:
                annotation_p[i] = PeptideFragmentAnnotation(
                    charge, mz[i], ion_type.decode(), ion_index)
        annotation.clear()

        spectrum = MsmsSpectrum(identifier, precursor_mz, precursor_charge,
                                np.asarray(<np.float32_t[:num_peaks]> mz),
                                np.asarray(<np.float32_t[:num_peaks]>
                                           intensity), annotation_p,
                                is_decoy=is_decoy)
        spectrum.peptide = peptide.decode()

        free(mz)
        free(intensity)

        return spectrum, spectrum_offset


cdef (string, int, int) parse_annotation(string raw) nogil:
    cdef size_t ion_index_end, ion_charge_end

    cdef char ion_type = raw.at(0)
    cdef int ion_index = -1
    cdef int charge = -1
    # Discard peaks that don't correspond to typical ion types.
    if ion_type == b'a' or ion_type == b'b' or ion_type == b'y':
        # The ion index is the subsequent numeric part.
        # `find_first_not_of` returns the position of the first character that
        # does not match or `npos` (-1).
        ion_index_end = raw.find_first_not_of(b'1234567890', 1)
        ion_index = stoi(raw.substr(1, ion_index_end), NULL, 10)
        # The ion is unmodified if the next character indicates the end of the
        # peak annotation or a specified charge.
        ion_charge_end = raw.find(b'/', ion_index_end)
        if ion_charge_end == ion_index_end:
            charge = 1
        elif raw.at(ion_index_end) == b'^':
            charge = stoi(raw.substr(ion_index_end + 1,
                                     ion_charge_end - ion_index_end - 1),
                          NULL, 10)
    return (string(1, ion_type), ion_index, charge)
