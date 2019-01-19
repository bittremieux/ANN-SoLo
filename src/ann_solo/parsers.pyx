# distutils: language=c++
# cython: language_level=3

import numpy as np
cimport numpy as np
from libc.stdlib cimport atoi, free, malloc
from libc.stdint cimport uint32_t
from libc.string cimport memcpy, strstr, strtok
from libcpp.string cimport string
from libcpp.vector cimport vector
from posix.fcntl cimport open, O_RDONLY
from posix.unistd cimport off_t

from ann_solo.spectrum import Spectrum


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
        self._mmap = <char*>mmap(NULL, self._size, PROT_READ, MAP_SHARED, fd, 0)
        self._pos = 0

    def __dealloc__(self):
        munmap(self._mmap, self._size)

    cdef uint32_t _read_int(self) nogil:
        cdef uint32_t result = 0
        memcpy(&result, self._mmap + self._pos, 4)
        self._pos += 4
        return result

    cdef double _read_double(self) nogil:
        cdef double result = 0.0
        memcpy(&result, self._mmap + self._pos, 8)
        self._pos += 8
        return result

    cdef char* _read_line(self) nogil:
        cdef size_t offset = 0
        while self._pos + offset < self._size:
            if self._mmap[self._pos + offset] == b'\n':
                offset += 1
                break
            offset += 1
        cdef char *result = <char*>malloc((offset + 1) * sizeof(char))
        memcpy(result, self._mmap + self._pos, offset)
        result[offset] = b'\0'
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
        cdef vector[(string, int)] annotation
        cdef bint is_decoy

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
            strtok(name, '.')
            peptide = strtok(NULL, '.')
            strtok(NULL, '/')
            precursor_charge = atoi(strtok(NULL, ' '))
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
                line = self._read_line()
                annotation.push_back(parse_annotation(line))
                free(line)
                self._skip_line()
            line = self._read_line()
            is_decoy = strstr(line, ' Remark=DECOY_') != NULL
            free(line)

        annotation_p = np.full((num_peaks,), None, object)
        for i in range(num_peaks):
            if annotation[i][1] != -1:
                annotation_p[i] = (annotation[i][0].decode(), annotation[i][1])
        annotation.clear()

        spectrum = Spectrum(identifier, precursor_mz, precursor_charge, None,
                            peptide.decode(), is_decoy)
        spectrum.set_peaks(np.asarray(<np.float32_t[:num_peaks]> mz),
                           np.asarray(<np.float32_t[:num_peaks]> intensity),
                           annotation_p)

        return spectrum, spectrum_offset


cdef (string, int) parse_annotation(string raw) nogil:
    cdef size_t ion_index_end
    cdef char ion_type
    cdef int ion_index = -1
    cdef int charge = -1
    # Discard peaks that don't correspond to typical ion types.
    ion_type = raw.at(0)
    if ion_type == b'a' or ion_type == b'b' or ion_type == b'y':
        # The ion index is the subsequent numeric part.
        # `find_first_not_of` returns the position of the first character that
        # does not match or the end of the string.
        ion_index_end = raw.find_first_not_of(b'1234567890', 1)
        ion_index = stoi(raw.substr(1, ion_index_end), NULL, 10)
        # The ion is unmodified if the next character indicates the end of the
        # peak annotation or a specified charge.
        if raw.at(ion_index_end + 1) == b'/':
            charge = 1
        elif raw.at(ion_index_end + 1) == b'^':
            charge = stoi(raw.substr(ion_index_end + 1, ion_index_end + 2),
                          NULL, 10)
    return (string(1, ion_type) + to_string(ion_index), charge)
