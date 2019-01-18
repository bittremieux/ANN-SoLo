# cython: language_level=3

import numpy as np
cimport numpy as np
from libc.stdlib cimport atoi, free, malloc
from libc.stdint cimport uint8_t, uint32_t
from libc.string cimport memcpy, strstr, strtok
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


cdef struct PeakAnnotation:
    char ion_type
    uint8_t ion_index


cdef class SplibParser:

    cdef void *_mmap
    cdef size_t _size
    cdef size_t _pos

    def __cinit__(self, char *filename):
        fd = open(filename, O_RDONLY)
        # Get total file size.
        cdef stat statbuf
        fstat(fd, &statbuf)
        self._size = statbuf.st_size
        # Memory map the spectral library file.
        self._mmap = mmap(NULL, self._size, PROT_READ, MAP_SHARED, fd, 0)
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
            if (<char*>self._mmap)[self._pos + offset] == b'\n':
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
            if (<char*>self._mmap)[self._pos + offset] == b'\n':
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
        cdef double *mz
        cdef double *intensity
        cdef PeakAnnotation *annotation
        cdef uint8_t is_decoy

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
            # TODO: Is it possible to directly create NumPy arrays without
            #       Python interaction?
            mz = <double*>malloc(num_peaks * sizeof(double))
            intensity = <double*>malloc(num_peaks * sizeof(double))
            annotation = <PeakAnnotation*>malloc(
                num_peaks * sizeof(PeakAnnotation))
            for i in range(num_peaks):
                mz[i] = self._read_double()
                intensity[i] = self._read_double()
                # TODO: Parse peak annotations.
                self._skip_line()
                # annotation_str = self._mm.readline().strip()
                # if not _ignore_annotations:
                #     annotation[i] = _parse_annotation(annotation_str)
                self._skip_line()
            line = self._read_line()
            is_decoy = strstr(line, ' Remark=DECOY_') != NULL
            free(line)

        mz_p = np.empty((num_peaks,), np.float32)
        intensity_p = np.empty((num_peaks,), np.float32)
        for i in range(num_peaks):
            mz_p[i] = mz[i]
            intensity_p[i] = intensity[i]

        free(mz)
        free(intensity)
        free(annotation)

        spectrum = Spectrum(identifier, precursor_mz, precursor_charge, None,
                            peptide.decode(), is_decoy)
        spectrum.set_peaks(mz_p, intensity_p, None)

        return spectrum, spectrum_offset
