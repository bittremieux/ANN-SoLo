from __future__ import division
import abc
import collections
import copy
import logging
import os
import six
import time
from collections import defaultdict

import annoy
import numexpr as ne
import numpy as np
import tqdm

import reader
import spectrum
import spectrum_match
from config import config


# define FileNotFoundError for Python 2
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


@six.add_metaclass(abc.ABCMeta)
class SpectralLibrary(object):
    """
    Spectral library search engine.

    Spectral library search engines identify unknown query spectra by comparing each query spectrum against relevant
    known spectra in the spectral library, after which the query spectrum is assigned the identity of the library
    spectrum to which is matches best.
    """

    _config_match_keys = []

    def __init__(self, lib_filename):
        """
        Initialize the spectral library search engine from the given spectral library file.

        Args:
            lib_filename: The spectral library file name.

        Raises:
            FileNotFoundError: The given spectral library file wasn't found or isn't supported.
        """
        try:
            self._library_reader = reader.get_spectral_library_reader(lib_filename, self._config_match_keys)
        except FileNotFoundError as e:
            logging.error(e)
            raise

    def search(self, query_filename):
        """
        Identify all unknown spectra in the given query file.

        Args:
            query_filename: The query file in .mgf format containing the unknown spectra.

        Returns:
            A list of SpectrumMatch identifications.
        """
        logging.info('Identifying file %s', query_filename)

        # read all spectra in the query file and split based on their precursor charge
        logging.info('Reading all query spectra')
        query_spectra = []
        for query_spectrum in tqdm.tqdm(reader.read_mgf(query_filename), desc='Query spectra read', unit='spectra'):
            # for queries with an unknown charge, try all possible charge states
            for charge in [2, 3] if query_spectrum.precursor_charge is None else [query_spectrum.precursor_charge]:
                query_spectrum_charge = copy.copy(query_spectrum)   # TODO: don't needlessly copy
                query_spectrum_charge.precursor_charge = charge
                query_spectrum_charge.process_peaks()
                if query_spectrum_charge.is_valid():      # discard low-quality spectra
                    query_spectra.append(query_spectrum_charge)

        # sort the spectra based on their precursor charge and precursor mass
        query_spectra.sort(key=lambda spec: (spec.precursor_charge, spec.precursor_mz))

        # identify all spectra
        logging.info('Identifying all query spectra')
        query_matches = {}
        for query_spectrum in tqdm.tqdm(query_spectra, desc='Query spectra identified', unit='spectra', smoothing=0):
            query_match = self._find_match(query_spectrum)

            # discard spectra that couldn't be identified
            if query_match.sequence is not None:
                # make sure we only retain the best identification
                # (i.e. for duplicated spectra if the precursor charge was unknown)
                if query_match.query_id not in query_matches or\
                   query_match.search_engine_score > query_matches[query_match.query_id].search_engine_score:
                    query_matches[query_match.query_id] = query_match

        logging.info('Finished identifying file %s', query_filename)

        return query_matches.values()

    def _find_match(self, query):
        """
        Identifies the given query Spectrum.

        Args:
            query: The query Spectrum to be identified.

        Returns:
            A SpectrumMatch identification. If the query couldn't be identified SpectrumMatch.sequence will be None.
        """
        # discard low-quality spectra
        if not query.is_valid():
            return spectrum.SpectrumMatch(query)

        start_total = start_candidates = time.time()

        # find all candidate library spectra for which a match has to be computed
        candidates = self._filter_library_candidates(query)

        stop_candidates = start_match = time.time()

        if len(candidates) > 0:
            # find the best matching candidate spectrum
            match_candidate, match_score = spectrum_match.get_best_match(query, candidates)
            identification = spectrum.SpectrumMatch(query, match_candidate, match_score)
        else:
            identification = spectrum.SpectrumMatch(query)

        stop_match = stop_total = time.time()

        # store performance data
        identification.num_candidates = len(candidates)
        identification.time_candidates = stop_candidates - start_candidates
        identification.time_match = stop_match - start_match
        identification.time_total = stop_total - start_total

        return identification

    @abc.abstractmethod
    def _filter_library_candidates(self, query, tol_mass=None, tol_mode=None):
        """
        Find all candidate matches for the given query in the spectral library.

        Args:
            query: The query Spectrum for which candidate Spectra are retrieved from the spectral library.
            tol_mass: The size of the precursor mass window.
            tol_mode: The unit in which the precursor mass window is specified ('Da' or 'ppm').

        Returns:
            The candidate Spectra in the spectral library that need to be compared to the given query Spectrum.
        """
        pass

    def _get_mass_filter_idx(self, mass, charge, tol_mass, tol_mode):
        """
        Get the identifiers of all candidate matches that fall within the given window around the specified mass.

        Args:
            mass: The mass around which the window to identify the candidates is centered.
            charge: The precursor charge of the candidate matches.
            tol_mass: The size of the precursor mass window.
            tol_mode: The unit in which the precursor mass window is specified ('Da' or 'ppm').

        Returns:
            A list containing the identifiers of the candidate matches that fall within the given mass window.
        """
        # check which mass differences fall within the precursor mass window
        lib_masses = self._library_reader.spec_info[charge]['precursor_mass']
        if tol_mode == 'Da':
            mass_filter = ne.evaluate('abs(mass - lib_masses) * charge <= tol_mass')
        elif tol_mode == 'ppm':
            mass_filter = ne.evaluate('abs(mass - lib_masses) / lib_masses * 10**6 <= tol_mass')
        else:
            mass_filter = np.arange(len(lib_masses))

        return self._library_reader.spec_info[charge]['id'][mass_filter]


class SpectralLibraryBf(SpectralLibrary):
    """
    Traditional spectral library search engine.

    A traditional spectral library search engine uses the 'brute force' approach. This means that all library spectra
    within the precursor mass window are considered as candidate matches when identifying a query spectrum.
    """

    def _filter_library_candidates(self, query, tol_mass=None, tol_mode=None):
        """
        Find all candidate matches for the given query in the spectral library.

        Candidate matches are solely filtered on their precursor mass: they are included if their precursor mass falls
        within the specified window around the precursor mass from the query.

        Args:
            query: The query Spectrum for which candidate Spectra are retrieved from the spectral library.
            tol_mass: The size of the precursor mass window.
            tol_mode: The unit in which the precursor mass window is specified ('Da' or 'ppm').

        Returns:
            The candidate Spectra in the spectral library that need to be compared to the given query Spectrum.
        """
        if tol_mass is None:
            tol_mass = config.precursor_tolerance_mass
        if tol_mode is None:
            tol_mode = config.precursor_tolerance_mode

        # filter the candidates on the precursor mass window
        candidate_idx = self._get_mass_filter_idx(query.precursor_mz, query.precursor_charge, tol_mass, tol_mode)

        # read the candidates

        candidates = []
        with self._library_reader as lib_reader:
            for idx in candidate_idx:
                candidate = lib_reader.get_spectrum(idx, True)
                if candidate.is_valid():
                    candidates.append(candidate)

        return candidates


class SpectralLibraryAnn(SpectralLibrary):
    """
    Approximate nearest neighbors spectral library search engine.

    The spectral library uses an approximate nearest neighbor (ANN) technique to retrieve only the most similar library
    spectra to a query spectrum as potential matches during identification.
    """

    _config_match_keys = ['min_mz', 'max_mz', 'bin_size', 'num_trees']

    def __init__(self, lib_filename):
        """
        Initialize the spectral library from the given spectral library file.

        Further, the ANN indices are loaded from the associated index files. If these are missing, new index files are
        built and stored for all charge states separately.

        Args:
            lib_filename: The spectral library file name.

        Raises:
            FileNotFoundError: The given spectral library file wasn't found or isn't supported.
        """
        # get the spectral library reader in the super-class initialization
        super(self.__class__, self).__init__(lib_filename)

        # load the ANN index if it exists, or create a new one
        self._ann_indices = defaultdict(lambda: annoy.AnnoyIndex(spectrum.get_dim(config.min_mz, config.max_mz, config.bin_size)))

        do_create = False

        # load the ANN index for each charge
        base_filename, _ = os.path.splitext(lib_filename)
        for charge in self._library_reader.spec_info:
            if not os.path.isfile('{}_{}.idxann'.format(base_filename, charge)):
                do_create = True
                logging.warning('Missing idxann file for charge {}'.format(charge))
            else:
                self._ann_indices[charge] = annoy.AnnoyIndex(spectrum.get_dim(config.min_mz, config.max_mz, config.bin_size))
                self._ann_indices[charge].load('{}_{}.idxann'.format(base_filename, charge))

        # create the ANN index if required
        if do_create:
            # make sure that no old data remains
            for ann_index in self._ann_indices.values():
                ann_index.unload()
            self._ann_indices.clear()

            # add all spectra to the ANN indices
            logging.debug('Adding the spectra to the spectral library ANN indices')
            charge_counts = collections.defaultdict(int)
            with self._library_reader as lib_reader:
                for lib_spectrum, _ in tqdm.tqdm(lib_reader._get_all_spectra(), desc='Library spectra added', unit='spectra'):
                    lib_spectrum.process_peaks()
                    if lib_spectrum.is_valid():
                        self._ann_indices[lib_spectrum.precursor_charge].add_item(
                            charge_counts[lib_spectrum.precursor_charge], lib_spectrum.get_vector())
                        charge_counts[lib_spectrum.precursor_charge] += 1

            # build the ANN indices
            logging.debug('Building the spectral library ANN indices')

            num_trees = config.num_trees
            for ann_index in self._ann_indices.values():
                ann_index.build(num_trees)

            # store the ANN indices
            logging.debug('Saving the spectral library ANN indices')
            for charge, ann_index in six.iteritems(self._ann_indices):
                ann_index.save('{}_{}.idxann'.format(base_filename, charge))

            logging.info('Finished creating the spectral library ANN indices')

    def _filter_library_candidates(self, query, tol_mass=None, tol_mode=None, num_candidates=None, k=None, ann_cutoff=None):
        """
        Find all candidate matches for the given query in the spectral library.

        First, the most similar candidates are retrieved from the ANN index to restrict the search space to only
        the most relevant candidates. Next, these candidates are further filtered on their precursor mass similar
        to the brute-force approach.

        Args:
            query: The query Spectrum for which candidate Spectra are retrieved from the spectral library.
            tol_mass: The size of the precursor mass window.
            tol_mode: The unit in which the precursor mass window is specified ('Da' or 'ppm').
            num_candidates: The number of candidates to retrieve from the ANN index.
            k: The number of nodes to search in the ANN index during candidate retrieval.
            ann_cutoff: The minimum number of candidates for the query to use the ANN index to reduce the search space.

        Returns:
            The candidate Spectra in the spectral library that need to be compared to the given query Spectrum.
        """
        if tol_mass is None:
            tol_mass = config.precursor_tolerance_mass
        if tol_mode is None:
            tol_mode = config.precursor_tolerance_mode
        if num_candidates is None:
            num_candidates = config.num_candidates
        if k is None:
            k = config.search_k
        if ann_cutoff is None:
            ann_cutoff = config.ann_cutoff

        # filter the candidates on the precursor mass window
        mass_filter = self._get_mass_filter_idx(query.precursor_mz, query.precursor_charge, tol_mass, tol_mode)

        # if there are too many candidates, refine using the ANN index
        if len(mass_filter) > ann_cutoff:
            # retrieve the most similar candidates from the ANN index
            ann_charge_ids = np.asarray(self._ann_indices[query.precursor_charge].get_nns_by_vector(
                query.get_vector(), num_candidates, k))
            # convert the numbered index for this specific charge to global identifiers
            ann_filter = self._library_reader.spec_info[query.precursor_charge]['id'][ann_charge_ids]

            # select the candidates passing both the ANN filter and precursor mass filter
            candidate_idx = np.intersect1d(ann_filter, mass_filter, True)
        else:
            candidate_idx = mass_filter

        # read the candidates
        candidates = []
        with self._library_reader as lib_reader:
            for idx in candidate_idx:
                candidate = lib_reader.get_spectrum(idx, True)
                if candidate.is_valid():
                    candidates.append(candidate)

        return candidates
