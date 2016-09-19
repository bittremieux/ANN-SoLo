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
import nmslib_vector
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

    def _filter_library_candidates(self, query, tol_mass=None, tol_mode=None, num_candidates=None, ann_cutoff=None):
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
        if ann_cutoff is None:
            ann_cutoff = config.ann_cutoff

        # filter the candidates on the precursor mass window
        mass_filter = self._get_mass_filter_idx(query.precursor_mz, query.precursor_charge, tol_mass, tol_mode)

        # if there are too many candidates, refine using the ANN index
        if len(mass_filter) > ann_cutoff and query.precursor_charge in self._ann_filenames:
            # retrieve the most similar candidates from the ANN index
            ann_charge_ids = self._query_ann(query, num_candidates)
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

    @abc.abstractmethod
    def _get_ann_index(self, charge):
        """
        Get the ANN index for the specified charge.

        This allows on-demand loading of the ANN indices and prevents having to keep a large amount of data for the
        index into memory (depending on the ANN method).
        The previously used index is cached to avoid reloading the same index (only a single index is cached to prevent
        using an excessive amount of memory). If no index for the specified charge is cached yet, this index is loaded
        from the disk.

        To prevent loading the same index multiple times (incurring a significant performance quality) it is CRUCIAL
        that query spectra are sorted by precursor charge so the previous index can be reused.

        Args:
            charge: The precursor charge for which the ANN index is retrieved.

        Returns:
            The ANN index for the specified precursor charge.
        """
        pass

    @abc.abstractmethod
    def _query_ann(self, query, num_candidates):
        """
        Retrieve the nearest neighbors for the given query vector from its corresponding ANN index.

        Args:
            query: The query spectrum.
            num_candidates: The number of candidate neighbors to retrieve.

        Returns:
            A NumPy array containing the identifiers of the candidate neighbors retrieved from the ANN index.
        """
        pass


class SpectralLibraryAnnoy(SpectralLibraryAnn):
    """
    Approximate nearest neighbors (ANN) spectral library search engine using the Annoy library for ANN retrieval.

    Annoy constructs a random projection tree forest for ANN retrieval.
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

        self._ann_filenames = {}
        self._current_index = None, None
        do_create = False

        # check if an ANN index exists for each charge
        base_filename, _ = os.path.splitext(lib_filename)
        ann_charges = [charge for charge in self._library_reader.spec_info if
                       len(self._library_reader.spec_info[charge]['id']) > config.ann_cutoff]
        create_ann_charges = []
        for charge in ann_charges:
            self._ann_filenames[charge] = '{}_{}.idxann'.format(base_filename, charge)
            if not os.path.isfile(self._ann_filenames[charge]):
                do_create = True
                create_ann_charges.append(charge)
                logging.warning('Missing ANN index file for charge {}'.format(charge))

        # create the missing ANN indices
        if do_create:
            # add all spectra to the ANN indices
            logging.debug('Adding the spectra to the spectral library ANN indices')
            ann_indices = defaultdict(lambda: annoy.AnnoyIndex(spectrum.get_dim(config.min_mz, config.max_mz, config.bin_size)))
            charge_counts = collections.defaultdict(int)
            with self._library_reader as lib_reader:
                for lib_spectrum, _ in tqdm.tqdm(lib_reader._get_all_spectra(), desc='Library spectra added', unit='spectra'):
                    # discard infrequent precursor charges
                    charge = lib_spectrum.precursor_charge
                    if charge in create_ann_charges:
                        lib_spectrum.process_peaks()
                        if lib_spectrum.is_valid():
                            ann_indices[charge].add_item(charge_counts[charge], lib_spectrum.get_vector())
                            charge_counts[charge] += 1

            # build the ANN indices
            logging.debug('Building the spectral library ANN indices')

            # build only the ANN indices that contain sufficient points
            num_trees = config.num_trees
            for charge, ann_index in six.iteritems(ann_indices):
                logging.debug('Creating new ANN index for charge {}'.format(charge))
                ann_index.build(num_trees)
                logging.debug('Saving the ANN index for charge {}'.format(charge))
                ann_index.save(self._ann_filenames[charge])
                # unload the index to prevent using excessive memory
                ann_index.unload()

            logging.info('Finished creating the spectral library ANN indices')

    def _get_ann_index(self, charge):
        """
        Get the ANN index for the specified charge.

        This allows on-demand loading of the ANN indices and prevents having to keep a large amount of data for the
        index into memory (depending on the ANN method).
        The previously used index is cached to avoid reloading the same index (only a single index is cached to prevent
        using an excessive amount of memory). If no index for the specified charge is cached yet, this index is loaded
        from the disk.

        To prevent loading the same index multiple times (incurring a significant performance quality) it is CRUCIAL
        that query spectra are sorted by precursor charge so the previous index can be reused.

        Args:
            charge: The precursor charge for which the ANN index is retrieved.

        Returns:
            The ANN index for the specified precursor charge.
        """
        if self._current_index[0] != charge:
            logging.debug('Loading the ANN index for charge {}'.format(charge))
            # unload the previous index
            if self._current_index[1] is not None:
                self._current_index[1].unload()
            # load the new index
            self._current_index = charge, annoy.AnnoyIndex(spectrum.get_dim(config.min_mz, config.max_mz, config.bin_size))
            self._current_index[1].load(self._ann_filenames[charge])

        return self._current_index[1]

    def _query_ann(self, query, num_candidates):
        """
        Retrieve the nearest neighbors for the given query vector from its corresponding ANN index.

        Args:
            query: The query spectrum.
            num_candidates: The number of candidate neighbors to retrieve.

        Returns:
            A NumPy array containing the identifiers of the candidate neighbors retrieved from the ANN index.
        """
        ann_index = self._get_ann_index(query.precursor_charge)
        return np.asarray(ann_index.get_nns_by_vector(query.get_vector(), num_candidates, config.search_k))


class SpectralLibraryHnsw(SpectralLibraryAnn):
    """
    Approximate nearest neighbors (ANN) spectral library search engine using the nmslib library for ANN retrieval.

    Potentially any (A)NN technique provided by nmslib can be used, although here we use a hard-coded hierarchical
    navigable small-word (HNSW) graph for ANN retrieval.
    """

    _config_match_keys = ['min_mz', 'max_mz', 'bin_size', 'M']

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

        self._ann_filenames = {}
        self._current_index = None, None
        do_create = False

        # check if an ANN index exists for each charge
        base_filename, _ = os.path.splitext(lib_filename)
        ann_charges = [charge for charge in self._library_reader.spec_info if
                       len(self._library_reader.spec_info[charge]['id']) > config.ann_cutoff]
        create_ann_charges = []
        for charge in ann_charges:
            self._ann_filenames[charge] = '{}_{}.idxhnsw'.format(base_filename, charge)
            if not os.path.isfile(self._ann_filenames[charge]):
                do_create = True
                create_ann_charges.append(charge)
                logging.warning('Missing ANN index file for charge {}'.format(charge))

        # create the missing ANN indices
        if do_create:
            # add all spectra to the ANN indices
            logging.debug('Adding the spectra to the spectral library ANN indices')
            ann_indices = defaultdict(
                lambda: nmslib_vector.init('cosinesimil', [], 'hnsw', nmslib_vector.DataType.VECTOR, nmslib_vector.DistType.FLOAT))
            charge_counts = collections.defaultdict(int)
            with self._library_reader as lib_reader:
                for lib_spectrum, _ in tqdm.tqdm(lib_reader._get_all_spectra(), desc='Library spectra added', unit='spectra'):
                    # discard infrequent precursor charges
                    charge = lib_spectrum.precursor_charge
                    if charge in create_ann_charges:
                        lib_spectrum.process_peaks()
                        if lib_spectrum.is_valid():
                            nmslib_vector.addDataPoint(ann_indices[charge], charge_counts[charge], lib_spectrum.get_vector().tolist())
                            charge_counts[charge] += 1

            # build the ANN indices
            logging.debug('Building the spectral library ANN indices')

            # build only the ANN indices that contain sufficient points
            index_param = ['M={}'.format(config.M), 'post=2', 'efConstruction=800']
            for charge, ann_index in six.iteritems(ann_indices):
                logging.debug('Creating new ANN index for charge {}'.format(charge))
                nmslib_vector.createIndex(ann_index, index_param)
                logging.debug('Saving the ANN index for charge {}'.format(charge))
                nmslib_vector.saveIndex(ann_index, self._ann_filenames[charge])
                # unload the index to prevent running out of memory
                nmslib_vector.freeIndex(ann_index)

            logging.info('Finished creating the spectral library ANN indices')

    def _get_ann_index(self, charge):
        """
        Get the ANN index for the specified charge.

        This allows on-demand loading of the ANN indices and prevents having to keep a large amount of data for the
        index into memory (depending on the ANN method).
        The previously used index is cached to avoid reloading the same index (only a single index is cached to prevent
        using an excessive amount of memory). If no index for the specified charge is cached yet, this index is loaded
        from the disk.

        To prevent loading the same index multiple times (incurring a significant performance quality) it is CRUCIAL
        that query spectra are sorted by precursor charge so the previous index can be reused.

        Args:
            charge: The precursor charge for which the ANN index is retrieved.

        Returns:
            The ANN index for the specified precursor charge.
        """
        if self._current_index[0] != charge:
            logging.debug('Loading the ANN index for charge {}'.format(charge))
            # unload the previous index
            if self._current_index[1] is not None:
                nmslib_vector.freeIndex(self._current_index[1])
            # create the new index
            self._current_index = charge, nmslib_vector.init('cosinesimil', [], 'hnsw', nmslib_vector.DataType.VECTOR, nmslib_vector.DistType.FLOAT)
            # add all spectra with the given charge to the index (FIXME: silly nmslib behavior)
            spec_count = 0
            with self._library_reader as lib_reader:
                for lib_spectrum, _ in tqdm.tqdm(lib_reader._get_all_spectra(), desc='Library spectra added', unit='spectra'):
                    # discard spectra with a dissimilar precursor charge
                    if charge == lib_spectrum.precursor_charge:
                        lib_spectrum.process_peaks()
                        if lib_spectrum.is_valid():
                            nmslib_vector.addDataPoint(self._current_index[1], spec_count, lib_spectrum.get_vector().tolist())
                            spec_count += 1
            # load the index
            nmslib_vector.loadIndex(self._current_index[1], self._ann_filenames[charge])
            # set query parameters
            nmslib_vector.setQueryTimeParams(self._current_index[1], ['ef={}'.format(config.ef)])

        return self._current_index[1]

    def _query_ann(self, query, num_candidates):
        """
        Retrieve the nearest neighbors for the given query vector from its corresponding ANN index.

        Args:
            query: The query spectrum.
            num_candidates: The number of candidate neighbors to retrieve.

        Returns:
            A NumPy array containing the identifiers of the candidate neighbors retrieved from the ANN index.
        """
        ann_index = self._get_ann_index(query.precursor_charge)
        return np.asarray(nmslib_vector.knnQuery(ann_index, num_candidates, query.get_vector().tolist()))
