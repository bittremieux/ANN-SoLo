import abc
import collections
import copy
import hashlib
import logging
import multiprocessing.pool
import os
import time

import faiss
import numexpr as ne
import numpy as np
import tqdm

from ann_solo import reader, spectrum, spectrum_match, util
from ann_solo.config import config


class SpectralLibrary(metaclass=abc.ABCMeta):
    """
    Spectral library search engine.

    Spectral library search engines identify unknown query spectra by comparing
    each query spectrum against relevant known spectra in the spectral library,
    after which the query spectrum is assigned the identity of the library
    spectrum to which is matches best.
    """

    _config_match_keys = []

    def __init__(self, lib_filename):
        """
        Initialize the spectral library search engine from the given spectral
        library file.

        Args:
            lib_filename: The spectral library file name.

        Raises:
            FileNotFoundError: The given spectral library file wasn't found or
                isn't supported.
        """
        try:
            self._library_reader = reader.get_spectral_library_reader(
                lib_filename, self._get_config_hash())
            self._library_reader.open()
        except FileNotFoundError as e:
            logging.error(e)
            raise

    def _get_config_hash(self):
        config_match = {config_key: config[config_key]
                        for config_key in self._config_match_keys}
        return hashlib.sha1(str(config_match).encode('utf-8')).hexdigest()

    def shutdown(self):
        """
        Release any resources to gracefully shut down.
        """
        self._library_reader.close()

    def search(self, query_filename):
        """
        Identify all unknown spectra in the given query file.

        Args:
            query_filename: The query file in .mgf format containing the
                unknown spectra.

        Returns:
            A list of SpectrumMatch identifications.
        """
        logging.info('Processing file %s', query_filename)

        # Read all spectra in the query file and
        # split based on their precursor charge.
        logging.info('Reading all query spectra')
        query_spectra = collections.defaultdict(list)
        for query_spectrum in tqdm.tqdm(
                reader.read_mgf(query_filename),
                desc='Query spectra read', unit='spectra', smoothing=0):
            # For queries with an unknown charge, try all possible charges.
            if query_spectrum.precursor_charge is not None:
                query_spectra_charge = [query_spectrum]
            else:
                query_spectra_charge = []
                for charge in (2, 3):
                    query_spectra_charge.append(copy.copy(query_spectrum))
                    query_spectra_charge[-1].precursor_charge = charge
            for query_spectrum_charge in query_spectra_charge:
                # Discard low-quality spectra.
                if query_spectrum_charge.process_peaks().is_valid():
                    (query_spectra[query_spectrum_charge.precursor_charge]
                     .append(query_spectrum_charge))

        # Identify all spectra.
        logging.info('Processing all query spectra')
        query_matches = {}
        # Cascade level 1: standard search settings.
        precursor_tols = [(config.precursor_tolerance_mass,
                           config.precursor_tolerance_mode)]
        # Cascade level 2: open search settings.
        if config.precursor_tolerance_mass_open is not None and \
                config.precursor_tolerance_mode_open is not None:
            precursor_tols.append((config.precursor_tolerance_mass_open,
                                   config.precursor_tolerance_mode_open))
        for level, (tol_mass, tol_mode) in enumerate(precursor_tols, 1):
            level_matches = {}
            logging.debug('Level %d precursor mass tolerance: %s %s',
                          level, tol_mass, tol_mode)
            for charge, query_spectra_charge in query_spectra.items():
                logging.debug('Process %d spectra with precursor charge %d',
                              len(query_spectra_charge), charge)
                for query_match in self._find_match_batch(
                        query_spectra_charge, charge, tol_mass, tol_mode):
                    # Make sure we only retain the best identification
                    # (i.e. for duplicated spectra if the
                    # precursor charge was unknown).
                    if (query_match.query_id not in level_matches or
                            query_match.search_engine_score >
                            (level_matches[query_match.query_id]
                                .search_engine_score)):
                        level_matches[query_match.query_id] = query_match

            # Filter SSMs on FDR after each cascade level.
            if level == 1:
                # Small precursor mass window: standard FDR.
                def filter_fdr(ssms):
                    return util.filter_fdr(ssms, config.fdr)
            else:
                # Open search: group FDR.
                def filter_fdr(ssms):
                    return util.filter_group_fdr(ssms, config.fdr,
                                                 config.fdr_tolerance_mass,
                                                 config.fdr_tolerance_mode,
                                                 config.fdr_min_group_size)
            for accepted_ssm in filter_fdr(level_matches.values()):
                query_matches[accepted_ssm.query_id] = accepted_ssm

            logging.debug('%d spectra identified at %.2f FDR after search '
                          'level %d', len(query_matches), config.fdr, level)

            # Process the remaining spectra in the next cascade level.
            query_spectra_remaining = {}
            for charge, query_spectra_charge in query_spectra.items():
                query_spectra_remaining[charge] = [
                    spec for spec in query_spectra_charge
                    if spec.identifier not in query_matches]
            query_spectra = query_spectra_remaining

        logging.info('Finished processing file %s', query_filename)

        return query_matches.values()

    def _find_match_batch(self, queries, charge, tol_mass, tol_mode):
        """
        Finds the best library matches for a batch of query spectra with the
        same precursor charge.

        Args:
            queries: The query spectra to be identified.
            charge: The query spectra's precursor charge.
            tol_mass: The size of the precursor mass window.
            tol_mode: The unit in which the precursor mass window is specified
                ('Da' or 'ppm').

        Returns:
            A list of SpectrumMatch identifications for each query spectrum.
        """
        identifications = []
        # Find all candidate library spectra
        # to which a match has to be computed.
        for query, candidates in zip(queries, self._filter_library_candidates(
                queries, charge, tol_mass, tol_mode)):
            # Find the best matching candidate.
            if candidates:
                candidate, score, _ = spectrum_match.get_best_match(
                    query, candidates)
                identification = spectrum.SpectrumMatch(
                    query, candidate, score)
                identification.num_candidates = len(candidates)
                identifications.append(identification)

        return identifications

    @abc.abstractmethod
    def _filter_library_candidates(self, queries, charge, tol_mass, tol_mode):
        """
        Find all candidate matches for the given query in the spectral library.

        Args:
            queries: The query spectra for which candidate spectra are
                retrieved from the spectral library.
            charge: The query spectra's precursor charge.
            tol_mass: The size of the precursor mass window.
            tol_mode: The unit in which the precursor mass window is specified
                ('Da' or 'ppm').

        Returns:
            An iterator over the library candidate spectra for each query
            spectrum.
        """
        pass

    def _get_mass_filter_idx(self, query_mzs, charge, tol_val, tol_mode):
        """
        Get the identifiers of all library candidates that fall within the
        given window around the specified precursor masses.

        Args:
            query_mzs: The precursor masses around which the window to
                identify the candidates is centered.
            charge: The query spectra's precursor charge.
            tol_val: The size of the precursor mass window.
            tol_mode: The unit in which the precursor mass window is specified
                ('Da' or 'ppm').

        Returns:
            An iterator over the library identifiers of the candidate matches
            for each query spectrum, or an iterator in case the spectral
            library doesn't contain any spectra with the given precursor
            charge.
        """
        if charge not in self._library_reader.spec_info['charge']:
            yield from ()
        else:
            charge_spectra = self._library_reader.spec_info['charge'][charge]
            lib_mzs = charge_spectra['precursor_mass'].reshape((1, -1))
            # TODO: Check speed difference with numpy.where() or with a single
            #       loop through the library spectra.
            if tol_mode == 'Da':
                mass_filter = ne.evaluate(
                    'abs(query_mzs - lib_mzs) * charge <= tol_val')
            elif tol_mode == 'ppm':
                mass_filter = ne.evaluate(
                    'abs(query_mzs - lib_mzs) / lib_mzs * 10**6 <= tol_val')
            else:
                mass_filter = np.full(
                    (query_mzs.shape[0], lib_mzs.shape[1]), True)

            for query_filter in mass_filter:
                yield charge_spectra['id'][query_filter]


class SpectralLibraryBf(SpectralLibrary):
    """
    Traditional spectral library search engine.

    A traditional spectral library search engine uses the 'brute force'
    approach. This means that all library spectra within the precursor mass
    window are considered as candidate matches when identifying a query
    spectrum.
    """

    def _filter_library_candidates(self, queries, charge, tol_mass, tol_mode):
        """
        Find all candidate matches for the given query in the spectral library.

        Candidate matches are solely filtered on their precursor mass: they are
        included if their precursor mass falls within the specified window
        around the precursor mass from the query.

        Args:
            queries: The query spectra for which candidate spectra are
                retrieved from the spectral library.
            charge: The query spectra's precursor charge.
            tol_mass: The size of the precursor mass window.
            tol_mode: The unit in which the precursor mass window is specified
                ('Da' or 'ppm').

        Returns:
            An iterator over the library candidate spectra for each query
            spectrum.
        """
        precursor_mzs = (np.asarray([query.precursor_mz for query in queries])
                         .reshape((-1, 1)))
        # Filter the candidates for all queries on the precursor mass window.
        for candidate_idx in self._get_mass_filter_idx(
                precursor_mzs, charge, tol_mass, tol_mode):
            # Yield all candidates for each query.
            candidates = []
            for idx in candidate_idx:
                candidate = self._library_reader.get_spectrum(idx, True)
                if candidate.is_valid():
                    candidates.append(candidate)

            yield candidates


class SpectralLibraryAnn(SpectralLibrary):
    """
    Approximate nearest neighbors spectral library search engine.

    The spectral library uses an approximate nearest neighbor (ANN) technique
    to retrieve only the most similar library spectra to a query spectrum as
    potential matches during identification.
    """

    _ann_filenames = {}

    _ann_index_lock = multiprocessing.Lock()

    def _filter_library_candidates(self, query, tol_mass, tol_mode,
                                   num_candidates=None, ann_cutoff=None):
        """
        Find all candidate matches for the given query in the spectral library.

        First, the most similar candidates are retrieved from the ANN index to
        restrict the search space to only the most relevant candidates. Next,
        these candidates are further filtered on their precursor mass similar
        to the brute-force approach.

        Args:
            query: The query Spectrum for which candidate Spectra are retrieved
                from the spectral library.
            tol_mass: The size of the precursor mass window.
            tol_mode: The unit in which the precursor mass window is specified
                ('Da' or 'ppm').
            num_candidates: The number of candidates to retrieve from the ANN
                index.
            ann_cutoff: The minimum number of candidates for the query to use
                the ANN index to reduce the search space.

        Returns:
            The candidate Spectra in the spectral library that need to be
                compared to the given query Spectrum.
        """
        if num_candidates is None:
            num_candidates = config.num_candidates
        if ann_cutoff is None:
            ann_cutoff = config.ann_cutoff

        charge = query.precursor_charge

        # filter the candidates on the precursor mass window
        mass_filter = self._get_mass_filter_idx(
            query.precursor_mz, charge, tol_mass, tol_mode)

        # if there are too many candidates, refine using the ANN index
        if len(mass_filter) > ann_cutoff and charge in self._ann_filenames:
            # retrieve the most similar candidates from the ANN index
            ann_filter = self._query_ann(query, num_candidates)

            # select the candidates passing both the ANN filter
            # and precursor mass filter
            candidate_idx = np.intersect1d(ann_filter, mass_filter, True)
        else:
            candidate_idx = mass_filter

        # read the candidates
        candidates = []
        for idx in candidate_idx:
            candidate = self._library_reader.get_spectrum(idx, True)
            if candidate.is_valid():
                candidates.append(candidate)

        return candidates

    @abc.abstractmethod
    def _get_ann_index(self, charge):
        """
        Get the ANN index for the specified charge.

        This allows on-demand loading of the ANN indices and prevents having to
        keep a large amount of data for the index into memory (depending on
        the ANN method).
        The previously used index is cached to avoid reloading the same index
        (only a single index is cached to prevent using an excessive amount of
        memory). If no index for the specified charge is cached yet, this index
        is loaded from the disk.

        To prevent loading the same index multiple times (incurring a
        significant performance quality) it is CRUCIAL that query spectra are
        sorted by precursor charge so the previous index can be reused.

        Args:
            charge: The precursor charge for which the ANN index is retrieved.

        Returns:
            The ANN index for the specified precursor charge.
        """
        pass

    @abc.abstractmethod
    def _query_ann(self, query, num_candidates):
        """
        Retrieve the nearest neighbors for the given query vector from its
        corresponding ANN index.

        Args:
            query: The query spectrum.
            num_candidates: The number of candidate neighbors to retrieve.

        Returns:
            A NumPy array containing the identifiers of the candidate neighbors
            retrieved from the ANN index.
        """
        pass


class SpectralLibraryFaiss(SpectralLibraryAnn):

    """
    Approximate nearest neighbors (ANN) spectral library search engine using
    the FAISS library for ANN retrieval.
    """

    _config_match_keys = ['min_mz', 'max_mz', 'bin_size', 'hash_len',
                          'num_list']

    def __init__(self, lib_filename, lib_spectra=None):
        """
        Initialize the spectral library from the given spectral library file.

        Further, the ANN indices are loaded from the associated index files. If
        these are missing, new index files are built and stored for all charge
        states separately.

        Args:
            lib_filename: The spectral library file name.
            lib_spectra: All valid spectra from the spectral library. Avoids
                re-reading a large spectral library if not `None`. This needs
                to be an iterable giving tuples of which the first element is a
                Spectrum and the second element is ignored.

        Raises:
            FileNotFoundError: The given spectral library file wasn't found or
                isn't supported.
        """
        # Get the spectral library reader in the super-class initialization.
        super().__init__(lib_filename)

        self._current_index = None, None

        verify_file_existence = True
        if self._library_reader.is_recreated:
            logging.warning(
                'ANN indices were created using non-compatible settings')
            verify_file_existence = False
        # Check if an ANN index exists for each charge.
        base_filename, _ = os.path.splitext(lib_filename)
        base_filename = '{}_{}'.format(
            base_filename, self._get_config_hash()[:7])
        # No need to build an ANN index for infrequent precursor charges.
        min_num_items = max(config.ann_cutoff, config.num_list)
        ann_charges = [charge for charge, charge_info in
                       self._library_reader.spec_info['charge'].items()
                       if len(charge_info['id']) > min_num_items]
        create_ann_charges = []
        for charge in sorted(ann_charges):
            self._ann_filenames[charge] = '{}_{}.idxann'.format(
                base_filename, charge)
            if not verify_file_existence or \
                    not os.path.isfile(self._ann_filenames[charge]):
                create_ann_charges.append(charge)
                logging.warning(
                    'Missing ANN index file for charge {}'.format(charge))

        # Create the missing FAISS indices.
        if create_ann_charges:
            logging.debug(
                'Adding the spectra to the spectral library ANN indices')
            # Collect vectors for all spectra per precursor charge.
            charge_vectors = {charge: {'x': [], 'ids': []}
                              for charge in create_ann_charges}
            lib_spectra_it = (lib_spectra if lib_spectra is not None
                              else self._library_reader._get_all_spectra())
            for lib_spectrum, _ in tqdm.tqdm(
                    lib_spectra_it,
                    desc='Library spectra added', unit='spectra'):
                charge = lib_spectrum.precursor_charge
                if charge in charge_vectors.keys() and \
                        lib_spectrum.process_peaks().is_valid():
                    charge_vectors[charge]['x'].append(
                        lib_spectrum.get_hashed_vector())
                    charge_vectors[charge]['ids'].append(
                        lib_spectrum.identifier)
            # Build an individual FAISS index per precursor charge.
            logging.debug('Building the spectral library ANN indices')
            for charge, vector_ids in charge_vectors.items():
                logging.debug(
                    'Creating new ANN index for charge {}'.format(charge))
                # TODO: GPU
                quantizer = faiss.IndexFlatIP(config.hash_len)
                ann_index = faiss.IndexIVFFlat(quantizer, config.hash_len,
                                               config.num_list,
                                               faiss.METRIC_INNER_PRODUCT)
                vectors = np.asarray(vector_ids['x'])
                ids = np.asarray(vector_ids['ids'], np.int64)
                ann_index.train(vectors)
                ann_index.add_with_ids(vectors, ids)
                logging.debug(
                    'Saving the ANN index for charge {}'.format(charge))
                faiss.write_index(ann_index, self._ann_filenames[charge])

            logging.info('Finished creating the spectral library ANN indices')

    def _get_ann_index(self, charge):
        """
        Get the ANN index for the specified charge.

        This allows on-demand loading of the ANN indices and prevents having to
        keep a large amount of data for the index into memory (depending on the
        ANN method).
        The previously used index is cached to avoid reloading the same index
        (only a single index is cached to prevent using an excessive amount of
        memory). If no index for the specified charge is cached yet, this index
        is loaded from the disk.

        To prevent loading the same index multiple times (incurring a
        significant performance quality) it is CRUCIAL that query spectra are
        sorted by precursor charge so the previous index can be reused.

        Args:
            charge: The precursor charge for which the ANN index is retrieved.

        Returns:
            The ANN index for the specified precursor charge.
        """
        with self._ann_index_lock:
            if self._current_index[0] != charge:
                logging.debug(
                    'Loading the ANN index for charge {}'.format(charge))
                index = faiss.read_index(self._ann_filenames[charge])
                index.nprobe = config.num_probe
                self._current_index = charge, index

            return self._current_index[1]

    def _query_ann(self, query, num_candidates):
        """
        Retrieve the nearest neighbors for the given query vector from its
        corresponding ANN index.

        Args:
            query: The query spectrum.
            num_candidates: The number of candidate neighbors to retrieve.

        Returns:
            A NumPy array containing the identifiers of the candidate neighbors
            retrieved from the ANN index.
        """
        ann_index = self._get_ann_index(query.precursor_charge)
        _, candidate_idx = ann_index.search(
            np.expand_dims(query.get_hashed_vector(), axis=0), num_candidates)
        return candidate_idx[candidate_idx != -1]
