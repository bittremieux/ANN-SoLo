import abc
import collections
import copy
import hashlib
import logging
import multiprocessing
import multiprocessing.pool
import os
import time

import annoy
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

    def __init__(self, lib_filename, lib_spectra=None):
        """
        Initialize the spectral library search engine from the given spectral
        library file.

        Args:
            lib_filename: The spectral library file name.
            lib_spectra: All valid spectra from the spectral library. Avoids
                re-reading a large spectral library if not `None`.

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

        # read all spectra in the query file and
        # split based on their precursor charge
        logging.info('Reading all query spectra')
        query_spectra = []
        for query_spectrum in tqdm.tqdm(
                reader.read_mgf(query_filename),
                desc='Query spectra read', unit='spectra', smoothing=0):
            # for queries with an unknown charge, try all possible charges
            for charge in [2, 3] if query_spectrum.precursor_charge is None\
                                 else [query_spectrum.precursor_charge]:
                query_spectrum_charge = (
                    copy.copy(query_spectrum)
                    if query_spectrum.precursor_charge is None
                    else query_spectrum)
                query_spectrum_charge.precursor_charge = charge
                # discard low-quality spectra
                if query_spectrum_charge.process_peaks().is_valid():
                    query_spectra.append(query_spectrum_charge)

        # sort the spectra based on their precursor charge and precursor mass
        query_spectra.sort(
                key=lambda spec: (spec.precursor_charge, spec.precursor_mz))

        # identify all spectra
        logging.info('Processing all query spectra')
        query_matches = {}
        # cascade level 1: standard search settings
        precursor_tols = [(config.precursor_tolerance_mass,
                           config.precursor_tolerance_mode)]
        # cascade level 2: open search settings
        if config.precursor_tolerance_mass_open is not None and \
                config.precursor_tolerance_mode_open is not None:
            precursor_tols.append((config.precursor_tolerance_mass_open,
                                   config.precursor_tolerance_mode_open))
        for cascade_level, (tol_mass, tol_mode) in enumerate(precursor_tols):
            level_matches = {}
            logging.debug('Level {} precursor mass tolerance: {} {}'.format(
                    cascade_level + 1, tol_mass, tol_mode))
            for query_spectrum in tqdm.tqdm(
                    query_spectra, desc='Query spectra processed',
                    unit='spectra', smoothing=0):
                # identify the query spectra in this cascade level
                query_match = self._find_match(
                        query_spectrum, tol_mass, tol_mode)
    
                if query_match.sequence is not None:
                    # make sure we only retain the best identification
                    # (i.e. for duplicated spectra if the
                    # precursor charge was unknown)
                    if (query_match.query_id not in level_matches or
                                query_match.search_engine_score >
                                level_matches[query_match.query_id].search_engine_score):
                        level_matches[query_match.query_id] = query_match
                
            # filter SSMs on FDR
            if cascade_level == 0:
                # small precursor mass window: standard FDR
                def filter_fdr(ssms):
                    return util.filter_fdr(ssms, config.fdr)
            else:
                # open search: group FDR
                def filter_fdr(ssms):
                    return util.filter_group_fdr(ssms, config.fdr,
                                                 config.fdr_tolerance_mass,
                                                 config.fdr_tolerance_mode,
                                                 config.fdr_min_group_size)
            for accepted_ssm in filter_fdr(level_matches.values()):
                    query_matches[accepted_ssm.query_id] = accepted_ssm
            
            logging.debug(
                    '{} spectra identified at {:.0%} FDR '
                    'after search level {}'.format(
                            len(query_matches), config.fdr, cascade_level + 1))
            
            # process the remaining spectra in the next cascade level
            query_spectra = [spec for spec in query_spectra
                             if spec.identifier not in query_matches]

        logging.info('Finished processing file %s', query_filename)

        return query_matches.values()

    def _find_match(self, query, tol_mass, tol_mode):
        """
        Identifies the given query Spectrum.

        Args:
            query: The query Spectrum to be identified.

        Returns:
            A SpectrumMatch identification. If the query couldn't be
            identified SpectrumMatch.sequence will be None.
        """
        # discard low-quality spectra
        if not query.is_valid():
            return spectrum.SpectrumMatch(query)

        start_total = start_candidates = time.time()

        # find all candidate library spectra
        # for which a match has to be computed
        candidates = self._filter_library_candidates(query, tol_mass, tol_mode)

        stop_candidates = start_match = time.time()

        if len(candidates) > 0:
            # find the best matching candidate spectrum
            match_candidate, match_score, _ = spectrum_match.get_best_match(
                    query, candidates)
            identification = spectrum.SpectrumMatch(
                    query, match_candidate, match_score)
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
    def _filter_library_candidates(self, query, tol_mass, tol_mode):
        """
        Find all candidate matches for the given query in the spectral library.

        Args:
            query: The query Spectrum for which candidate Spectra are retrieved
                from the spectral library.
            tol_mass: The size of the precursor mass window.
            tol_mode: The unit in which the precursor mass window is specified
                ('Da' or 'ppm').

        Returns:
            The candidate Spectra in the spectral library that need to be
            compared to the given query Spectrum.
        """
        pass

    def _get_mass_filter_idx(self, mass, charge, tol_mass, tol_mode):
        """
        Get the identifiers of all candidate matches that fall within the given
        window around the specified mass.

        Args:
            mass: The mass around which the window to identify the candidates
                is centered.
            charge: The precursor charge of the candidate matches.
            tol_mass: The size of the precursor mass window.
            tol_mode: The unit in which the precursor mass window is specified
                ('Da' or 'ppm').

        Returns:
            A list containing the identifiers of the candidate matches that
            fall within the given mass window.
        """
        if charge not in self._library_reader.spec_info['charge']:
            return []
        else:
            # check which mass differences fall
            # within the precursor mass window
            charge_spectra = self._library_reader.spec_info['charge'][charge]
            lib_masses = charge_spectra['precursor_mass']
            if tol_mode == 'Da':
                mass_filter = ne.evaluate(
                        'abs(mass - lib_masses) * charge <= tol_mass')
            elif tol_mode == 'ppm':
                mass_filter = ne.evaluate(
                        'abs(mass - lib_masses) / lib_masses * 10**6 <= tol_mass')
            else:
                mass_filter = np.arange(len(lib_masses))
    
            return charge_spectra['id'][mass_filter]


class SpectralLibraryBf(SpectralLibrary):
    """
    Traditional spectral library search engine.

    A traditional spectral library search engine uses the 'brute force'
    approach. This means that all library spectra within the precursor mass
    window are considered as candidate matches when identifying a query
    spectrum.
    """

    def _filter_library_candidates(self, query, tol_mass, tol_mode):
        """
        Find all candidate matches for the given query in the spectral library.

        Candidate matches are solely filtered on their precursor mass: they are
        included if their precursor mass falls within the specified window
        around the precursor mass from the query.

        Args:
            query: The query Spectrum for which candidate Spectra are retrieved
                from the spectral library.
            tol_mass: The size of the precursor mass window.
            tol_mode: The unit in which the precursor mass window is specified
                ('Da' or 'ppm').

        Returns:
            The candidate Spectra in the spectral library that need to be
            compared to the given query Spectrum.
        """
        # filter the candidates on the precursor mass window
        candidate_idx = self._get_mass_filter_idx(
                query.precursor_mz, query.precursor_charge, tol_mass, tol_mode)

        # read the candidates
        candidates = []
        for idx in candidate_idx:
            candidate = self._library_reader.get_spectrum(idx, True)
            if candidate.is_valid():
                candidates.append(candidate)

        return candidates


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
            ann_charge_ids = self._query_ann(query, num_candidates)
            # convert the numbered index for this specific charge
            # to global identifiers
            ann_filter = self._library_reader.spec_info['charge'][charge]['id'][ann_charge_ids]

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


class SpectralLibraryAnnoy(SpectralLibraryAnn):
    """
    Approximate nearest neighbors (ANN) spectral library search engine using
    the Annoy library for ANN retrieval.

    Annoy constructs a random projection tree forest for ANN retrieval.
    """

    _config_match_keys = ['min_mz', 'max_mz', 'bin_size', 'num_trees']

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
        # get the spectral library reader in the super-class initialization
        super().__init__(lib_filename)

        self._current_index = None, None
        
        do_create = False
        verify_file_existence = True
        if self._library_reader.is_recreated:
            logging.warning(
                    'ANN indices were created using non-compatible settings')
            verify_file_existence = False
        # check if an ANN index exists for each charge
        base_filename, _ = os.path.splitext(lib_filename)
        base_filename = '{}_{}'.format(
                base_filename, self._get_config_hash()[:7])
        # no need to build an ANN index for infrequent precursor charges
        ann_charges = sorted([
            charge for charge in self._library_reader.spec_info['charge']
            if len(self._library_reader.spec_info['charge'][charge]['id']) > config.ann_cutoff])
        create_ann_charges = []
        for charge in ann_charges:
            self._ann_filenames[charge] = '{}_{}.idxann'.format(
                    base_filename, charge)
            if not verify_file_existence or\
               not os.path.isfile(self._ann_filenames[charge]):
                do_create = True
                create_ann_charges.append(charge)
                logging.warning(
                        'Missing ANN index file for charge {}'.format(charge))

        # create the missing ANN indices
        if do_create:
            # add all spectra to the ANN indices
            logging.debug(
                    'Adding the spectra to the spectral library ANN indices')
            ann_indices = {}
            dim = spectrum.get_dim(
                    config.min_mz, config.max_mz, config.bin_size)[0]
            for charge in create_ann_charges:
                ann_index = annoy.AnnoyIndex(dim, 'angular')
                ann_index.set_seed(42)
                ann_indices[charge] = ann_index
            charge_counts = collections.Counter()
            lib_spectra_it = (lib_spectra if lib_spectra is not None
                              else self._library_reader._get_all_spectra())
            for lib_spectrum, _ in tqdm.tqdm(
                    lib_spectra_it,
                    desc='Library spectra added', unit='spectra'):
                charge = lib_spectrum.precursor_charge
                if charge in ann_indices.keys():
                    if lib_spectrum.process_peaks().is_valid():
                        ann_indices[charge].add_item(
                                charge_counts[charge],
                                lib_spectrum.get_vector())
                    charge_counts[charge] += 1

            # build the ANN indices
            logging.debug('Building the spectral library ANN indices')
            try:
                # os.sched_getaffinity is only available on some platforms
                num_cpus = len(os.sched_getaffinity(0))
            except AttributeError:
                num_cpus = os.cpu_count()
            with multiprocessing.pool.ThreadPool(
                    min(len(ann_indices), num_cpus)) as pool:
                pool.map(self._build_ann_index,
                         [(charge, ann_index, config.num_trees)
                          for charge, ann_index in ann_indices.items()])

            logging.info('Finished creating the spectral library ANN indices')
            
    def _build_ann_index(self, args):
        charge, ann_index, num_trees = args
        logging.debug('Creating new ANN index for charge {}'.format(charge))
        ann_index.build(num_trees)
        logging.debug('Saving the ANN index for charge {}'.format(charge))
        ann_index.save(self._ann_filenames[charge])
        # unload the index to prevent using excessive memory
        ann_index.unload()

    def shutdown(self):
        """
        Release any resources to gracefully shut down.
        
        The active ANN index is unloaded.
        """
        # unload the ANN index
        if self._current_index[1] is not None:
            self._current_index[1].unload()
            
        super().shutdown()

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
                # unload the previous index
                if self._current_index[1] is not None:
                    self._current_index[1].unload()
                # load the new index
                dim = spectrum.get_dim(
                        config.min_mz, config.max_mz, config.bin_size)[0]
                index = annoy.AnnoyIndex(dim, 'angular')
                index.set_seed(42)
                index.load(self._ann_filenames[charge])
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
        return np.asarray(ann_index.get_nns_by_vector(
                query.get_vector(), num_candidates, config.search_k))
