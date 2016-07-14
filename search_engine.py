import abc
import copy
import logging
import os
import six
import time
from collections import defaultdict

import annoy
import joblib
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

    def __init__(self, lib_filename):
        """
        Initialize the spectral library from the given spectral library file.

        Args:
            lib_filename: The spectral library file containing all library spectra.

        Raises:
            FileNotFoundError: The given spectral library file wasn't found.
        """
        self._base_filename, _ = os.path.splitext(lib_filename)

        try:
            self._load()
        except FileNotFoundError as e:
            logging.error(e)
            raise
        except ValueError as e:
            logging.warning(e)

            self._reset()
            self._create()

    def _load(self):
        """
        Load existing information for the spectral library.

        For each spectrum in the spectral library its offset in the spectral library file is retrieved for quick
        random-access reading of the spectra. Further, the precursor mass for each spectrum is retrieved to be able to
        filter on a precursor mass window. Finally, the settings used to construct the existing spectral library are
        retrieved.

        Raises:
            FileNotFoundError: The given spectral library file wasn't found.
            ValueError: No information file for the spectral library found.
        """
        logging.info('Loading the spectral library information')

        # make sure all required files are present
        if not os.path.isfile(self._base_filename + '.splib') and not os.path.isfile(self._base_filename + '.sptxt'):
            raise FileNotFoundError('Missing spectral library file (required file format: splib or sptxt)')
        if not os.path.isfile(self._base_filename + '.spcfg'):
            raise ValueError('Missing spcfg file')  # this means we should just recreate the spectral library

        # load the spectral library information
        self._offsets, self._precursor_masses, load_config = joblib.load(self._base_filename + '.spcfg')

        # check if the configuration is compatible to the loaded configuration
        if not self._is_valid_config(load_config):
            raise ValueError('The spectral library search engine was created using a non-compatible configuration')

        logging.info('Finished loading the spectral library information')

    @abc.abstractmethod
    def _is_valid_config(self, load_config):
        """
        Check if the configuration used to previously build the spectral library search engine conforms to the current
        configuration.

        Args:
            load_config: The configuration used to previously build the spectral library search engine.

        Returns:
            True if the configuration is valid, False if not.
        """
        pass

    def _reset(self):
        """Reset the spectral library."""
        self._offsets = self._precursor_masses = None

    def _create(self):
        """
        Create new information for the spectral library.

        For each spectrum in the spectral library its offset in the spectral library file is stored to enable quick
        random-access reading of the spectra. Further, the precursor mass for each spectrum is stored to be able to
        filter on a precursor mass window. Finally, the settings used to construct the existing spectral library are
        stored.
        """
        logging.info('Creating the spectral library information from file %s', self._base_filename)

        # read all the spectra in the spectral library
        offsets = defaultdict(list)
        precursor_masses = defaultdict(list)
        with reader.get_spectral_library_reader(self._base_filename) as lib_reader:
            for library_spectrum, offset in tqdm.tqdm(lib_reader.get_all_spectra(),
                                                      desc='Library spectra read', unit='spectra'):
                # store the spectrum information for easy retrieval, discard low-quality spectra
                library_spectrum.process_peaks()
                if library_spectrum.is_processed_and_high_quality():
                    offsets[library_spectrum.precursor_charge].append(offset)
                    precursor_masses[library_spectrum.precursor_charge].append(library_spectrum.precursor_mz)

                    self._add_library_spectrum(library_spectrum)

        # convert the standard lists to NumPy arrays for better performance later on
        self._offsets = {}
        for charge, offset in offsets.items():
            self._offsets[charge] = np.array(offset)
        self._precursor_masses = {}
        for charge, masses in precursor_masses.items():
            self._precursor_masses[charge] = np.array(masses)

        # store the information
        logging.debug('Saving the spectral library information')
        joblib.dump((self._offsets, self._precursor_masses, config.get_build_config()), self._base_filename + '.spcfg',
                    compress=9, protocol=2)

        logging.info('Finished creating the spectral library information')

    @abc.abstractmethod
    def _add_library_spectrum(self, library_spectrum):
        """
        Additional processing to add a library spectrum for spectral library searching.

        Args:
            library_spectrum: The Spectrum to add.
        """
        pass

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
                query_spectrum_charge = copy.copy(query_spectrum)
                query_spectrum_charge.precursor_charge = charge
                query_spectrum_charge.process_peaks()
                if query_spectrum_charge.is_processed_and_high_quality():      # discard low-quality spectra
                    query_spectra.append(query_spectrum_charge)

        # sort the spectra based on their precursor charge and precursor mass
        query_spectra.sort(key=lambda spec: (spec.precursor_charge, spec.precursor_mz))

        # identify all spectra
        logging.info('Identifying all query spectra')
        query_matches = {}
        with reader.get_spectral_library_reader(self._base_filename) as lib_reader:
            for query_spectrum in tqdm.tqdm(
                    query_spectra, desc='Query spectra identified', unit='spectra', smoothing=0):
                query_match = self._find_match(query_spectrum, lib_reader)

                # discard spectra that couldn't be identified
                if query_match.sequence is not None:
                    # make sure we only retain the best identification
                    # (i.e. for duplicated spectra if the precursor charge was unknown)
                    if query_match.query_id not in query_matches or\
                       query_match.search_engine_score > query_matches[query_match.query_id].search_engine_score:
                        query_matches[query_match.query_id] = query_match

        logging.info('Finished identifying file %s', query_filename)

        return query_matches.values()

    def _find_match(self, query, lib_reader):
        """
        Identifies the given query Spectrum.

        Args:
            query: The query Spectrum to be identified.
            lib_reader: The spectral library reader to read spectra.

        Returns:
            A SpectrumMatch identification. If the query couldn't be identified SpectrumMatch.sequence will be None.
        """
        # discard low-quality spectra
        if not query.is_processed_and_high_quality():
            return spectrum.SpectrumMatch(query)

        start_total = start_candidates = time.time()

        # find all candidate library spectra for which a match as to be computed
        candidates = self._filter_library_candidates(query, lib_reader)

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
    def _filter_library_candidates(self, query, lib_reader, tol_mass=None, tol_mode=None):
        """
        Find all candidate matches for the given query in the spectral library.

        Args:
            query: The query Spectrum for which candidate Spectra are retrieved from the spectral library.
            lib_reader: The spectral library reader to read spectra.
            tol_mass: The size of the precursor mass window.
            tol_mode: The unit in which the precursor mass window is specified ('Da' or 'ppm').

        Returns:
            The candidate Spectra in the spectral library that need to be compared to the given query Spectrum.
        """
        pass

    def _get_mass_filter_idx(self, mass, charge, tol_mass, tol_mode):
        """
        Get the indices of all candidate matches that fall within the given window around the specified mass.

        Args:
            mass: The mass around which the window to identify the candidates is centered.
            charge: The precursor charge of the candidate matches.
            tol_mass: The size of the precursor mass window.
            tol_mode: The unit in which the precursor mass window is specified ('Da' or 'ppm').

        Returns:
            A dict with as key each of the allowed charges and for each charge the indices of the candidate matches
            within the mass window.
        """
        # check which mass differences fall within the precursor mass window
        lib_masses = self._precursor_masses[charge]
        if tol_mode == 'Da':
            mass_filter = np.where(ne.evaluate('abs(mass - lib_masses) * charge') <= tol_mass)[0]
        elif tol_mode == 'ppm':
            mass_filter = np.where(ne.evaluate('abs(mass - lib_masses) / lib_masses * 10**6') <= tol_mass)[0]
        else:
            mass_filter = np.arange(len(lib_masses))

        return mass_filter


class SpectralLibraryBf(SpectralLibrary):
    """
    Traditional spectral library search engine.

    A traditional spectral library search engine uses the 'brute force' approach. This means that all library spectra
    within the precursor mass window are considered as candidate matches when identifying a query spectrum.
    """

    def _is_valid_config(self, load_config):
        """
        Check if the configuration used to previously build the spectral library search engine conforms to the current
        configuration.

        For the brute-force approach, no specific configuration is used to build the search engine, so just always
        return True.

        Args:
            load_config: The configuration used to previously build the spectral library search engine.

        Returns:
            True.
        """
        return True

    def _add_library_spectrum(self, library_spectrum):
        pass

    def _filter_library_candidates(self, query, lib_reader, tol_mass=None, tol_mode=None):
        """
        Find all candidate matches for the given query in the spectral library.

        Candidate matches are solely filtered on their precursor mass: they are included if their precursor mass falls
        within the specified window around the precursor mass from the query.

        Args:
            query: The query Spectrum for which candidate Spectra are retrieved from the spectral library.
            lib_reader: The spectral library reader to read spectra.
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
        mass_filter = self._get_mass_filter_idx(query.precursor_mz, query.precursor_charge, tol_mass, tol_mode)

        # read the candidates
        candidates = []
        for offset in self._offsets[query.precursor_charge][mass_filter]:
            candidate = lib_reader.get_single_spectrum(offset, True)
            if candidate.is_processed_and_high_quality():
                candidates.append(candidate)

        return candidates


class SpectralLibraryAnn(SpectralLibrary):
    """
    Approximate nearest neighbors spectral library search engine.

    The spectral library uses an approximate nearest neighbor (ANN) technique to retrieve only the most similar library
    spectra to a query spectrum as potential matches during identification.
    """

    def __init__(self, lib_filename):
        """
        Initialize the spectral library from the given spectral library file.

        Args:
            lib_filename: The spectral library file containing all library spectra.

        Raises:
            FileNotFoundError: The given spectral library file wasn't found.
        """
        self._ann_indices = defaultdict(lambda: annoy.AnnoyIndex(spectrum.get_dim(config.min_mz, config.max_mz,
                                                                                  config.bin_size)))

        super(self.__class__, self).__init__(lib_filename)

    def _load(self):
        """
        Load existing information for the spectral library.

        For each spectrum in the spectral library its offset in the spectral library file is retrieved for quick
        random-access reading of the spectra. Further, the precursor mass for each spectrum is retrieved to be able to
        filter on a precursor mass window. Finally, the settings used to construct the existing spectral library are
        retrieved.
        Furthermore, the ANN indices for all charges are loaded.

        Raises:
            FileNotFoundError: The given spectral library file wasn't found.
            ValueError: If no information file for the spectral library found, if the ANN index was previously created
                        using non-compatible settings, or if some of the ANN indices are missing.
        """
        # retain the current settings which will be overwritten on loading
        super(self.__class__, self)._load()

        # load the ANN index for each charge
        for charge in self._offsets.keys():
            if not os.path.isfile('{}_{}.idxann'.format(self._base_filename, charge)):
                raise ValueError('Missing idxann file for charge {}'.format(charge))
            else:
                self._ann_indices[charge] = annoy.AnnoyIndex(spectrum.get_dim(config.min_mz, config.max_mz,
                                                                              config.bin_size))
                self._ann_indices[charge].load('{}_{}.idxann'.format(self._base_filename, charge))

    def _is_valid_config(self, load_config):
        """
        Check if the configuration used to previously build the spectral library search engine conforms to the current
        configuration.

        Args:
            load_config: The configuration used to previously build the spectral library search engine.

        Returns:
            True if the configuration is valid, False if not.
        """
        return load_config == config.get_build_config()

    def _reset(self):
        """
        Reset the spectral library.

        All ANN indices are released.
        """
        for ann_index in self._ann_indices.values():
            ann_index.unload()
        self._ann_indices.clear()

        super(self.__class__, self)._reset()

    def _create(self, num_trees=None):
        """
        Create new information for the spectral library.

        For each spectrum in the spectral library its offset in the spectral library file is stored to enable quick
        random-access reading of the spectra. Further, the precursor mass for each spectrum is stored to be able to
        filter on a precursor mass window. Finally, the settings used to construct the existing spectral library are
        stored.
        Furthermore, ANN indices are built and stored for all charge states separately.

        Args:
            num_trees: The number of individual trees to build.
        """
        if num_trees is None:
            num_trees = config.num_trees

        super(self.__class__, self)._create()

        # build the ANN indices
        logging.debug('Building the spectral library ANN indices')
        for ann_index in self._ann_indices.values():
            ann_index.build(num_trees)

        # store the ANN indices
        logging.debug('Saving the spectral library ANN indices')
        for charge, ann_index in self._ann_indices.items():
            ann_index.save('{}_{}.idxann'.format(self._base_filename, charge))

        logging.info('Finished creating the spectral library ANN indices')

    def _add_library_spectrum(self, library_spectrum):
        self._ann_indices[library_spectrum.precursor_charge].add_item(
            self._ann_indices[library_spectrum.precursor_charge].get_n_items(), library_spectrum.get_vector())

    def _filter_library_candidates(self, query, lib_reader, tol_mass=None, tol_mode=None,
                                   num_candidates=None, k=None, ann_cutoff=None):
        """
        Find all candidate matches for the given query in the spectral library.

        First, the most similar candidates are retrieved from the ANN index to restrict the search space to only
        the most relevant candidates. Next, these candidates are further filtered on their precursor mass similar
        to the brute-force approach.

        Args:
            query: The query Spectrum for which candidate Spectra are retrieved from the spectral library.
            lib_reader: The spectral library reader to read spectra.
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
            ann_filter = np.zeros(num_candidates, np.uint32)
            for i, candidate_i in enumerate(self._ann_indices[query.precursor_charge].get_nns_by_vector(
                    query.get_vector(), num_candidates, k)):
                ann_filter[i] = candidate_i

            # select the candidates passing both the ANN filter and precursor mass filter
            candidate_filter = np.intersect1d(ann_filter, mass_filter, True)
        else:
            candidate_filter = mass_filter

        # read the candidates
        candidates = []
        for offset in self._offsets[query.precursor_charge][candidate_filter]:
            candidate = lib_reader.get_single_spectrum(offset, True)
            if candidate.is_processed_and_high_quality():
                candidates.append(candidate)

        return candidates
