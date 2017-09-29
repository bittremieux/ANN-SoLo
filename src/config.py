import configargparse


class Config:
    """
    Spectral library configuration.

    Configuration settings can be specified in a config.ini file in the working directory, or as command-line arguments.
    """

    def __init__(self):
        """
        Initialize the configuration settings and provide sensible default values if possible.
        """

        self._parser = configargparse.ArgParser(description='Approximate nearest neighbor spectral library searching',
                                                default_config_files=['./config.ini'])

        # IO
        self._parser.add_argument('spectral_library_filename', help='spectral library file')
        self._parser.add_argument('query_filename', help='mgf file to identify')
        self._parser.add_argument('out_filename',
                                  help='the name of the mzTab output file containing the identifications')

        # PREPROCESSING
        # spectral library resolution to round mass values
        self._parser.add_argument('--resolution', default=None, type=int,
                                  help='spectral library resolution; masses will be rounded to the given number of '
                                       'decimals (default: no rounding)')

        # minimum and maximum fragment peak mass values
        self._parser.add_argument('--min_mz', default=11, type=int,
                                  help='minimum m/z value (inclusive, default: %(default)s Da)')
        self._parser.add_argument('--max_mz', default=2010, type=int,
                                  help='maximum m/z value (inclusive, default: %(default)s Da)')

        # remove peaks around the precursor mass from fragment spectra
        self._parser.add_argument('--remove_precursor', action='store_true',
                                  help='remove peaks around the precursor mass (default: no peaks are removed)')
        self._parser.add_argument('--remove_precursor_tolerance', default=0, type=float,
                                  help='the window (in Da) around the precursor mass to remove peaks '
                                       '(default: %(default)s Da)')

        # minimum fragment peak intensity to filter out noise peaks
        self._parser.add_argument('--min_intensity', default=0.001, type=float,
                                  help='remove peaks with a lower intensity relative to the maximum intensity '
                                       '(default: %(default)s)')

        # minimum number of fragment peaks or mass range (Dalton)
        self._parser.add_argument('--min_peaks', default=10, type=int,
                                  help='discard spectra with less peaks (default: %(default)s)')
        self._parser.add_argument('--min_mz_range', default=250, type=int,
                                  help='discard spectra with a smaller mass range (default: %(default)s)')

        # maximum number of fragment peaks to use for each spectrum
        self._parser.add_argument('--max_peaks_used', default=50, type=int,
                                  help='only use the specified most intense peaks (default: %(default)s)')

        # manner in which to scale the peak intensities
        self._parser.add_argument('--scaling', default='sqrt', type=str, choices=['sqrt', 'rank'],
                                  help='to reduce the influence of very intense peaks, scale the peaks by their square '
                                       'root or by their rank  (default: %(default)s)')

        # MATCHING
        # maximum SSM precursor mass tolerance
        self._parser.add_argument('--precursor_tolerance_mass', type=float, required=True,
                                  help='precursor mass tolerance')
        self._parser.add_argument('--precursor_tolerance_mode', type=str, choices=['Da', 'ppm'], required=True,
                                  help='precursor mass tolerance unit (options: %(choices)s)')

        # fragment peak matching
        self._parser.add_argument('--fragment_mz_tolerance', type=float, required=True,
                                  help='fragment mass tolerance (Da)')

        # shifted dot product
        self._parser.add_argument('--allow_peak_shifts', action='store_true',
                                  help='allow shifted peaks according to the precursor mass difference to accommodate '
                                       'for PTMs while calculating the dot product match score')

        # MODE
        # use an ANN index or the conventional brute-force mode
        self._parser.add_argument('--mode', type=str, choices=['ann', 'bf'], required=True,
                                  help="search using an approximate nearest neighbors or the conventional (brute-force)"
                                       " mode; 'bf': brute-force, 'ann': approximate nearest neighbors indexing")

        # bin size for the ANN index (Dalton)
        self._parser.add_argument('--bin_size', default=1.0, type=float,
                                  help='ANN vector bin width (default: %(default)s Da)')

        # number of candidates to retrieve from the ANN index for each query
        self._parser.add_argument('--num_candidates', default=1000, type=int,
                                  help='number of candidates to retrieve from the ANN index for each query '
                                       '(default: %(default)s)')

        # minimum number of candidates for a query before ANN indexing is used
        self._parser.add_argument('--ann_cutoff', default=20000, type=int,
                                  help='minimum number of candidates for a query before ANN indexing is used to refine '
                                       'the candidates (default: %(default)s)')

        # custom Annoy parameters
        # number of ANN trees
        self._parser.add_argument('--num_trees', default=100, type=int,
                                  help='number of ANN trees (default: %(default)s)')

        # number of nodes to explore during ANN searching
        self._parser.add_argument('--search_k', default=50000, type=int,
                                  help='number of nodes to explore in the ANN index during searching '
                                       '(only required when using Annoy mode; default: %(default)s)')

        # filled in 'parse', contains the specified settings
        self._namespace = None

    def parse(self, args_str=None):
        """
        Parse the configuration settings.

        Args:
            args_str: If None, the arguments are taken from sys.argv. Arguments that are not explicitly specified are
                      taken from the configuration file.
        """
        self._namespace = vars(self._parser.parse_args(args_str))

    def __getattr__(self, option):
        if self._namespace is not None:
            return self._namespace[option]
        else:
            raise RuntimeError('The configuration has not been initialized')

    def __getitem__(self, item):
        return self.__getattr__(item)


config = Config()
