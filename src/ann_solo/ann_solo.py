import logging
from typing import List, Union

from ann_solo import spectral_library
from ann_solo import writer
from ann_solo.config import config


def ann_solo(spectral_library_filename: str, query_filename: str,
             out_filename: str, args: List[str]) -> None:
    """
    Run ANN-SoLo.

    The identified PSMs will be stored in the given file.

    Parameters
    ----------
    spectral_library_filename : str
        The spectral library file name.
    query_filename : str
        The query spectra file name.
    out_filename : str
        The mzTab output file name.
    args : List[str]
        List of additional search settings. List items either MUST match the
        commandline arguments (including '--' prefix;
        https://github.com/bittremieux/ANN-SoLo/wiki/Parameters) or MUST be
        values as strings.
    """
    # Explicitly set the search parameters when run from Python.
    main([spectral_library_filename, query_filename, out_filename, *args])


def main(args: Union[str, List[str]] = None):
    # Initialize logging.
    logging.basicConfig(format='{asctime} [{levelname}/{processName}] '
                               '{module}.{funcName} : {message}',
                        style='{', level=logging.DEBUG)

    # Load the configuration.
    config.parse(args)

    # Perform the search.
    spec_lib = spectral_library.SpectralLibrary(
        config.spectral_library_filename)
    identifications = spec_lib.search(config.query_filename)
    writer.write_mztab(identifications, config.out_filename,
                       spec_lib._library_reader)
    spec_lib.shutdown()

    logging.shutdown()


if __name__ == '__main__':
    # Use search parameters from sys.argv when run from CMD.
    main()
