import logging

from ann_solo import spectral_library
from ann_solo import writer
from ann_solo.config import config


def main():
    # Initialize logging.
    logging.basicConfig(format='{asctime} [{levelname}/{processName}] '
                               '{module}.{funcName} : {message}',
                        style='{', level=logging.DEBUG)

    # Load the configuration.
    config.parse()

    # Perform the search.
    spec_lib = spectral_library.SpectralLibrary(
        config.spectral_library_filename)
    identifications = spec_lib.search(config.query_filename)
    writer.write_mztab(identifications, config.out_filename,
                       spec_lib._library_reader)
    spec_lib.shutdown()

    logging.shutdown()


if __name__ == '__main__':
    main()
