import logging

from ann_solo import spectral_library, writer
from ann_solo.config import config


def main():
    # initialize logging
    logging.basicConfig(
            format='%(asctime)s [%(levelname)s/%(processName)s] '
                   '%(module)s.%(funcName)s : %(message)s',
            level=logging.DEBUG)

    # load the config
    config.parse()

    # execute the search
    if config.mode == 'bf':
        spec_lib = spectral_library.SpectralLibraryBf(
                config.spectral_library_filename)
    elif config.mode == 'ann':
        spec_lib = spectral_library.SpectralLibraryAnnoy(
                config.spectral_library_filename)

    identifications = spec_lib.search(config.query_filename)
    writer.write_mztab(identifications, config.out_filename,
                       spec_lib._library_reader)
    spec_lib.shutdown()

    logging.shutdown()


if __name__ == '__main__':
    main()
