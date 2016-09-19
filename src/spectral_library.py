import logging

import search_engine
import writer
from config import config


if __name__ == '__main__':
    # initialize logging
    logging.basicConfig(format='%(asctime)s [%(levelname)s/%(processName)s] %(module)s.%(funcName)s : %(message)s',
                        level=logging.DEBUG)

    # load the config
    config.parse()

    # execute the search
    if config.mode == 'bf':
        spec_lib = search_engine.SpectralLibraryBf(config.spectral_library_filename)
    elif config.mode == 'annoy':
        spec_lib = search_engine.SpectralLibraryAnnoy(config.spectral_library_filename)
    elif config.mode == 'hnsw':
        spec_lib = search_engine.SpectralLibraryHnsw(config.spectral_library_filename)
    else:
        spec_lib = search_engine.SpectralLibraryBf(config.spectral_library_filename)

    identifications = spec_lib.search(config.query_filename)
    writer.write_mztab(identifications, config.out_filename)

    logging.shutdown()
