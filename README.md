ANN SoLo
========

For more information:

* [Official code website](https://bitbucket.org/proteinspector/ann-solo/)

**ANN SoLo** (Approximate Nearest Neighbor Spectral Library) uses approximate nearest neighbor indexing to significantly speed up spectral library searching in the case of (very) large search spaces, for example such as when performing an open modification search with a very wide precursor mass window.

The software is available as open-source under the Apache 2.0 license.

Application
-----------

**TODO:** Add information on command-line arguments and how to use the application.

Docker
------

A [Docker](https://www.docker.com/) file is provided to easily execute ANN SoLo on any operating system. Execute the following steps in the project's directory to build and run the Docker container:

    docker build -t ann-solo .
    docker run -v [local-path-to-data]:/data ann-solo [args]

And provide the required command-line arguments. The local data directory will be mounted as `/data`, so a spectral library file `lib.spql` present in this directory can be referred to as `/data/lib.spql`.

Dependencies
------------

Although the Python code itself is compatible with both Python 2 and Python 3, currently only Python 2.7 is fully supported due to limitations imposed by external libraries.
Not all external modules can be installed through `pip`, some of these can only be installed from source. Therefore, for ease of use it is recommended to run through Docker (see above).

For more information on the required dependencies and the installation procedure, see `build.sh`.

Contact
-------

For more information you can visit the [official code website](https://bitbucket.org/proteinspector/ann-solo/) and send a message through Bitbucket or send a mail to <wout.bittremieux@uantwerpen.be>.
