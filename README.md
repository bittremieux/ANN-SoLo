ANN-SoLo
========

![ANN-SoLo](ann-solo.png)

For more information:

* [Official code website](https://github.com/bittremieux/ANN-SoLo)

**ANN-SoLo** (**A**pproximate **N**earest **N**eighbor **S**pectral **L**ibrary) is a spectral library search engine for fast and accurate open modification searching. ANN-SoLo uses approximate nearest neighbor indexing to speed up open modification searching by selecting only a limited number of the most relevant library spectra to compare to an unknown query spectrum. This is combined with a cascade search strategy to maximize the number of identified unmodified and modified spectra while strictly controlling the false discovery rate and the shifted dot product score to sensitively match modified spectra to their unmodified counterpart.

The software is available as open-source under the Apache 2.0 license.

If you use ANN-SoLo in your work, please cite the following publications:

- Wout Bittremieux, Pieter Meysman, William Stafford Noble, Kris Laukens. **Fast Open Modification Spectral Library Searching through Approximate Nearest Neighbor Indexing.** _Journal of Proteome Research_ (2018). [doi:10.1021/acs.jproteome.8b00359](https://doi.org/10.1021/acs.jproteome.8b00359)

- Wout Bittremieux, Kris Laukens, William Stafford Noble. **Extremely fast and accurate open modification spectral library searching of high-resolution mass spectra using feature hashing and graphics processing units.** _bioRxiv_ (2019). [doi:10.1101/627497](https://doi.org/10.1101/627497)

Running ANN-SoLo
----------------

ANN-SoLo requires Python 3.6 or higher. The GPU-powered version of ANN-SoLo can be used on Linux systems, while the CPU-only version supports both the Linux and OSX platforms. An Nvidia CUDA-enabled GPU is required to use the GPU-powered version of ANN-SoLo. Please refer to the Faiss installation instructions linked below for more information on OS and GPU support.

- **NumPy** needs to be available prior to the installation of ANN-SoLo.
- The **Faiss** installation depends on a specific GPU version. Please refer to the [Faiss installation instructions](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md) for more information.

The recommended way to install ANN-SoLo is using pip:

    pip install ann_solo

For detailed instructions on how to run ANN-SoLo see the `src` folder or run `ann_solo -h` to get detailed information about the ANN-SoLo command-line interface.

### Dependencies

ANN-SoLo has the following dependencies:

- [ConfigArgParse](https://github.com/bw2/ConfigArgParse)
- [Cython](https://cython.org/)
- [Faiss](https://github.com/facebookresearch/faiss)
- [Joblib](https://joblib.readthedocs.io/)
- [Matplotlib](http://matplotlib.org/)
- [mmh3](https://pypi.org/project/mmh3/)
- [Numba](http://numba.pydata.org/)
- [NumExpr](https://github.com/pydata/numexpr)
- [NumPy](https://www.numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Pyteomics](http://pyteomics.readthedocs.io/)
- [SciPy](https://www.scipy.org/)
- [spectrum_utils](https://github.com/bittremieux/spectrum_utils)
- [tqdm](https://tqdm.github.io/)

We recommend installing these dependencies using conda. Any missing dependencies will be automatically installed when you install ANN-SoLo.

Contact
-------

For more information you can visit the [official code website](https://github.com/bittremieux/ANN-SoLo) or send an email to <wout.bittremieux@uantwerpen.be>.
