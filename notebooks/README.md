# ANN-SoLo notebooks

These notebooks allow you to easily recreate all figures and tables in the ANN-SoLo manuscripts ([doi:10.1021/acs.jproteome.8b00359](https://doi.org/10.1021/acs.jproteome.8b00359) and [doi:10.1101/627497](https://doi.org/10.1101/627497)).

## Notebook overview

* `ann_index.ipynb`: Illustration of the ANN indexing procedure (v0.1).
* `hek293_stats.ipynb`: Comparison between ANN-SoLo, SpectraST, and MSFragger on the HEK293 data set (v0.1).
* `iprg2012_ann_hyperparameters.ipynb`: Effect of the ANN hyperparameters on index construction and querying for the iPRG2012 data set (v0.2).
* `iprg2012_cascade.ipynb`: Comparison between a cascade open search and a direct open search for the iPRG2012 data set (v0.2).
* `iprg2012_consensus.ipynb`: Comparison of the ANN-SoLo identifications to the iPRG2012 consensus results (v0.2).
* `iprg2012_fdr.ipynb`: Number of identifications for various types of searches for the iPRG2012 data set (v0.2).
* `iprg2012_num_candidates.ipynb`: Evaluation of potentially missed identifications versus the number of candidates retrieved from the ANN index for the iPRG2012 data set (v0.2).
* `iprg2012_profiling.ipynb`: Code profiling of brute-force versus ANN searches for the iPRG2012 data set (v0.2).
* `iprg2012_spectrum_hashing.ipynb`: Evaluation of vectorization bin width and hash length for the iPRG2012 data set (v0.2).
* `iprg2012_spectrum_representation.ipynb`: Evaluation of spectrum representation options for the iPRG2012 data set (v0.2).
* `kim2014_stats.ipynb`: Analysis of the Kim draft human proteome identification results (v0.2).
* `spec_lib_size.ipynb`: Historical evolution of spectral library sizes.

## Data

The necessary data to execute the notebooks is available on the PRIDE repository at identifier [PXD009861](https://www.ebi.ac.uk/pride/archive/projects/PXD009861) (v0.1) and [PXD013641](https://www.ebi.ac.uk/pride/archive/projects/PXD013641) (v0.2), and on the MassIVE repository at identifier [RMSV000000091.4](https://massive.ucsd.edu/ProteoSAFe/reanalysis.jsp?task=b25b8c664eb8477a9991c477a40af8c2) (v0.2).

The directory structure is based on the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) project. A `data` directory is expected at the same level of the `notebooks` directory. Additionally, ANN-SoLo should be installed as a Python package or the ANN-SoLo source code should be present in the `src` directory. The full directory and file structure is as follows:

```
├── data
│   ├── external
│   │   ├── hek293
│   │   │   └── b19xx_293T_proteinID_01A_QE3_122212.raw <--- The HEK293 raw files downloaded from PRIDE.
│   │   ├── iprg2012
│   │   |   ├── iPRG2012.mgf             <--- The iPRG2012 query file downloaded from MassIVE.
│   │   |   └── iprg2012ConsensusSpectrumIDcomparison.tsv <--- The iPRG2012 consensus spectrum identifications downloaded from MassIVE.
│   │   └── kim2014                      <--- The Kim2014 raw files downloaded from PRIDE.
│   ├── interim
│   │   ├── hek293                       <--- The HEK293 mgf files converted from raw.
│   │   └── kim2014                      <--- The Kim2014 mgf files converted from raw.
│   ├── processed
│   │   ├── hek293
│   │   │   ├── massive_human_hcd_unique_targetdecoy.splib <--- The compiled spectral library for the HEK293 data set.
│   │   │   ├── ann-solo                 <--- The ANN-SoLo mzTab identification results and logs.
│   │   │   │   ├── std
│   │   │   │   └── oms
│   │   │   └── msfragger                <--- The MSFragger tab-separated identification results and logs.
│   │   │   │   ├── std
│   │   │   │   └── oms
│   │   │   └── spectrast                <--- The SpectraST tab-separated (xls extension converted to txt) identification results and logs.
│   │   │       ├── std
│   │   │       └── oms
│   │   ├── iprg2012
│   │   │   ├── human_yeast_targetdecoy.splib  <--- The compiled spectral library for the iPRG2012 data set.
│   │   │   ├── human_yeast_targetdecoy.pepidx <--- The compiled spectral library for the iPRG2012 data set.
│   │   │   ├── ann_hyperparameters      <--- The ANN-SoLo mzTab identification results and logs for various hyperparameter settings.
│   │   │   ├── brute_force              <--- The brute-force mzTab identification results and logs.
│   │   │   ├── build_trees              <--- The logs to build the ANN indexes.
│   │   │   ├── profiling                <--- The ANN-SoLo profiling results.
│   │   │   └── spectrum_representation  <--- The ANN-SoLo mzTab identification results to evaluate spectrum representation options.
│   │   └── kim2014
│   │       ├── massivekb_targetdecoy.splib  <--- The compiled spectral library for the Kim2014 data set.
│   │       └── gpu                      <--- The Kim2014 mzTab identification results.
│   └── raw
├── notebooks
└── src
```
