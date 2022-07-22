import pytest

import numpy as np
import spectrum_utils.spectrum as sus

from ann_solo import spectrum
from ann_solo import spectrum_similarity as sim


def test_cosine_identical():
    n_peaks = 10
    mz1 = np.linspace(0, 10, n_peaks)
    int1 = spectrum._norm_intensity(np.random.exponential(1.0, n_peaks))
    spec1 = sus.MsmsSpectrum("spectrum1", 500, 2, mz1, int1)
    spec2 = sus.MsmsSpectrum("spectrum2", 500, 2, np.copy(mz1), np.copy(int1))
    peak_matches = np.asarray([[i, i] for i in range(n_peaks)])
    ssm = spectrum.SpectrumSpectrumMatch(spec1, spec2, peak_matches)
    sim_factory = sim.SpectrumSimilarityFactory(ssm)
    assert sim_factory.cosine() == pytest.approx(1.0)


def test_cosine_no_match():
    n_peaks = 10
    mz1 = np.linspace(0, 10, n_peaks)
    int1 = spectrum._norm_intensity(np.random.exponential(1.0, n_peaks))
    spec1 = sus.MsmsSpectrum("spectrum1", 500, 2, mz1, int1)
    mz2 = np.linspace(0.5, 10.5, n_peaks)
    int2 = spectrum._norm_intensity(np.random.exponential(1.0, n_peaks))
    spec2 = sus.MsmsSpectrum("spectrum2", 500, 2, mz2, int2)
    peak_matches = np.asarray([])
    ssm = spectrum.SpectrumSpectrumMatch(spec1, spec2, peak_matches)
    sim_factory = sim.SpectrumSimilarityFactory(ssm)
    assert sim_factory.cosine() == pytest.approx(0.0)
