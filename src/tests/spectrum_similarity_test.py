import pytest

import numpy as np
import spectrum_utils.spectrum as sus

from ann_solo import spectrum
from ann_solo import spectrum_similarity as sim


@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(1)


@pytest.fixture
def sim_calc_all_match():
    n_peaks = 10
    mz = np.linspace(0, 10, n_peaks)
    intensity = spectrum._norm_intensity(np.random.exponential(1.0, n_peaks))
    spec1 = sus.MsmsSpectrum("spectrum1", 500, 2, mz, intensity)
    spec2 = sus.MsmsSpectrum(
        "spectrum2", 500, 2, np.copy(mz), np.copy(intensity)
    )
    peak_matches = np.asarray([(i, i) for i in range(n_peaks)])
    ssm = spectrum.SpectrumSpectrumMatch(spec1, spec2, peak_matches)
    return sim.SpectrumSimilarityCalculator(ssm)


@pytest.fixture
def sim_calc_no_match():
    n_peaks = 10
    mz = np.linspace(0.5, 10.5, n_peaks)
    intensity = spectrum._norm_intensity(np.random.exponential(1.0, n_peaks))
    spec1 = sus.MsmsSpectrum("spectrum1", 500, 2, mz, intensity)
    spec2 = sus.MsmsSpectrum("spectrum2", 500, 2, mz, intensity)
    peak_matches = np.asarray([])
    ssm = spectrum.SpectrumSpectrumMatch(spec1, spec2, peak_matches)
    return sim.SpectrumSimilarityCalculator(ssm)


def test_cosine(sim_calc_all_match, sim_calc_no_match):
    assert sim_calc_all_match.cosine() == pytest.approx(1.0)
    assert sim_calc_no_match.cosine() == pytest.approx(0.0)


def test_spectral_contrast_angle(sim_calc_all_match, sim_calc_no_match):
    assert sim_calc_all_match.spectral_contrast_angle() == pytest.approx(1.0)
    assert sim_calc_no_match.spectral_contrast_angle() == pytest.approx(0.0)


def test_hypergeometric_score(sim_calc_all_match, sim_calc_no_match):
    assert sim_calc_all_match.hypergeometric_score(
        min_mz=101, max_mz=1500, fragment_mz_tol=0.1
    ) == pytest.approx(0.0)
    assert (
        sim_calc_no_match.hypergeometric_score(
            min_mz=101, max_mz=1500, fragment_mz_tol=0.1
        )
        > 0.0
    )


def test_kendalltau(sim_calc_all_match, sim_calc_no_match):
    assert sim_calc_all_match.kendalltau() == pytest.approx(1.0)
    assert sim_calc_no_match.kendalltau() == pytest.approx(-1.0)


def test_ms_for_id_v1(sim_calc_all_match, sim_calc_no_match):
    assert sim_calc_all_match.ms_for_id_v1() > 10.0
    assert sim_calc_no_match.ms_for_id_v1() == pytest.approx(0.0)


def test_ms_for_id_v2(sim_calc_all_match, sim_calc_no_match):
    assert sim_calc_all_match.ms_for_id_v2() > 10.0
    assert sim_calc_no_match.ms_for_id_v2() == pytest.approx(0.0)


def test_manhattan(sim_calc_all_match, sim_calc_no_match):
    assert sim_calc_all_match.manhattan() == pytest.approx(0.0)
    assert np.isinf(sim_calc_no_match.manhattan())


def test_euclidean(sim_calc_all_match, sim_calc_no_match):
    assert sim_calc_all_match.euclidean() == pytest.approx(0.0)
    assert np.isinf(sim_calc_no_match.euclidean())
