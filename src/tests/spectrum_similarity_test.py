import pytest

import numpy as np
import spectrum_utils.spectrum as sus

from ann_solo import spectrum
from ann_solo import spectrum_similarity as sim


@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(1)


@pytest.fixture
def all_match():
    # MS2PIP (HCD v20210416) simulated spectrum of HPYLEDR/2.
    mz = np.asarray(
        [
            138.066,  # b1
            235.119,  # b2
            398.182,  # b3
            511.266,  # b4
            640.309,  # b5
            755.336,  # b6
            175.119,  # y1
            290.146,  # y2
            419.188,  # y3
            532.273,  # y4
            695.336,  # y5
            792.389,  # y6
        ]
    )
    intensity = np.asarray(
        [
            0.03675187,  # b1
            0.41731364,  # b2
            0.00473946,  # b3
            0.00332476,  # b4
            0.00320261,  # b5
            0.00670335,  # b6
            0.40390085,  # y1
            0.09983288,  # y2
            0.01661951,  # y3
            0.05734070,  # y4
            0.22102276,  # y5
            0.77388125,  # y6
        ]
    )
    spec1 = sus.MsmsSpectrum("HPYLEDR", 465.227, 2, mz, intensity)
    spec2 = sus.MsmsSpectrum(
        "HPYLEDR", 465.227, 2, np.copy(mz), np.copy(intensity)
    )
    peak_matches = np.asarray([(i, i) for i in range(len(mz))])
    ssm = spectrum.SpectrumSpectrumMatch(spec1, spec2, peak_matches)
    return sim.SpectrumSimilarityCalculator(ssm)


@pytest.fixture
def no_match():
    # MS2PIP (HCD v20210416) simulated spectrum of HPYLEDR/2.
    mz1 = np.asarray(
        [
            138.066,  # b1
            235.119,  # b2
            398.182,  # b3
            511.266,  # b4
            640.309,  # b5
            755.336,  # b6
            175.119,  # y1
            290.146,  # y2
            419.188,  # y3
            532.273,  # y4
            695.336,  # y5
            792.389,  # y6
        ]
    )
    intensity1 = np.asarray(
        [
            0.03675187,  # b1
            0.41731364,  # b2
            0.00473946,  # b3
            0.00332476,  # b4
            0.00320261,  # b5
            0.00670335,  # b6
            0.40390085,  # y1
            0.09983288,  # y2
            0.01661951,  # y3
            0.05734070,  # y4
            0.22102276,  # y5
            0.77388125,  # y6
        ]
    )
    spec1 = sus.MsmsSpectrum("HPYLEDR", 465.227, 2, mz1, intensity1)
    # MS2PIP (HCD v20210416) simulated spectrum of GDLVLFDK/2.
    mz2 = np.asarray(
        [
            58.0287,  # b1
            173.056,  # b2
            286.140,  # b3
            385.208,  # b4
            498.292,  # b5
            645.361,  # b6
            760.388,  # b7
            147.113,  # y1
            262.140,  # y2
            409.208,  # y3
            522.292,  # y4
            621.361,  # y5
            734.445,  # y6
            849.472,  # y7
        ]
    )
    intensity2 = np.asarray(
        [
            0.00000000,  # b1
            0.12522728,  # b2
            0.18020111,  # b3
            0.04328780,  # b4
            0.00542208,  # b5
            0.00330758,  # b6
            0.00208561,  # b7
            0.26473886,  # y1
            0.30046007,  # y2
            0.56388106,  # y3
            0.49369887,  # y4
            0.43157844,  # y5
            0.20395883,  # y6
            0.00216236,  # y7
        ]
    )
    spec2 = sus.MsmsSpectrum("GDLVLFDK", 453.750, 2, mz2, intensity2)
    peak_matches = np.asarray([])
    ssm = spectrum.SpectrumSpectrumMatch(spec1, spec2, peak_matches)
    return sim.SpectrumSimilarityCalculator(ssm)


@pytest.fixture
def partial_match():
    # MS2PIP (HCD v20210416) simulated spectrum of DLGVLDFK/2.
    mz1 = np.asarray(
        [
            116.034,  # b1
            229.118,  # b2
            286.140,  # b3
            385.208,  # b4
            498.292,  # b5
            613.319,  # b6
            760.388,  # b7
            147.113,  # y1
            294.181,  # y2
            409.208,  # y3
            522.292,  # y4
            621.361,  # y5
            678.382,  # y6
            791.466,  # y7
        ]
    )
    intensity1 = np.asarray(
        [
            0.00000000,  # b1
            0.24194328,  # b2
            0.13076611,  # b3
            0.02920486,  # b4
            0.00316699,  # b5
            0.00426051,  # b6
            0.00131579,  # b7
            0.33024615,  # y1
            0.54129990,  # y2
            0.24971860,  # y3
            0.34601156,  # y4
            0.05075963,  # y5
            0.58027458,  # y6
            0.00585116,  # y7
        ]
    )
    spec1 = sus.MsmsSpectrum("HPYLLFDK", 453.750, 2, mz1, intensity1)
    # MS2PIP (HCD v20210416) simulated spectrum of GDLVLFDK/2.
    mz2 = np.asarray(
        [
            58.0287,  # b1
            173.056,  # b2
            286.140,  # b3
            385.208,  # b4
            498.292,  # b5
            645.361,  # b6
            760.388,  # b7
            147.113,  # y1
            262.140,  # y2
            409.208,  # y3
            522.292,  # y4
            621.361,  # y5
            734.445,  # y6
            849.472,  # y7
        ]
    )
    intensity2 = np.asarray(
        [
            0.00000000,  # b1
            0.12522728,  # b2
            0.18020111,  # b3
            0.04328780,  # b4
            0.00542208,  # b5
            0.00330758,  # b6
            0.00208561,  # b7
            0.26473886,  # y1
            0.30046007,  # y2
            0.56388106,  # y3
            0.49369887,  # y4
            0.43157844,  # y5
            0.20395883,  # y6
            0.00216236,  # y7
        ]
    )
    spec2 = sus.MsmsSpectrum("GDLVLFDK", 453.750, 2, mz2, intensity2)
    peak_matches = np.asarray(
        [[1, 1], [3, 4], [5, 5], [6, 6], [7, 7], [8, 8], [10, 9], [12, 12]]
    )
    ssm = spectrum.SpectrumSpectrumMatch(spec1, spec2, peak_matches)
    return sim.SpectrumSimilarityCalculator(ssm)


def test_cosine(all_match, no_match, partial_match):
    assert all_match.cosine() == pytest.approx(1.0)
    assert no_match.cosine() == pytest.approx(0.0)
    assert partial_match.cosine() == pytest.approx(0.44582117)


def test_spectral_contrast_angle(all_match, no_match, partial_match):
    assert all_match.spectral_contrast_angle() == pytest.approx(1.0)
    assert no_match.spectral_contrast_angle() == pytest.approx(0.0)
    assert partial_match.spectral_contrast_angle() == pytest.approx(0.29417655)


def test_hypergeometric_score(all_match, no_match):
    assert all_match.hypergeometric_score(
        min_mz=101, max_mz=1500, fragment_mz_tol=0.1
    ) == pytest.approx(0.0)
    assert (
            no_match.hypergeometric_score(
            min_mz=101, max_mz=1500, fragment_mz_tol=0.1
        )
            > 0.0
    )


def test_kendalltau(all_match, no_match):
    assert all_match.kendalltau() == pytest.approx(1.0)
    assert no_match.kendalltau() == pytest.approx(-1.0)


def test_ms_for_id_v1(all_match, no_match):
    assert all_match.ms_for_id_v1() > 10.0
    assert no_match.ms_for_id_v1() == pytest.approx(0.0)


def test_ms_for_id_v2(all_match, no_match):
    assert all_match.ms_for_id_v2() > 10.0
    assert no_match.ms_for_id_v2() == pytest.approx(0.0)


def test_manhattan(all_match, no_match):
    assert all_match.manhattan() == pytest.approx(0.0)
    assert np.isinf(no_match.manhattan())


def test_euclidean(all_match, no_match):
    assert all_match.euclidean() == pytest.approx(0.0)
    assert np.isinf(no_match.euclidean())
