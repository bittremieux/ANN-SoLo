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
def all_match_top():
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
    return sim.SpectrumSimilarityCalculator(ssm, 5)


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
def no_match_top():
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
    return sim.SpectrumSimilarityCalculator(ssm, 5)


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


@pytest.fixture
def partial_match_top():
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
    return sim.SpectrumSimilarityCalculator(ssm, 5)


def test_cosine(
    all_match,
    all_match_top,
    no_match,
    no_match_top,
    partial_match,
    partial_match_top,
):
    assert all_match.cosine() == pytest.approx(1.0)
    assert all_match_top.cosine() == pytest.approx(1.0)
    assert no_match.cosine() == pytest.approx(0.0)
    assert no_match_top.cosine() == pytest.approx(0.0)
    assert partial_match.cosine() == pytest.approx(0.44582117)
    assert partial_match_top.cosine() == pytest.approx(0.85880862)


def test_n_matched_peaks(
    all_match,
    all_match_top,
    no_match,
    no_match_top,
    partial_match,
    partial_match_top,
):
    assert all_match.n_matched_peaks() == 12
    assert all_match_top.n_matched_peaks() == 5
    assert no_match.n_matched_peaks() == 0
    assert no_match_top.n_matched_peaks() == 0
    assert partial_match.n_matched_peaks() == 8
    assert partial_match_top.n_matched_peaks() == 4


def test_frac_n_peaks_query(
    all_match,
    all_match_top,
    no_match,
    no_match_top,
    partial_match,
    partial_match_top,
):
    assert all_match.frac_n_peaks_query() == pytest.approx(1.0)
    with pytest.raises(NotImplementedError):
        assert all_match_top.frac_n_peaks_query()
    assert no_match.frac_n_peaks_query() == pytest.approx(0.0)
    with pytest.raises(NotImplementedError):
        no_match_top.frac_n_peaks_query()
    assert partial_match.frac_n_peaks_query() == pytest.approx(8 / 14)
    with pytest.raises(NotImplementedError):
        partial_match_top.frac_n_peaks_query()


def test_frac_n_peaks_library(
    all_match,
    all_match_top,
    no_match,
    no_match_top,
    partial_match,
    partial_match_top,
):
    assert all_match.frac_n_peaks_library() == pytest.approx(1.0)
    assert all_match_top.frac_n_peaks_library() == pytest.approx(1.0)
    assert no_match.frac_n_peaks_library() == pytest.approx(0.0)
    assert no_match_top.frac_n_peaks_library() == pytest.approx(0.0)
    assert partial_match.frac_n_peaks_library() == pytest.approx(8 / 14)
    assert partial_match_top.frac_n_peaks_library() == pytest.approx(4 / 5)


def test_frac_intensity_query(
    all_match,
    all_match_top,
    no_match,
    no_match_top,
    partial_match,
    partial_match_top,
):
    assert all_match.frac_intensity_query() == pytest.approx(1.0)
    with pytest.raises(NotImplementedError):
        all_match_top.frac_intensity_query()
    assert no_match.frac_intensity_query() == pytest.approx(0.0)
    with pytest.raises(NotImplementedError):
        no_match_top.frac_intensity_query()
    assert partial_match.frac_intensity_query() == pytest.approx(0.45378598)
    with pytest.raises(NotImplementedError):
        partial_match_top.frac_intensity_query()


def test_frac_intensity_library(
    all_match,
    all_match_top,
    no_match,
    no_match_top,
    partial_match,
    partial_match_top,
):
    assert all_match.frac_intensity_library() == pytest.approx(1.0)
    assert all_match_top.frac_intensity_library() == pytest.approx(1.0)
    assert no_match.frac_intensity_library() == pytest.approx(0.0)
    assert no_match_top.frac_intensity_library() == pytest.approx(0.0)
    assert partial_match.frac_intensity_library() == pytest.approx(0.75759018)
    assert partial_match_top.frac_intensity_library() == pytest.approx(
        0.85374497
    )


def test_mean_squared_error(
    all_match,
    all_match_top,
    no_match,
    no_match_top,
    partial_match,
    partial_match_top,
):
    assert all_match.mean_squared_error("mz") == pytest.approx(0.0)
    assert all_match_top.mean_squared_error("mz") == pytest.approx(0.0)
    assert np.isinf(no_match.mean_squared_error("mz"))
    assert np.isinf(no_match_top.mean_squared_error("mz"))
    # The following two tests are not super useful because exact fragment m/z
    # values are used for both spectra.
    assert partial_match.mean_squared_error("mz") == pytest.approx(0.0)
    assert partial_match_top.mean_squared_error("mz") == pytest.approx(0.0)
    assert all_match.mean_squared_error("intensity") == pytest.approx(0.0)
    assert all_match_top.mean_squared_error("intensity") == pytest.approx(0.0)
    assert np.isinf(no_match.mean_squared_error("intensity"))
    assert np.isinf(no_match_top.mean_squared_error("intensity"))
    assert partial_match.mean_squared_error("intensity") == pytest.approx(
        0.03405894
    )
    assert partial_match_top.mean_squared_error("intensity") == pytest.approx(
        0.06745593
    )
    with pytest.raises(ValueError):
        all_match_top.mean_squared_error("unknown")


def test_spectral_contrast_angle(
    all_match,
    all_match_top,
    no_match,
    no_match_top,
    partial_match,
    partial_match_top,
):
    assert all_match.spectral_contrast_angle() == pytest.approx(1.0)
    assert all_match_top.spectral_contrast_angle() == pytest.approx(1.0)
    assert no_match.spectral_contrast_angle() == pytest.approx(0.0)
    assert no_match_top.spectral_contrast_angle() == pytest.approx(0.0)
    assert partial_match.spectral_contrast_angle() == pytest.approx(0.29417655)
    assert partial_match_top.spectral_contrast_angle() == pytest.approx(
        0.65758974
    )


def test_hypergeometric_score(
    all_match,
    all_match_top,
    no_match,
    no_match_top,
    partial_match,
    partial_match_top,
):
    params = dict(min_mz=101, max_mz=1500, fragment_mz_tol=0.1)
    assert all_match.hypergeometric_score(**params) == pytest.approx(100.0)
    assert all_match_top.hypergeometric_score(**params) == pytest.approx(100.0)
    assert no_match.hypergeometric_score(**params) == pytest.approx(4.27409242)
    assert no_match_top.hypergeometric_score(**params) == pytest.approx(
        6.32786559
    )
    assert partial_match.hypergeometric_score(**params) == pytest.approx(
        57.90893056
    )
    assert partial_match_top.hypergeometric_score(**params) == pytest.approx(
        42.94264115
    )


def test_kendalltau(
    all_match,
    all_match_top,
    no_match,
    no_match_top,
    partial_match,
    partial_match_top,
):
    assert all_match.kendalltau() == pytest.approx(19.29406731)
    assert all_match_top.kendalltau() == pytest.approx(4.09434456)
    assert no_match.kendalltau() == pytest.approx(0.0)
    assert no_match_top.kendalltau() == pytest.approx(0.0)
    assert partial_match.kendalltau() == pytest.approx(4.25896654)
    assert partial_match_top.kendalltau() == pytest.approx(0.0)


def test_ms_for_id_v1(
    all_match,
    all_match_top,
    no_match,
    no_match_top,
    partial_match,
    partial_match_top,
):
    assert all_match.ms_for_id_v1() == pytest.approx(1000.0)
    assert all_match_top.ms_for_id_v1() == pytest.approx(1000.0)
    assert no_match.ms_for_id_v1() == pytest.approx(0.0)
    assert no_match_top.ms_for_id_v1() == pytest.approx(0.0)
    assert partial_match.ms_for_id_v1() == pytest.approx(21.03216848)
    assert partial_match_top.ms_for_id_v1() == pytest.approx(10.48956478)


def test_ms_for_id_v2(
    all_match,
    all_match_top,
    no_match,
    no_match_top,
    partial_match,
    partial_match_top,
):
    assert all_match.ms_for_id_v2() == pytest.approx(154.45107128)
    with pytest.raises(NotImplementedError):
        all_match_top.ms_for_id_v2()
    assert no_match.ms_for_id_v2() == pytest.approx(0.0)
    with pytest.raises(NotImplementedError):
        no_match_top.ms_for_id_v2()
    assert partial_match.ms_for_id_v2() == pytest.approx(30.03222119)
    with pytest.raises(NotImplementedError):
        partial_match_top.ms_for_id_v2()


def test_manhattan(
    all_match,
    all_match_top,
    no_match,
    no_match_top,
    partial_match,
    partial_match_top,
):
    assert all_match.manhattan() == pytest.approx(0.0)
    with pytest.raises(NotImplementedError):
        all_match_top.manhattan()
    assert np.isinf(no_match.manhattan())
    with pytest.raises(NotImplementedError):
        no_match_top.manhattan()
    assert partial_match.manhattan() == pytest.approx(2.98346427)
    with pytest.raises(NotImplementedError):
        partial_match_top.manhattan()


def test_euclidean(
    all_match,
    all_match_top,
    no_match,
    no_match_top,
    partial_match,
    partial_match_top,
):
    assert all_match.euclidean() == pytest.approx(0.0)
    with pytest.raises(NotImplementedError):
        all_match_top.euclidean()
    assert np.isinf(no_match.euclidean())
    with pytest.raises(NotImplementedError):
        no_match_top.euclidean()
    assert partial_match.euclidean() == pytest.approx(1.05278566)
    with pytest.raises(NotImplementedError):
        partial_match_top.euclidean()


def test_chebyshev(
    all_match,
    all_match_top,
    no_match,
    no_match_top,
    partial_match,
    partial_match_top,
):
    assert all_match.chebyshev() == pytest.approx(0.0)
    with pytest.raises(NotImplementedError):
        all_match_top.chebyshev()
    assert np.isinf(no_match.chebyshev())
    with pytest.raises(NotImplementedError):
        no_match_top.chebyshev()
    assert partial_match.chebyshev() == pytest.approx(0.5802746)
    with pytest.raises(NotImplementedError):
        partial_match_top.chebyshev()


def test_pearsonr(
    all_match,
    all_match_top,
    no_match,
    no_match_top,
    partial_match,
    partial_match_top,
):
    assert all_match.pearsonr() == pytest.approx(1.0)
    assert all_match_top.pearsonr() == pytest.approx(1.0)
    assert no_match.pearsonr() == pytest.approx(0.0)
    assert no_match_top.pearsonr() == pytest.approx(0.0)
    assert partial_match.pearsonr() == pytest.approx(0.69570652)
    assert partial_match_top.pearsonr() == pytest.approx(0.24177300)


def test_spearmanr(
    all_match,
    all_match_top,
    no_match,
    no_match_top,
    partial_match,
    partial_match_top,
):
    assert all_match.spearmanr() == pytest.approx(1.0)
    assert all_match_top.spearmanr() == pytest.approx(1.0)
    assert no_match.spearmanr() == pytest.approx(0.0)
    assert no_match_top.spearmanr() == pytest.approx(0.0)
    assert partial_match.spearmanr() == pytest.approx(0.59933680)
    assert partial_match_top.spearmanr() == pytest.approx(0.19999999)


def test_braycurtis(
    all_match,
    all_match_top,
    no_match,
    no_match_top,
    partial_match,
    partial_match_top,
):
    assert all_match.braycurtis() == pytest.approx(0.0)
    with pytest.raises(NotImplementedError):
        all_match_top.braycurtis()
    assert no_match.braycurtis() == pytest.approx(1.0)
    with pytest.raises(NotImplementedError):
        no_match_top.braycurtis()
    assert partial_match.braycurtis() == pytest.approx(0.58102504)
    with pytest.raises(NotImplementedError):
        partial_match_top.braycurtis()


def test_canberra(
    all_match,
    all_match_top,
    no_match,
    no_match_top,
    partial_match,
    partial_match_top,
):
    assert all_match.canberra() == pytest.approx(0.0)
    with pytest.raises(NotImplementedError):
        all_match_top.canberra()
    assert np.isinf(no_match.canberra())
    with pytest.raises(NotImplementedError):
        no_match_top.canberra()
    assert partial_match.canberra() == pytest.approx(12.30376030)
    with pytest.raises(NotImplementedError):
        partial_match_top.canberra()


def test_ruzicka(
    all_match,
    all_match_top,
    no_match,
    no_match_top,
    partial_match,
    partial_match_top,
):
    assert all_match.ruzicka() == pytest.approx(1.0)
    with pytest.raises(NotImplementedError):
        all_match_top.ruzicka()
    assert no_match.ruzicka() == pytest.approx(0.0)
    with pytest.raises(NotImplementedError):
        no_match_top.ruzicka()
    assert partial_match.ruzicka() == pytest.approx(0.26500210)
    with pytest.raises(NotImplementedError):
        partial_match_top.ruzicka()


def test_scribe_fragment_acc(
    all_match,
    all_match_top,
    no_match,
    no_match_top,
    partial_match,
    partial_match_top,
):
    assert all_match.scribe_fragment_acc() == pytest.approx(10.0)
    assert all_match_top.scribe_fragment_acc() == pytest.approx(10.0)
    assert no_match.scribe_fragment_acc() == pytest.approx(0.0)
    assert no_match_top.scribe_fragment_acc() == pytest.approx(0.0)
    assert partial_match.scribe_fragment_acc() == pytest.approx(0.86739458)
    assert partial_match_top.scribe_fragment_acc() == pytest.approx(1.02137350)


def test_entropy(
    all_match,
    all_match_top,
    no_match,
    no_match_top,
    partial_match,
    partial_match_top,
):
    # Unweighted entropy.
    assert all_match.entropy(False) == pytest.approx(1.0)
    with pytest.raises(NotImplementedError):
        all_match_top.entropy(False)
    assert no_match.entropy(False) == pytest.approx(0.0)
    with pytest.raises(NotImplementedError):
        no_match_top.entropy(False)
    assert partial_match.entropy(False) == pytest.approx(0.53600209)
    with pytest.raises(NotImplementedError):
        partial_match_top.entropy(False)
    # Weighted entropy.
    assert all_match.entropy(True) == pytest.approx(1.0)
    with pytest.raises(NotImplementedError):
        all_match_top.entropy(True)
    assert no_match.entropy(True) == pytest.approx(0.0)
    with pytest.raises(NotImplementedError):
        no_match_top.entropy(True)
    assert partial_match.entropy(True) == pytest.approx(0.59836031)
    with pytest.raises(NotImplementedError):
        partial_match_top.entropy(True)
