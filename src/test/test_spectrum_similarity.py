import unittest

import numpy as np
import spectrum_utils.spectrum as sus

from ann_solo import spectrum
from ann_solo import spectrum_similarity as sim


class TestSpectrumSimilarityMethods(unittest.TestCase):

    def setUp(self):
        # Create first SpectrumSimilarityFactory object
        # All matching peaks
        n_peaks = 10
        mz1 = np.linspace(0, 10, n_peaks)
        int1 = spectrum._norm_intensity(np.random.exponential(1.0, n_peaks))
        spec1 = sus.MsmsSpectrum("spectrum1", 500, 2, mz1, int1)
        spec2 = sus.MsmsSpectrum("spectrum2", 500, 2, np.copy(mz1),
                                 np.copy(int1))
        peak_matches = np.asarray([(i, i) for i in range(n_peaks)])
        ssm = spectrum.SpectrumSpectrumMatch(spec1, spec2, peak_matches)
        self.sim_factory_1 = sim.SpectrumSimilarityFactory(ssm)

        # Create second SpectrumSimilarityFactory object
        # No matching peaks
        mz2 = np.linspace(0.5, 10.5, n_peaks)
        int2 = spectrum._norm_intensity(np.random.exponential(1.0, n_peaks))
        spec2 = sus.MsmsSpectrum("spectrum2", 500, 2, mz2, int2)
        peak_matches = np.asarray([])
        ssm = spectrum.SpectrumSpectrumMatch(spec1, spec2, peak_matches)
        self.sim_factory_2 = sim.SpectrumSimilarityFactory(ssm)

    def tearDown(self):
        pass

    def test_cosine(self):
        # Test value is within a range
        self.assertTrue(0<=self.sim_factory_1.cosine()<=1)
        self.assertTrue(0<=self.sim_factory_2.cosine()<=1)
        #Test identical spectra
        self.assertAlmostEqual(self.sim_factory_1.cosine(), 1.0)
        # Test no matching spectra
        self.assertAlmostEqual(self.sim_factory_2.cosine(), 0.0)

    def test_spectral_contrast_angle(self):
        # Test value is within a range
        self.assertTrue(0<=self.sim_factory_1.spectral_contrast_angle()<=1)
        self.assertTrue(0<=self.sim_factory_2.spectral_contrast_angle()<=1)
        #Test identical spectra
        self.assertAlmostEqual(self.sim_factory_1.spectral_contrast_angle(),
                               1.0)
        # Test no matching spectra
        self.assertAlmostEqual(self.sim_factory_2.spectral_contrast_angle(),
                               0.0)

    def test_hypergeometric_score(self):
        # Test value is within a range
        self.assertTrue(0<=self.sim_factory_1.hypergeometric_score(min_mz=101,
                                                max_mz=1500,bin_size=0.1)<=1)
        self.assertTrue(0<=self.sim_factory_2.hypergeometric_score(min_mz=101,
                                                max_mz=1500,bin_size=0.1)<=1)
        #Test identical spectra
        self.assertAlmostEqual(np.around(
                                self.sim_factory_1.hypergeometric_score(
                                    min_mz=101, max_mz=1500,bin_size=0.1), 3),
                            0.0)
        # Test no matching spectra
        self.assertAlmostEqual(np.around(
                                self.sim_factory_2.hypergeometric_score(
                                    min_mz=101, max_mz=1500,bin_size=0.1), 3),
                            0.007)

    def test_kendalltau(self):
        # Test value is within a range
        self.assertTrue(-1<=self.sim_factory_1.kendalltau()<=1)
        self.assertTrue(-1<=self.sim_factory_2.kendalltau()<=1)
        #Test identical spectra
        self.assertAlmostEqual(self.sim_factory_1.kendalltau(),
                               1.0)
        # Test no matching spectra
        self.assertAlmostEqual(self.sim_factory_2.kendalltau(),
                               -1.0)

    def test_ms_for_id_v1(self):
        # Test value is within a range
        self.assertTrue(self.sim_factory_1.ms_for_id_v1()>=0)
        self.assertTrue(self.sim_factory_2.ms_for_id_v1()>=0)
        #Test identical spectra
        self.assertAlmostEqual(self.sim_factory_1.ms_for_id_v1(),
                               819200.0)
        # Test no matching spectra
        self.assertAlmostEqual(self.sim_factory_2.ms_for_id_v1(),
                               0.0)

    def test_ms_for_id_v2(self):
        # Test value is within a range
        self.assertTrue(self.sim_factory_1.ms_for_id_v2()>=0)
        # Test no matching spectra
        self.assertAlmostEqual(self.sim_factory_2.ms_for_id_v2(),
                               0.0)

    def test_manhattan(self):
        # Test value is within a range
        self.assertTrue(self.sim_factory_1.manhattan()>=0)
        #Test identical spectra
        self.assertAlmostEqual(self.sim_factory_1.manhattan(),
                               0.0)


    def test_euclidean(self):
        # Test value is within a range
        self.assertTrue(self.sim_factory_1.euclidean()>=0)
        #Test identical spectra
        self.assertAlmostEqual(self.sim_factory_1.euclidean(),
                               0.0)
