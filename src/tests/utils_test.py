import unittest.mock

import numpy as np
import spectrum_utils.spectrum as sus

from ann_solo import spectrum
from ann_solo import utils


def test_score_ssms():
    # This function tests mainly the confidence assignment module integrated
    # within the rescoring module which invokes the
    # SpectrumSimilarityCalculator module.
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
    peak_matches = np.asarray([(i, i) for i in range(len(mz))])
    intensity /= np.linalg.norm(intensity)
    spec1 = sus.MsmsSpectrum("HPYLEDR", 465.227, 2, mz, intensity)
    ssms = []
    for i in range(12):
        intensity_new = np.copy(intensity)
        intensity_new[-1] *= 1 + i / 100
        intensity_new /= np.linalg.norm(intensity_new)
        spec2 = sus.MsmsSpectrum("HPYLEDR", 465.227, 2, mz, intensity_new)
        spec2.peptide = "HPYLEDR"
        spec2.is_decoy = i in [3, 4, 8, 9, 11]
        ssms.append(spectrum.SpectrumSpectrumMatch(spec1, spec2, peak_matches))

    # Target SSMs get actual q-values, decoy SSMs get NaN q-value.
    q_values = [
        1 / 3,
        1 / 3,
        1 / 3,
        np.nan,
        np.nan,
        1 / 2,
        1 / 2,
        1 / 2,
        np.nan,
        np.nan,
        5 / 7,
        np.nan,
    ]
    with unittest.mock.patch(
        "ann_solo.config.config._namespace",
        {"min_mz": 11, "max_mz": 2010, "bin_size": 0.04},
    ) as _:
        q_values_calc = [ssm.q for ssm in utils.score_ssms(ssms, 0.33, None)]
    np.testing.assert_array_equal(q_values, q_values_calc)
