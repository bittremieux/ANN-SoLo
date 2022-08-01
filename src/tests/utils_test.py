import pytest

import numpy as np
import spectrum_utils.spectrum as sus

from ann_solo import spectrum
from ann_solo import utils


@pytest.fixture
def ssms_list_without_rescore():
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
    spec1 = sus.MsmsSpectrum("HPYLEDR", 465.227, 2, mz, intensity)
    intensity2 = np.copy(intensity)
    mz2 = np.copy(mz)
    ssms = []
    for index,value in enumerate(intensity2):
        intensity2[index] = value + index * 0.02
        spec2 = sus.MsmsSpectrum(
            "HPYLEDR", 465.227, 2, mz2, intensity2
        )
        ssm = spectrum.SpectrumSpectrumMatch(spec1, spec2, peak_matches)
        ssm.library_spectrum.peptide = "HPYLEDR"
        ssm.library_spectrum.is_decoy = False
        if index in [3, 4, 8, 9, 11]:
            ssm.library_spectrum.is_decoy = True
        ssms.append(ssm)

    return utils.score_ssms(
            ssms,
            0.01,
            None
        )


def test_score_ssms(
        ssms_list_without_rescore,
):
    #This function tests mainly the confidence assignment module
    #Integrated within the rescoring module
    #Which invokes the SpectrumSimilarityCalculator module
    q_value_expected = [
        0.33,
        0.33,
        0.33,
        0.6,
        0.6,
        0.6,
        0.6,
        0.66,
        0.66,
        0.71,
        0.71,
        0.86
    ]
    q_value_calculated = [ssm.q for ssm in ssms_list_without_rescore]
    assert q_value_calculated == q_value_expected

