from collections import Counter
import pytest
import unittest.mock

import numpy as np
from spectrum_utils.spectrum import MsmsSpectrum

from ann_solo.decoy_generator import shuffle_and_reposition



@pytest.fixture
def original_spectrum():
    # Original spectrum of the peptide
    # YYVC[Carbamidomethyl]TAPHC[Carbamidomethyl]GHR/4
    mz = np.asarray(
        [
            153.06617736816406,
            162.10269165039062,
            166.06126403808594,
            175.11920166015625,
            183.14962768554688,
            199.1807403564453,
            207.12403869628906,
            235.11917114257812,
            235.1443634033203,
            238.09422302246094,
            244.07455444335938,
            262.086181640625,
            271.66168212890625,
            289.6346740722656,
            295.1407775878906,
            299.1388854980469,
            300.1423034667969,
            315.1116027832031,
            323.1368713378906,
            327.1340026855469,
            328.136962890625,
            333.648193359375,
            334.14801025390625,
            338.64007568359375,
            339.1412048339844,
            352.1619873046875,
            369.19921875,
            373.16925048828125,
            382.67572021484375,
            390.2002258300781,
            417.6916809082031,
            426.203125,
            431.5727844238281,
            456.2585754394531,
            463.9232482910156,
            464.2568664550781,
            493.7205505371094,
            494.2242126464844,
            529.231201171875,
            530.2319946289062,
            543.7504272460938,
            548.2357788085938,
            607.3517456054688,
            666.289306640625,
            667.2920532226562,
            704.2771606445312,
            742.3851928710938,
            763.3427124023438,
            764.3442993164062
        ]
    )
    intensity = np.asarray(
        [
            1071.728645161472,
            809.8761808720759,
            2428.9034813888347,
            7298.865129098706,
            1540.6752234145301,
            909.5405590347412,
            1430.82773602237,
            4982.694473603787,
            3986.7699939105973,
            939.8304474708489,
            899.7238086554951,
            879.4582762056547,
            844.3561250321301,
            1710.608272503445,
            2625.697335308857,
            8358.5258852753,
            1187.6097721422761,
            1316.5550782011837,
            1473.718756304733,
            6245.454223180473,
            997.8472030321042,
            8904.243512812627,
            2542.688699674476,
            2019.7663575059619,
            866.9728045327622,
            894.8670977202303,
            3090.435495853796,
            1648.2794319433585,
            3597.6962173824763,
            1302.7624500757302,
            1150.575615003153,
            1725.2700797173006,
            3326.1228780186425,
            749.9283679348098,
            2103.1557841036115,
            1408.339116116147,
            1826.2734598401316,
            1224.996940121014,
            10000.0,
            1821.3624760291411,
            1061.0614164599915,
            1367.4616805039689,
            773.3596148661226,
            3559.0688362735455,
            871.025518035615,
            1167.3611787922255,
            987.5668633698173,
            5233.770418119861,
            2279.527723804536
        ]
    )

    return MsmsSpectrum("YYVC[Carbamidomethyl]TAPHC[Carbamidomethyl]GHR",
                                380.918862288762, 4, mz, intensity)


def test_shuffle_and_reposition(original_spectrum):
    # Get decoy
    with unittest.mock.patch(
        "ann_solo.config.config._namespace",
        {"fragment_tol_mass": 10, "fragment_tol_mode": 'ppm'},
    ) as _:
        decoy_spectrum = shuffle_and_reposition(original_spectrum)

    # Assert that the shapes of of arrays are the same
    assert np.shape(original_spectrum.mz) == np.shape(decoy_spectrum.mz)
    assert np.shape(original_spectrum.intensity) == np.shape(
        decoy_spectrum.intensity)

    # Assert that the range of decoy M/Zs is within range of original M/Zs
    assert np.min(decoy_spectrum.mz) >= np.min(original_spectrum.mz)
    assert np.max(decoy_spectrum.mz) <= np.max(original_spectrum.mz)

    # Assert that the sequences are not similar but are made of same residues
    assert decoy_spectrum.proforma.sequence != original_spectrum.proforma.sequence
    assert Counter(decoy_spectrum.proforma.sequence) == \
           Counter(original_spectrum.proforma.sequence)