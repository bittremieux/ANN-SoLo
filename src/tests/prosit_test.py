import unittest.mock

import numpy as np

from ann_solo.prosit import get_predictions


def test_get_predictions():
    # This function tests the spectrum prediction of a peptide using prosit.

    peptide_sequences = ["AAAAAKAK"]
    precursor_charges = [1]
    collision_energies = [25]

    _intensities = np.asarray(
        [
            0.06730208545923233,
            0.517386794090271,
            0.16177022457122803,
            0.40451985597610474,
            0.40442216396331787,
            0.23247307538986206,
            0.637698769569397,
            0.3326859176158905,
            0.940611720085144,
            1.0
        ]
    )

    _mz = np.asarray(
        [
            218.14991760253906,
            346.244873046875,
            214.11862182617188,
            417.281982421875,
            285.1557312011719,
            488.3191223144531,
            356.1928405761719,
            559.356201171875,
            484.2878112792969,
            555.324951171875
        ]
    )

    _annotation = np.asarray(
        [
            b'y2+1',
            b'y3+1',
            b'b3+1',
            b'y4+1',
            b'b4+1',
            b'y5+1',
            b'b5+1',
            b'y6+1',
            b'b6+1',
            b'b7+1'
        ]
    )


    with unittest.mock.patch(
        "ann_solo.config.config._namespace",
        {"prosit_batch_size": 1000,
         "prosit_server_url": "koina.proteomicsdb.org:443",
         "prosit_model_name": "Prosit_2020_intensity_HCD"},
    ) as _:
        predictions = get_predictions(
            peptide_sequences, precursor_charges, collision_energies)

    np.testing.assert_array_equal(_intensities,
                                  np.asarray(predictions['intensities'][0]))
    np.testing.assert_array_equal(_mz,
                                  np.asarray(predictions['mz'][0]))
    np.testing.assert_array_equal(_annotation,
                                  np.asarray(predictions['annotation'][0]))


