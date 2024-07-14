from typing import Dict, List

import numpy as np
import tqdm
from koinapy import Koina

from ann_solo.config import config

def get_predictions(peptides: List[str], precursor_charges: List[int],
                    collision_energies: List[int], decoy: bool = False) -> \
        Dict[str, np.ndarray]:
    """
    Predict spectra from the list of peptides.

    Parameters
    ----------
    peptides: List(str)
        List of peptides.
    precursor_charges: List(int)
        Synced list of precursor_charges.
    collision_energies: List(int)
        Synced list of collision_energies.
    decoy: bool = False
        Boolean precising whether the peptides are target or decoys.


    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary of spectra for each peptide, particularly containing
        intensities,  mz,  annotations for each spectrum.
    """

    batch_size = config.prosit_batch_size
    len_inputs = list(peptides)

    for i in tqdm.tqdm(range(0, len_inputs, batch_size),
            desc='Prosit peptides batch prediction:',
            unit=('decoy' if decoy else 'target') + ' peptides'):

        inputs = pd.DataFrame()
        inputs['peptide_sequences'] = np.array(peptides[i:i + batch_size])
        inputs['precursor_charges'] = np.array(precursor_charges[i:i + batch_size])
        inputs['collision_energies'] = np.array(collision_energies[i:i + batch_size])

        model = Koina(config.prosit_model_name, config.prosit_server_url)
        koina_predictions = model.predict(inputs)

        grouped_predictions = koina_predictions.groupby(
            ['peptide_sequences', 'precursor_charges', 'collision_energies']
        ).agg(
            {
                'intensities': list,
                'mz': list,
                'annotation': list
            }
        ).reset_index()

        predictions = grouped_predictions.to_dict(orient='list')

        yield predictions
