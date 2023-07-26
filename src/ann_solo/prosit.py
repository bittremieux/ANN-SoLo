from typing import Dict, List

import numpy as np
import tqdm
import tritonclient.grpc as grpcclient

from ann_solo.config import config

def get_predictions(peptides: List(str), precursor_charges: List(int),
                    collision_energies: List(int), decoy: bool = False) -> \
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
        Boolean precising whether the peptides are genuine or decoys.


    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary of spectra for each peptide, particularly containing
        intensities,  mz,  annotations for each spectrum.
    """
    nptype_convert = {
        np.dtype('float32'): 'FP32',
        np.dtype('O'): 'BYTES',
        np.dtype('int16'): 'INT16',
        np.dtype('int32'): 'INT32',
        np.dtype('int64'): 'INT64',
    }

    server_url = config.prosit_server_url
    model_name = config.prosit_model_name
    batch_size = config.prosit_batch_size
    inputs = {
        'peptide_sequences': np.array(peptides, dtype=np.dtype("O")).reshape
            ([tot ,1]),
        'precursor_charges': np.array(precursor_charges, dtype=np.dtype("int32")).reshape
            ([tot ,1]),
        'collision_energies': np.array(collision_energies, dtype=np.dtype("float32")).reshape
            ([tot ,1]),
    }
    outputs = [ 'intensities',  'mz',  'annotation', ]

    triton_client = grpcclient.InferenceServerClient(url=server_url, ssl=True)

    koina_outputs = []
    for name in outputs:
        koina_outputs.append(grpcclient.InferRequestedOutput(name))

    len_inputs = list(inputs.values())[0].shape[0]
    for i in tqdm.tqdm(range(0, len_inputs, batch_size),
            desc='Prosit peptides batch prediction:',
            unit=('decoy' if decoy else 'genuine') + ' peptides'):
        predictions = {name: [] for name in outputs}
        if len_inputs < i+ batch_size:
            current_batchsize = len_inputs
        else:
            current_batchsize = batch_size

        koina_inputs = []
        for iname, iarr in inputs.items():
            koina_inputs.append(
                grpcclient.InferInput(iname, [current_batchsize, 1],
                                      nptype_convert[iarr.dtype])
            )
            koina_inputs[-1].set_data_from_numpy(iarr[i:i + current_batchsize])

        prediction = triton_client.infer(model_name, inputs=koina_inputs,
                                         outputs=koina_outputs)

        for name in outputs:
            predictions[name].append(prediction.as_numpy(name))

        yield predictions
