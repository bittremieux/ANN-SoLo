from ann_solo import ann_solo


ann_solo.ann_solo('../../data/interim/iprg2012/human_yeast_targetdecoy.splib',
                  '../../data/external/iprg2012/iPRG2012.mgf',
                  'iPRG2012.mztab',
                  ['--precursor_tolerance_mass', '20',
                   '--precursor_tolerance_mode', 'ppm',
                   '--fragment_mz_tolerance', '0.02',
                   '--remove_precursor'])
