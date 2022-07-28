import os
import random

import numpy as np


def set_seeds(my_seed=42):
    os.environ['PYTHONHASHSEED'] = str(my_seed)
    random.seed(my_seed)
    np.random.seed(my_seed)
