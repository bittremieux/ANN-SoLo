import distutils.core

import Cython.Build
import numpy as np


distutils.core.setup(
    name='ANN-SoLo',
    ext_modules=Cython.Build.cythonize('*.pyx'),
    include_dirs=[np.get_include()],
)
