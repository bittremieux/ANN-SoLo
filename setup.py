import distutils.core
import distutils.extension

import numpy as np
import Cython.Distutils


ext_module = distutils.extension.Extension('spectrum_match', ['spectrum_match.pyx'], language='c++',
                                           extra_compile_args=['-std=c++11'], extra_link_args=['-std=c++11'])

distutils.core.setup(
    name='Approximate nearest neighbors spectral library', cmdclass={'build_ext': Cython.Distutils.build_ext},
    ext_modules=[ext_module], include_dirs=[np.get_include()],
)
