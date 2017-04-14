import distutils.core
import distutils.extension

import Cython.Distutils
import numpy as np

compile_args = ['-O3', '-march=native', '-ffast-math', '-std=c++14', '-fopenmp']
ext_module = distutils.extension.Extension('spectrum_match', ['spectrum_match.pyx', 'SpectrumMatch.cpp'],
                                           language='c++', extra_compile_args=compile_args, extra_link_args=compile_args)

distutils.core.setup(
    name='ANN-SoLo', cmdclass={'build_ext': Cython.Distutils.build_ext},
    ext_modules=[ext_module], include_dirs=[np.get_include()],
)
