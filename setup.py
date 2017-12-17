from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

# import os
# from os.path import join
# from Cython.Build import cythonize


# def configuration(parent_package='', top_path=None):
#     from numpy.distutils.misc_util import Configuration

#     config = Configuration('deep_walk', parent_package, top_path)

#     # config.add_subpackage('tests')

#     # Section LibSVM

#     # we compile both libsvm and libsvm_sparse
#     config.add_library('random_walk-skl',
#                        sources=[join('src', 'static_graph.c')],
#                        # depends=[join('src', 'libsvm', 'svm.cpp'),
#                        #          join('src', 'libsvm', 'svm.h')],
#                        # Force C++ linking in case gcc is picked up instead
#                        # of g++ under windows with some versions of MinGW
#                        # extra_link_args=['-lstdc++'],
#                        )

#     libsvm_sources = [cythonize('random_walk.pyx')]
#     libsvm_depends = [join('src', 'static_graph.c')]

#     config.add_extension('random_walk',
#                          sources=libsvm_sources,
#                          include_dirs=[numpy.get_include(), 'src'],
#                          libraries=['random_walk-skl'],
#                          depends=libsvm_depends)
#     return config


if __name__ == '__main__':
    # from numpy.distutils.core import setup
    # setup(configuration=configuration)
    exts = [Extension(name='word2vec',
                      sources=['word2vec.pyx',
                               'src/word2vec.c']),
            Extension(name='random_walk',
                      sources=['random_walk.pyx',
                               'src/static_graph.c',
                               'src/kthread.c'])]

    setup(name='MyProject',
          cmdclass={'build_ext': build_ext},
          ext_modules=exts,
          include_dirs=[numpy.get_include(), 'src'])

# from distutils.core import setup
# from Cython.Build import cythonize


# setup(
#     ext_modules=cythonize('static_graph.pyx')
# )
