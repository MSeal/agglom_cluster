import os
import sys
from collections import defaultdict
from setuptools import setup, find_packages, Extension

VERSION = '2.0.1'
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
BUILD_ARGS = defaultdict(lambda: ['-O3', '-g0'])
BUILD_ARGS['msvc'] = ['/EHsc']

def cleanup_pycs():
    file_tree = os.walk(os.path.join(BASE_DIR, 'hunspell'))
    to_delete = []
    for root, directory, file_list in file_tree:
        if len(file_list):
            for file_name in file_list:
                if file_name.endswith(".pyc"):
                    to_delete.append(os.path.join(root, file_name))
    for file_path in to_delete:
        try:
            os.remove(file_path)
        except:
            pass

def read(fname):
    with open(fname) as fhandle:
            return fhandle.read()

def readMD(fname):
    # Utility function to read the README file.
    full_fname = os.path.join(os.path.dirname(__file__), fname)
    if 'PANDOC_PATH' in os.environ:
        import pandoc
        pandoc.core.PANDOC_PATH = os.environ['PANDOC_PATH']
        doc = pandoc.Document()
        with open(full_fname) as fhandle:
            doc.markdown = fhandle.read()
        return doc.rst
    else:
        return read(fname)

profiling = '--profile' in sys.argv or '-p' in sys.argv
linetrace = '--linetrace' in sys.argv or '-l' in sys.argv
building = 'build_ext' in sys.argv

datatypes = ['*.aff', '*.dic', '*.pxd', '*.pyx', '*.pyd', '*.pxd', '*.so', '*.lib', '*hpp']
packages = find_packages(exclude=['*.tests', '*.tests.*', 'tests.*', 'tests'])
required = [req.strip() for req in read('requirements.txt').splitlines() if req.strip()]
package_data = {'' : datatypes}

if building:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
    ext_modules = cythonize([
        Extension(
            'hac.cluster',
            [os.path.join('hac', 'cluster.pyx')],
            language='c'
        )
    ], force=True)
else:
    from setuptools.command.build_ext import build_ext
    ext_modules = [
        Extension(
            'hac.cluster',
            [os.path.join('hac', 'cluster.c')],
            language='c'
        )
    ]

class build_ext_compiler_check(build_ext):
    def build_extensions(self):
        compiler = self.compiler.compiler_type
        args = BUILD_ARGS[compiler]
        for ext in self.extensions:
            ext.extra_compile_args = args
        build_ext.build_extensions(self)

    def run(self):
        cleanup_pycs()
        build_ext.run(self)

setup(
    name="AgglomCluster",
    version=VERSION,
    author="Matthew Seal",
    author_email="mseal@opengov.com",
    description="Performs greedy agglomerative clustering on network-x graphs",
    packages=packages,
    long_description=readMD('README.md'),
    ext_modules=ext_modules,
    install_requires=required,
    cmdclass={ 'build_ext': build_ext_compiler_check },
    license='LGPL 2.1',
    test_suite='tests',
    url='https://github.com/MSeal/agglom_cluster',
    download_url='https://github.com/MSeal/agglom_cluster/tarball/v' + VERSION,
    package_data=package_data,
    zip_safe=False,
    keywords=['network-x', 'data', 'graphs', 'clustering', 'agglomerative'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Topic :: Utilities',
        'License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)',
        'Natural Language :: English',
        'Programming Language :: Python :: 2 :: Only'
    ]
)
