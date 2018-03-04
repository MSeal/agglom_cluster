import os
import sys
from collections import defaultdict
from setuptools import setup, find_packages, Extension

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
BUILD_ARGS = defaultdict(lambda: ['-O3', '-g0'])
BUILD_ARGS['msvc'] = ['/EHsc']

def cleanup_pycs():
    file_tree = os.walk(os.path.join(BASE_DIR, 'hac'))
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

python_2 = sys.version_info[0] == 2
def read(fname):
    with open(fname, 'rU' if python_2 else 'r') as fhandle:
        return fhandle.read()

VERSION = read(os.path.join(BASE_DIR, 'VERSION')).strip()

def pandoc_read_md(fname):
    if 'PANDOC_PATH' not in os.environ:
        raise ImportError("No pandoc path to use")
    import pandoc
    pandoc.core.PANDOC_PATH = os.environ['PANDOC_PATH']
    doc = pandoc.Document()
    doc.markdown = read(fname)
    return doc.rst

def pypandoc_read_md(fname):
    import pypandoc
    os.environ.setdefault('PYPANDOC_PANDOC', os.environ['PANDOC_PATH'])
    return pypandoc.convert_text(read(fname), 'rst', format='md')

def read_md(fname):
    # Utility function to read the README file.
    full_fname = os.path.join(os.path.dirname(__file__), fname)

    try:
        return pandoc_read_md(full_fname)
    except (ImportError, AttributeError):
        try:
            return pypandoc_read_md(full_fname)
        except (ImportError, AttributeError):
            return read(fname)
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
    long_description=read_md('README.md'),
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
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3'
    ]
)
