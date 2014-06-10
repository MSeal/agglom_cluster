import os
import shutil
from setuptools import setup, find_packages

VERSION = '1.0.2'

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

required = [req.strip() for req in read('requirements.txt').splitlines() if req.strip()]

setup(
    name="AgglomCluster",
    version=VERSION,
    author="Matthew Seal",
    author_email="mseal@opengov.com",
    description="Performs greedy agglomerative clustering on network-x graphs",
    packages=['agglomcluster'],
    long_description=readMD('README.md'),
    install_requires=required,
    license='LGPL 2.1',
    test_suite='tests',
    url='https://github.com/MSeal/agglom_cluster',
    download_url='https://github.com/MSeal/agglom_cluster/tarball/v' + VERSION,
    zip_safe=False,
    keywords=['network-x', 'data', 'graphs', 'clustering', 'agglomerative'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Utilities',
        'License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)',
        'Natural Language :: English',
        'Programming Language :: Python :: 2 :: Only'
    ]
)
