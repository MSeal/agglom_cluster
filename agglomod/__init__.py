import os
import glob
# Hack to get setup tools to correctly include all python files
__all__ = [module.split(os.path.sep)[-1].split('.')[0] 
           for module in glob.glob(os.path.dirname(__file__)+'/*.py') if '__' not in module]
__all__.extend([module.split(os.path.sep)[-2] 
                for module in glob.glob(os.path.dirname(__file__)+'/*/__init__.py')])
del glob
del os

from agglomod import NewmanGreedy