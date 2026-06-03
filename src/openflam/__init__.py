import os
import sys
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("openflam")
except PackageNotFoundError:  # package is not installed (e.g. running from source)
    __version__ = "unknown"

dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir_path)
from .hook import OpenFLAM