"""
ZENOWrapper
This package wraps the package ZENO in a compatible way with MDAnalysis.
"""

# Add imports here
from importlib.metadata import version

__version__ = version("zenowrapper")

from .zenowrapper_ext import add
from zenowrapper.analysis.main import Property, ZenoWrapper
#import .zenowrapper_ext.zenolib as zenolib