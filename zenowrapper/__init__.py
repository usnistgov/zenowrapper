"""
ZENOWrapper
This package wraps the package ZENO in a compatible way with MDAnalysis.
"""

# Add imports here
from importlib.metadata import version

__version__ = version("zenowrapper")
from .analysis.build import zenolib