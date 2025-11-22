"""
Setup script for zenowrapper with optional C++ extension building.

Set ZENOWRAPPER_SKIP_EXTENSION=1 to skip building the C++ extension.
This is useful for building documentation with Sphinx.

Example:
    ZENOWRAPPER_SKIP_EXTENSION=1 pip install -e .
"""

import os
import sys

# Check if we should skip the extension build
skip_extension = os.environ.get("ZENOWRAPPER_SKIP_EXTENSION", "").lower() in ("1", "true", "yes")

if skip_extension:
    # Use setuptools for a pure Python install
    from setuptools import setup
    from setuptools_scm import get_version

    # Write version file
    version = get_version(root=".", relative_to=__file__)
    version_file = "src/zenowrapper/_version.py"
    os.makedirs(os.path.dirname(version_file), exist_ok=True)
    with open(version_file, "w") as f:
        f.write(f'__version__ = "{version}"\n')

    setup(
        use_scm_version=True,
        setup_requires=["setuptools_scm"],
    )
else:
    # Use scikit-build-core for normal install with C++ extension
    # This will defer to pyproject.toml
    sys.exit(0)  # Let pyproject.toml handle it
