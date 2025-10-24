ZENOWrapper
==============================
[//]: # (Badges)

| **Latest release** | [![Last release tag][badge_release]][url_latest_release] ![GitHub commits since latest release (by date) for a branch][badge_commits_since]  |
| :----------------- | :------- |
| **Status**         | [![GH Actions Status][badge_actions]][url_actions] [![codecov][badge_codecov]][url_codecov] |
| **Community**      | [![License: NIST][badge_license]][nist-open]  [![Powered by MDAnalysis][badge_mda]][url_mda]|

[badge_actions]: https://github.com/usnistgov/zenowrapper/actions/workflows/gh-ci.yaml/badge.svg
[badge_codecov]: https://codecov.io/gh/usnistgov/zenowrapper/branch/main/graph/badge.svg
[badge_commits_since]: https://img.shields.io/github/commits-since/usnistgov/zenowrapper/latest
[badge_docs]: https://readthedocs.org/projects/zenowrapper/badge/?version=latest
[badge_license]: https://img.shields.io/badge/License-GPLv2-blue.svg
[badge_mda]: https://img.shields.io/badge/powered%20by-MDAnalysis-orange.svg?logoWidth=16&logo=data:image/x-icon;base64,AAABAAEAEBAAAAEAIAAoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJD+XwCY/fEAkf3uAJf97wGT/a+HfHaoiIWE7n9/f+6Hh4fvgICAjwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACT/yYAlP//AJ///wCg//8JjvOchXly1oaGhv+Ghob/j4+P/39/f3IAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJH8aQCY/8wAkv2kfY+elJ6al/yVlZX7iIiI8H9/f7h/f38UAAAAAAAAAAAAAAAAAAAAAAAAAAB/f38egYF/noqAebF8gYaagnx3oFpUUtZpaWr/WFhY8zo6OmT///8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgICAn46Ojv+Hh4b/jouJ/4iGhfcAAADnAAAA/wAAAP8AAADIAAAAAwCj/zIAnf2VAJD/PAAAAAAAAAAAAAAAAICAgNGHh4f/gICA/4SEhP+Xl5f/AwMD/wAAAP8AAAD/AAAA/wAAAB8Aov9/ALr//wCS/Z0AAAAAAAAAAAAAAACBgYGOjo6O/4mJif+Pj4//iYmJ/wAAAOAAAAD+AAAA/wAAAP8AAABhAP7+FgCi/38Axf4fAAAAAAAAAAAAAAAAiIiID4GBgYKCgoKogoB+fYSEgZhgYGDZXl5e/m9vb/9ISEjpEBAQxw8AAFQAAAAAAAAANQAAADcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAjo6Mb5iYmP+cnJz/jY2N95CQkO4pKSn/AAAA7gAAAP0AAAD7AAAAhgAAAAEAAAAAAAAAAACL/gsAkv2uAJX/QQAAAAB9fX3egoKC/4CAgP+NjY3/c3Nz+wAAAP8AAAD/AAAA/wAAAPUAAAAcAAAAAAAAAAAAnP4NAJL9rgCR/0YAAAAAfX19w4ODg/98fHz/i4uL/4qKivwAAAD/AAAA/wAAAP8AAAD1AAAAGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALGxsVyqqqr/mpqa/6mpqf9KSUn/AAAA5QAAAPkAAAD5AAAAhQAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADkUFBSuZ2dn/3V1df8uLi7bAAAATgBGfyQAAAA2AAAAMwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB0AAADoAAAA/wAAAP8AAAD/AAAAWgC3/2AAnv3eAJ/+dgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9AAAA/wAAAP8AAAD/AAAA/wAKDzEAnP3WAKn//wCS/OgAf/8MAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIQAAANwAAADtAAAA7QAAAMAAABUMAJn9gwCe/e0Aj/2LAP//AQAAAAAAAAAA
[badge_release]: https://img.shields.io/github/release-pre/usnistgov/zenowrapper.svg
[url_actions]: https://github.com/usnistgov/zenowrapper/actions?query=branch%3Amain+workflow%3Agh-ci
[url_codecov]: https://codecov.io/gh/usnistgov/zenowrapper/branch/main
[url_docs]: https://zenowrapper.readthedocs.io/en/latest/?badge=latest
[url_latest_release]: https://github.com/usnistgov/zenowrapper/releases
[url_mda]: https://www.mdanalysis.org

## [NIST Disclaimer][nist-disclaimer]

Certain commercial equipment, instruments, or materials are identified in this paper to foster understanding. Such identification does not imply recommendation or endorsement by the National Institute of Standards and Technology, nor does it imply that the materials or equipment identified are necessarily the best available for the purpose.

## Summary

This package provides a concise Python wrapper around the ZENO computation engine with an MDAnalysis‑friendly API. It automates preparing ZENO inputs, running ZENO computations, and returning results as Python objects (accepting MDAnalysis Universe/AtomGroup objects and trajectory frames), enabling transport and hydrodynamic analyses to be run directly within MDAnalysis workflows for reproducible analysis and scripting. See the ZenoWrapper API documentation for details.

ZENOWrapper is bound by a [Code of Conduct](https://github.com/usnistgov/zenowrapper/blob/main/CODE_OF_CONDUCT.md).

### [Documentation][docs4nist]

### Dependencies

This package is tested for python 3.10+ on all Windows, MacOS, and Linux systems.
[Scipy][scipy] must be installed before installation.

### Installation

To build ZENOWrapper from source, we highly recommend using virtual environments.
Below we provide instructions for `pip`.

#### Install ZENO

Follow the [installation instructions for ZENO](https://zeno.nist.gov/Compilation.html).
Then set an environmental variable for the path: `ZENOPATH='/Users/jennifer.clark/bin/ZENO'`

#### Download

``git clone https://github.com/usnistgov/zenowrapper``

#### User Install from Source

To build the package from source, run:

```
pip install .
```

If you want to create a development environment, install
the dependencies required for tests and docs with:

```
pip install ".[test,doc]"
```

#### Developer Install from Source

To build the package from source in editable mode, run:

```
pip install -e .
```

Initialize pre-commit for automatic formatting.

```
pre-commit install
```

## Copyright

Works of NIST employees are not not subject to copyright protection in the United States'

## License

The ZENOWrapper source code is hosted at https://github.com/usnistgov/zenowrapper
and is available under the [NIST LICENSE](https://github.com/usnistgov/zenowrapper/blob/main/LICENSE.md).
The license in this repository is superseded by the most updated language
on of the Public Access to NIST Research [*Copyright, Fair Use, and Licensing
Statement for SRD, Data, and Software*][nist-open].

## Contact

Jennifer A. Clark, PhD\
[Debra J. Audus, PhD][daudus] (debra.audus@nist.gov)\
[Jack F. Douglas, PhD][jdouglas]

Affilation:
[Polymer Analytics Project][polyanal]\
[Polymer and Complex Fluids Group][group1]\
[Materials Science and Engineering Division][msed]\
[Material Measurement Laboratory][mml]\
[National Institute of Standards and Technology][nist]

## Citation

Clark, J. A.; Audus, D. J.; Douglas, J. F. XXX, 2024. https://doi.org/10.18434/mds2-XXXX.

### Acknowledgements

Project based on the
[MDAnalysis Cookiecutter](https://github.com/MDAnalysis/cookiecutter-mda) version 0.1.
Please cite [MDAnalysis](https://github.com/MDAnalysis/mdanalysis#citation) when using ZENOWrapper in published work.

<!-- References -->

[nist-disclaimer]: https://www.nist.gov/open/license
[nist-open]: https://www.nist.gov/open/license#software
[docs4nist]: https://www.nist.gov/docs4nist/
[scipy]: https://scipy.org
[daudus]: https://www.nist.gov/people/debra-audus
[jdouglas]: https://www.nist.gov/people/jack-f-douglas
[polyanal]: https://www.nist.gov/programs-projects/polymer-analytics
[group1]: https://www.nist.gov/mml/materials-science-and-engineering-division/polymers-and-complex-fluids-group
[msed]: https://www.nist.gov/mml/materials-science-and-engineering-division
[mml]: https://www.nist.gov/mml
[nist]: https://www.nist.gov
