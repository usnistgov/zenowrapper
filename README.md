ZENOWrapper
==============================
[//]: # (Badges)

[![GH Actions Status][badge_actions]][url_actions]
[![License: NIST](https://img.shields.io/badge/License-NIST-blue.svg)][nist-open]
[![Powered by MDAnalysis][badge_mda]][url_mda]
[![Docs: NIST](https://img.shields.io/badge/Docs-NIST-blue.svg)][docs4nist]

[badge_actions]: https://github.com/usnistgov/zenowrapper/actions/workflows/gh-ci.yaml/badge.svg
[badge_mda]: https://img.shields.io/badge/powered%20by-MDAnalysis-orange.svg?logoWidth=16&logo=data:image/x-icon;base64,AAABAAEAEBAAAAEAIAAoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJD+XwCY/fEAkf3uAJf97wGT/a+HfHaoiIWE7n9/f+6Hh4fvgICAjwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACT/yYAlP//AJ///wCg//8JjvOchXly1oaGhv+Ghob/j4+P/39/f3IAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJH8aQCY/8wAkv2kfY+elJ6al/yVlZX7iIiI8H9/f7h/f38UAAAAAAAAAAAAAAAAAAAAAAAAAAB/f38egYF/noqAebF8gYaagnx3oFpUUtZpaWr/WFhY8zo6OmT///8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgICAn46Ojv+Hh4b/jouJ/4iGhfcAAADnAAAA/wAAAP8AAADIAAAAAwCj/zIAnf2VAJD/PAAAAAAAAAAAAAAAAICAgNGHh4f/gICA/4SEhP+Xl5f/AwMD/wAAAP8AAAD/AAAA/wAAAB8Aov9/ALr//wCS/Z0AAAAAAAAAAAAAAACBgYGOjo6O/4mJif+Pj4//iYmJ/wAAAOAAAAD+AAAA/wAAAP8AAABhAP7+FgCi/38Axf4fAAAAAAAAAAAAAAAAiIiID4GBgYKCgoKogoB+fYSEgZhgYGDZXl5e/m9vb/9ISEjpEBAQxw8AAFQAAAAAAAAANQAAADcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAjo6Mb5iYmP+cnJz/jY2N95CQkO4pKSn/AAAA7gAAAP0AAAD7AAAAhgAAAAEAAAAAAAAAAACL/gsAkv2uAJX/QQAAAAB9fX3egoKC/4CAgP+NjY3/c3Nz+wAAAP8AAAD/AAAA/wAAAPUAAAAcAAAAAAAAAAAAnP4NAJL9rgCR/0YAAAAAfX19w4ODg/98fHz/i4uL/4qKivwAAAD/AAAA/wAAAP8AAAD1AAAAGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALGxsVyqqqr/mpqa/6mpqf9KSUn/AAAA5QAAAPkAAAD5AAAAhQAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADkUFBSuZ2dn/3V1df8uLi7bAAAATgBGfyQAAAA2AAAAMwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB0AAADoAAAA/wAAAP8AAAD/AAAAWgC3/2AAnv3eAJ/+dgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9AAAA/wAAAP8AAAD/AAAA/wAKDzEAnP3WAKn//wCS/OgAf/8MAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIQAAANwAAADtAAAA7QAAAMAAABUMAJn9gwCe/e0Aj/2LAP//AQAAAAAAAAAA
[url_actions]: https://github.com/usnistgov/zenowrapper/actions?query=branch%3Amain+workflow%3Agh-ci
[url_mda]: https://www.mdanalysis.org

## [NIST Disclaimer][nist-disclaimer]

Certain commercial equipment, instruments, or materials are identified in this paper to foster understanding. Such identification does not imply recommendation or endorsement by the National Institute of Standards and Technology, nor does it imply that the materials or equipment identified are necessarily the best available for the purpose.

## Summary
ZENO computes material properties using numerical path integration techniques based on Brownian motion Monte Carlo methods (MCMs). These methods provide stochastic solutions to elliptic partial differential equations (PDEs), which represent the desired material properties or serve as intermediates for their computation. The mathematical framework maps these problems onto the electrostatic capacitance problem, where the stochastic solution corresponds to the probability of a random walk from infinity hitting the material's surface. ZENO employs the Walk-on-Spheres (WoS) algorithm to efficiently simulate Brownian motion, enabling larger jumps compared to traditional Brownian dynamics techniques, significantly reducing computational cost.

The ZENOWrapper package provides a Python interface to the ZENO computation engine, integrating it seamlessly with MDAnalysis. This integration automates the preparation of ZENO input files, execution of computations, and retrieval of results as Python objects. By accepting MDAnalysis Universe/AtomGroup objects and trajectory frames, ZENOWrapper enables hydrodynamic and transport property analyses to be conducted directly within MDAnalysis workflows, facilitating reproducible research.

Additionally, ZENOWrapper bridges ZENO’s specialized input system with any simulation package supported by MDAnalysis. It also leverages MDAnalysis’ parallelization capabilities and compatibility with interactive molecular dynamics (IMD3), enhancing the efficiency and scalability of hydrodynamic property calculations for macromolecular systems.

ZENOWrapper is bound by a [Code of Conduct](https://github.com/usnistgov/zenowrapper/blob/main/CODE_OF_CONDUCT.md).

## Documentation

Online: [NIST Pages][docs4nist]

### Dependencies

This package is tested for python 3.10+ on all Windows, MacOS, and Linux systems.
[Scipy][scipy] must be installed before installation.

### Installation

To build ZENOWrapper from source, we highly recommend using virtual environments.
Below we provide instructions for `pip`.

#### Install ZENO

Follow the [installation instructions for ZENO](https://zeno.nist.gov/Compilation.html).
Then set an environmental variable for the path: `ZENOPATH='/Your/Path/to/ZENO'` containing the `cpp` and `zeno-build` directories.

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

Works of NIST employees are not not subject to copyright protection in the United States

## License

The ZENOWrapper source code is hosted at https://github.com/usnistgov/zenowrapper
and is available under the [NIST LICENSE](https://github.com/usnistgov/zenowrapper/blob/main/LICENSE.md).
The license in this repository is superseded by the most updated language
on of the Public Access to NIST Research [*Copyright, Fair Use, and Licensing
Statement for SRD, Data, and Software*][nist-open].

## Contact

Jennifer A. Clark, PhD\
[Derek Juba][djuba] (derek.juba@nist.gov)\
[Walid Keyrouz][walidk] (walid.keyrouz@nist.gov)\
[Debra J. Audus, PhD][daudus] (debra.audus@nist.gov)\
[Jack F. Douglas, PhD][jdouglas]

Affilation:
[Polymer Analytics Project][polyanal]\
[Polymer and Complex Fluids Group][group1]\
[Materials Science and Engineering Division][msed]\
[Material Measurement Laboratory][mml]\
[National Institute of Standards and Technology][nist]

## Citation

- Clark, J. A., D. J. Audus, J. F. Douglas. XXX, 2024. https://doi.org/10.18434/mds2-XXXX
- Juba, D., W. Keyrouz, M. Mascagni, M.Brady. Procedia Computer Science, 80, 2026. https://doi.org/10.1016/j.procs.2016.05.319
- Juba, D., D. J. Audus, M. Mascagni, J. F. Douglas, W. Keyrouz Journal of Research of National Institute of Standards and Technology, 20, 2017. https://doi.org/10.6028/jres.122.020micro


### Acknowledgements

Project based on the
[MDAnalysis Cookiecutter](https://github.com/MDAnalysis/cookiecutter-mda) version 0.1.
Please cite [MDAnalysis](https://github.com/MDAnalysis/mdanalysis#citation) when using ZENOWrapper in published work.

<!-- References -->

[nist-disclaimer]: https://www.nist.gov/open/license
[nist-open]: https://www.nist.gov/open/license#software
[docs4nist]: https://pages.nist.gov/zenowrapper/en/main/index.html
[scipy]: https://scipy.org
[djuba]: https://www.nist.gov/people/derek-juba
[walidk]: https://www.nist.gov/people/walid-keyrouz
[daudus]: https://www.nist.gov/people/debra-audus
[jdouglas]: https://www.nist.gov/people/jack-f-douglas
[polyanal]: https://www.nist.gov/programs-projects/polymer-analytics
[group1]: https://www.nist.gov/mml/materials-science-and-engineering-division/polymers-and-complex-fluids-group
[msed]: https://www.nist.gov/mml/materials-science-and-engineering-division
[mml]: https://www.nist.gov/mml
[nist]: https://www.nist.gov
