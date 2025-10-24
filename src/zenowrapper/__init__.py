"""
ZenoWrapper - Hydrodynamic Property Calculations for Molecular Systems
========================================================================

ZenoWrapper provides a Python interface to the NIST ZENO software for computing
hydrodynamic and geometric properties of molecular structures from MD simulations.
This package integrates ZENO's Monte Carlo algorithms with MDAnalysis, enabling
high-throughput analysis of biomolecular trajectories.

**Key Applications**

Compute essential physical properties for understanding molecular behavior in solution:

- **Diffusion coefficients** - predict translational mobility and transport rates
- **Sedimentation coefficients** - analyze molecular mass and shape from ultracentrifugation
- **Intrinsic viscosity** - determine solution behavior and polymer characterization
- **Hydrodynamic radius** - estimate effective molecular size in solution
- **Friction coefficients** - quantify resistance to motion through solvent
- **Gyration tensor** - characterize molecular shape and anisotropy

**Scientific Impact**

These properties are critical for:

- Comparing simulation predictions with experimental measurements (AUC, DLS, viscometry)
- Characterizing protein complexes, polymers, and nanoparticles
- Validating force field accuracy in solution conditions
- Predicting macromolecular behavior in crowded cellular environments

**Algorithm**

ZENO employs two complementary Monte Carlo methods:

1. **Walk-on-Spheres (exterior)**: Solves Laplace's equation to compute electrical
   properties (capacitance, polarizability) via electrostatic-hydrodynamic analogy
2. **Interior Sampling**: Estimates geometric properties (volume, gyration tensor)
   by random point sampling within the molecular volume

These algorithms provide rigorous statistical uncertainties for all computed properties.

**References**

- ZENO Documentation: https://zeno.nist.gov/
- Calculations: https://zeno.nist.gov/Calculations.html
- Output Properties: https://zeno.nist.gov/Output.html
- Douglas, J.F. et al. (2017) J. Res. NIST 122: https://doi.org/10.6028/jres.122.020

**Quick Start**

:class:`zenowrapper.main.ZenoWrapper` : Main analysis class

"""

# Add imports here
from importlib.metadata import version

__version__ = version("zenowrapper")

from zenowrapper.main import ZenoWrapper as ZenoWrapper