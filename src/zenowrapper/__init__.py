"""
ZenoWrapper - Hydrodynamic Property Calculations for Molecular Systems
========================================================================

ZenoWrapper provides a Python interface to the NIST ZENO software for computing
hydrodynamic, electronic, and geometric properties of molecular structures from
MD simulations. This package integrates ZENO's Monte Carlo algorithms with
MDAnalysis, enabling high-throughput analysis of biomolecular trajectories.

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

- Comparing simulation predictions with experimental measurements (AUC, DLS, etc.)
- Characterizing protein complexes, polymers, and nanoparticles
- Validating force field accuracy in solution conditions
- Predicting macromolecular behavior in crowded cellular environments

**Algorithm**

ZENO employs two complementary Monte Carlo methods:

1. **Walk-on-Spheres (exterior)**: Solves Laplace's equation to compute electrical
   properties (capacitance, polarizability) and hydrodynamic properties via
   electrostatic-hydrodynamic analogy
2. **Interior Sampling**: Estimates geometric properties (volume, gyration tensor)
   by random point sampling within the molecular volume

These algorithms provide rigorous statistical uncertainties for all computed properties.

**References**

- ZENO Documentation: https://zeno.nist.gov/
- Calculations: https://zeno.nist.gov/Calculations.html
- Output Properties: https://zeno.nist.gov/Output.html
- Douglas, J.F. et al. (2017) J. Res. NIST 122: https://doi.org/10.6028/jres.122.020

**Usage Example**

.. code-block:: python

    import MDAnalysis as mda
    from zenowrapper import ZenoWrapper

    # Load system
    u = mda.Universe('protein.pdb', 'trajectory.dcd')

    # Define VdW radii by atom type
    type_radii = {
        'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80,
        'H': 1.20, 'P': 1.80
    }

    # Initialize analysis
    zeno = ZenoWrapper(
        u.select_atoms('protein'),
        type_radii=type_radii,
        n_walks=1000000,           # exterior calculation
        n_interior_samples=100000,  # interior calculation
        temperature=298.15,         # K, for diffusion coefficient
        viscosity=0.01,            # poise (water at 20°C)
        mass=50000.0,              # Da
        buoyancy_factor=0.73,      # typical for proteins in water
        length_units='A'           # Angstroms
    )

    # Run over trajectory frames
    zeno.run(start=0, stop=100, step=1, verbose=True)

    # Access results (each is a Property object with .values and .variance arrays)
    print(f"Hydrodynamic radius: {zeno.results.hydrodynamic_radius.values.mean():.2f} Å")
    print(f"Diffusion coefficient: {zeno.results.diffusion_coefficient.values.mean():.2e}")
    print(f"Volume: {zeno.results.volume.values.mean():.2f} Å³")
"""

# Add imports here
from __future__ import annotations

from zenowrapper.main import ZenoWrapper as ZenoWrapper

from ._version import __version__ as __version__

__all__ = ["ZenoWrapper", "__version__"]
