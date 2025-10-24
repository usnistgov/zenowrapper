"""
Hydrodynamic Property Calculations Using ZENO Monte Carlo Methods
===================================================================

This module provides the :class:`ZenoWrapper` class for computing hydrodynamic
and geometric properties of molecular systems via Monte Carlo integration.

The implementation wraps the NIST ZENO C++ library and integrates with MDAnalysis
for seamless analysis of molecular dynamics trajectories.

**Computed Properties**

**Electrical Properties (via exterior calculation)**:
    - Capacitance
    - Electric polarizability tensor and eigenvalues
    - Mean electric polarizability
    - Intrinsic conductivity

**Geometric Properties (via interior calculation)**:
    - Volume
    - Gyration tensor and eigenvalues

**Hydrodynamic Properties (derived via electrostatic-hydrodynamic analogy)**:
    - Hydrodynamic radius
    - Friction coefficient (requires viscosity)
    - Diffusion coefficient (requires temperature, viscosity)
    - Sedimentation coefficient (requires mass, buoyancy factor, viscosity)
    - Intrinsic viscosity
    - Viscometric radius

**Monte Carlo Algorithms**

1. **Walk-on-Spheres (exterior)**: Random walks launched from a sphere enclosing
   the molecule determine electrical properties by solving Laplace's equation.
   Each walk either hits the object or escapes to infinity, analogous to Zeno's
   paradox of Achilles and the Tortoise.

2. **Interior Sampling**: Random points sampled within the launch sphere determine
   volume and gyration properties based on whether points fall inside the molecule.

Both methods provide rigorous statistical uncertainties via variance estimation
and propagation of uncertainties.

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

**Notes**

- All results are stored as :class:`Property` objects with `.values` and `.variance`
  arrays indexed by frame number
- Computational cost scales with `n_walks` and `n_interior_samples`; adjust for
  accuracy vs. speed tradeoff
- Statistical uncertainties decrease as ~1/sqrt(N) for N samples
- The `seed` parameter ensures reproducibility when set to a fixed value

See Also
--------
MDAnalysis.analysis.base.AnalysisBase : Base class documentation
    https://docs.mdanalysis.org/stable/documentation_pages/analysis/base.html
ZENO Documentation : Algorithm and property details
    https://zeno.nist.gov/

References
----------
.. [1] Douglas, J.F., Zhou, H.X., and Hubbard, J.B. (2017)
       "Hydrodynamic friction and the capacitance of arbitrarily shaped objects"
       Phys. Rev. E 49: 5319
.. [2] ZENO Calculations: https://zeno.nist.gov/Calculations.html
.. [3] ZENO Output: https://zeno.nist.gov/Output.html

"""

from .zenowrapper_ext import zenolib
from typing import Union, TYPE_CHECKING

from MDAnalysis.core.universe import Universe
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.analysis.base import AnalysisBase
import numpy as np

# Helper functions to convert string units to ZENO enum values
def _get_length_unit(unit_str: str):
    """
    Convert length unit string to ZENO enum.
    
    Parameters
    ----------
    unit_str : str
        Length unit: 'm', 'cm', 'nm', 'A' (Angstrom), or 'L' (arbitrary length)
    
    Returns
    -------
    zenolib.Length
        Corresponding ZENO length unit enum
    """
    unit_map = {
        "m": zenolib.Length.m,
        "cm": zenolib.Length.cm,
        "nm": zenolib.Length.nm,
        "A": zenolib.Length.A,
        "L": zenolib.Length.L,
    }
    return unit_map.get(unit_str, zenolib.Length.L)

def _get_temperature_unit(unit_str: str):
    """
    Convert temperature unit string to ZENO enum.
    
    Parameters
    ----------
    unit_str : str
        Temperature unit: 'C' (Celsius) or 'K' (Kelvin)
    
    Returns
    -------
    zenolib.Temperature
        Corresponding ZENO temperature unit enum
    """
    unit_map = {
        "C": zenolib.Temperature.C,
        "K": zenolib.Temperature.K,
    }
    return unit_map.get(unit_str, zenolib.Temperature.K)

def _get_mass_unit(unit_str: str):
    """
    Convert mass unit string to ZENO enum.
    
    Parameters
    ----------
    unit_str : str
        Mass unit: 'Da' (Dalton), 'kDa' (kiloDalton), 'g' (gram), or 'kg' (kilogram)
    
    Returns
    -------
    zenolib.Mass
        Corresponding ZENO mass unit enum
    """
    unit_map = {
        "Da": zenolib.Mass.Da,
        "kDa": zenolib.Mass.kDa,
        "g": zenolib.Mass.g,
        "kg": zenolib.Mass.kg,
    }
    return unit_map.get(unit_str, zenolib.Mass.kg)

def _get_viscosity_unit(unit_str: str):
    """
    Convert viscosity unit string to ZENO enum.
    
    Parameters
    ----------
    unit_str : str
        Viscosity unit: 'p' (poise) or 'cp' (centipoise)
    
    Returns
    -------
    zenolib.Viscosity
        Corresponding ZENO viscosity unit enum
    """
    unit_map = {
        "p": zenolib.Viscosity.p,
        "cp": zenolib.Viscosity.cp,
    }
    return unit_map.get(unit_str, zenolib.Viscosity.p)

class Property:
    """
    Container for ZENO computed property with values and statistical uncertainties.
    
    Each property stores per-frame values and variances from Monte Carlo calculations.
    Uncertainties are rigorously estimated via variance propagation.
    
    Parameters
    ----------
    name : str
        Property name (e.g., 'capacitance', 'diffusion_coefficient')
    shape : tuple
        Shape of values array, typically (n_frames,) for scalars or (n_frames, 3, 3) for tensors
    unit : str or None
        Physical unit string for documentation (e.g., 'Å', 'Å³', 'cm²/s')
    
    Attributes
    ----------
    name : str
        Property identifier
    values : numpy.ndarray
        Computed values for each analyzed frame
    variance : numpy.ndarray
        Statistical variance (uncertainty²) for each frame value
    unit : str or None
        Physical units
    overall_value : float
        Mean across all frames (computed by :meth:`compute_total_values`)
    overall_variance : float
        Total variance across frames (computed by :meth:`compute_total_values`)
    
    Notes
    -----
    Statistical uncertainties from Monte Carlo sampling decrease as ~1/sqrt(N)
    where N is the number of random walks or interior samples.
    """
    def __init__(self, name, shape, unit):
        self.name = name
        self.values = np.nan * np.ones(shape, dtype=float)
        self.variance  = np.nan * np.ones(shape, dtype=float)
        self.unit = unit
        
    def add_value(self, index, value):
        """
        Store computed value for a specific frame.
        
        Parameters
        ----------
        index : int
            Frame index
        value : float or numpy.ndarray
            Computed property value
        """
        self.values[index] = value
        
    def add_variance(self, index, value):
        """
        Store variance (uncertainty²) for a specific frame.
        
        Parameters
        ----------
        index : int
            Frame index
        value : float or numpy.ndarray
            Statistical variance of the property
        """
        self.variance[index] = value
        
    def compute_total_values(self):
        """
        Compute overall statistics across all analyzed frames.
        
        Calculates mean value and total variance from per-frame results.
        
        Raises
        ------
        ValueError
            If any frame values are NaN (indicating incomplete analysis)
        """
        if np.any(np.isnan(self.values)):
            raise ValueError("Values of NaN found in {}".format(self.name))
        self.overall_value = np.nanmean(self.values)
        self.overall_variance = np.nansum(self.variance)


class ZenoWrapper(AnalysisBase):
    """
    Compute hydrodynamic and geometric properties via ZENO Monte Carlo methods.
    
    This analysis class wraps the NIST ZENO software to calculate physical properties
    of molecular structures from MD trajectories. It uses two Monte Carlo algorithms:
    Walk-on-Spheres (exterior) for electrical/hydrodynamic properties and Interior
    Sampling for geometric properties.
    
    The electrical properties are converted to hydrodynamic properties via the
    electrostatic-hydrodynamic analogy. All properties include rigorous statistical
    uncertainties from variance estimation.
    
    Parameters
    ----------
    atom_group : MDAnalysis.core.groups.AtomGroup
        Atoms to analyze (e.g., protein, polymer, nanoparticle)
    type_radii : dict
        Mapping of atom types to VdW radii in length_units, e.g.,
        ``{'C': 1.7, 'N': 1.55, 'O': 1.52}``. Required for all atom types present.
    n_walks : int, optional
        Number of random walks for exterior calculation (default: 1000000).
        Higher values improve accuracy but increase computation time.
    min_n_walks : int, optional
        Minimum walks before convergence check. If None, runs exactly n_walks.
    n_interior_samples : int, optional
        Number of sample points for interior calculation (default: 100000).
        Higher values improve volume/gyration accuracy.
    min_n_interior_samples : int, optional
        Minimum samples before convergence check. If None, runs exactly n_interior_samples.
    max_rsd_capacitance : float, optional
        Target relative standard deviation for capacitance (e.g., 0.01 for 1%).
        Computation stops early if reached.
    max_rsd_polarizability : float, optional
        Target relative standard deviation for polarizability.
    max_rsd_volume : float, optional
        Target relative standard deviation for volume.
    max_run_time : float, optional
        Maximum computation time in seconds per frame.
    temperature : float, optional
        Temperature for diffusion coefficient calculation. Required if diffusion
        coefficient is needed.
    size_scaling_factor : float, optional
        Scale all radii by this factor (default: 1.0)
    launch_radius : float, optional
        Radius of sphere enclosing molecule. Auto-computed if None.
    skin_thickness : float, optional
        Cutoff distance for Walk-on-Spheres termination. Auto-set if None.
    mass : float, optional
        Molecular mass for intrinsic viscosity with mass units and sedimentation
        coefficient. Specify in mass_units.
    viscosity : float, optional
        Solvent viscosity for friction/diffusion/sedimentation coefficients.
        Specify in viscosity_units (e.g., 0.01 poise for water at 20°C).
    buoyancy_factor : float, optional
        Buoyancy factor for sedimentation coefficient (typically 0.72-0.73 for
        proteins in water).
    temperature_units : str, optional
        'K' (Kelvin) or 'C' (Celsius), default 'K'
    viscosity_units : str, optional
        'p' (poise) or 'cp' (centipoise), default 'p'
    length_units : str, optional
        'm', 'cm', 'nm', 'A' (Angstrom), or 'L' (arbitrary), default 'L'
    mass_units : str, optional
        'Da', 'kDa', 'g', or 'kg', default 'kg'
    seed : int, optional
        Random seed for reproducibility. Default -1 (random seed).
    verbose : bool, optional
        Print progress information (default: False)
    **kwargs
        Additional arguments passed to :class:`MDAnalysis.analysis.base.AnalysisBase`
    
    Attributes
    ----------
    universe : MDAnalysis.core.universe.Universe
        Universe containing the trajectory
    atom_group : MDAnalysis.core.groups.AtomGroup
        Atoms being analyzed
    type_radii : numpy.ndarray
        Array of radii for each atom in atom_group
    results : MDAnalysis.analysis.base.Results
        Container for computed properties. Each property is a :class:`Property` object
        with ``.values`` and ``.variance`` arrays indexed by frame. Available properties
        depend on input parameters:
        
        **Always computed:**
            - ``capacitance`` : Molecular capacitance (length units)
            - ``electric_polarizability_tensor`` : 3×3 polarizability tensor (length³)
            - ``electric_polarizability_eigenvalues`` : Eigenvalues of tensor (length³)
            - ``electric_polarizability`` : Mean polarizability (length³)
            - ``intrinsic_conductivity`` : Dimensionless conductivity
            - ``volume`` : Molecular volume (length³)
            - ``gyration_tensor`` : 3×3 gyration tensor (length²)
            - ``gyration_eigenvalues`` : Eigenvalues of gyration tensor (length²)
            - ``capacitance_same_volume_sphere`` : Equivalent sphere radius (length)
            - ``hydrodynamic_radius`` : Effective radius in solution (length)
            - ``viscometric_radius`` : Radius from viscosity (length)
            - ``intrinsic_viscosity`` : Dimensionless intrinsic viscosity
            - ``prefactor_polarizability2intrinsic_viscosity`` : Shape factor
        
        **Requires viscosity:**
            - ``friction_coefficient`` : Resistance to motion (mass/time)
        
        **Requires temperature and viscosity:**
            - ``diffusion_coefficient`` : Translational diffusion (length²/time)
        
        **Requires mass, buoyancy_factor, and viscosity:**
            - ``sedimentation_coefficient`` : Sedimentation rate (time)
        
        **Requires mass:**
            - ``intrinsic_viscosity_mass`` : Intrinsic viscosity (length³/mass)
    
    n_frames : int
        Number of frames analyzed
    times : numpy.ndarray
        Timestamps of analyzed frames
    frames : numpy.ndarray
        Indices of analyzed frames
    
    Examples
    --------
    Basic usage computing hydrodynamic radius::
    
        >>> import MDAnalysis as mda
        >>> from zenowrapper import ZenoWrapper
        >>> u = mda.Universe('protein.pdb', 'traj.dcd')
        >>> type_radii = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8}
        >>> zeno = ZenoWrapper(u.select_atoms('protein'), type_radii=type_radii)
        >>> zeno.run(start=0, stop=100, step=10)
        >>> print(zeno.results.hydrodynamic_radius.values.mean())
    
    Computing diffusion coefficient::
    
        >>> zeno = ZenoWrapper(
        ...     u.atoms,
        ...     type_radii=type_radii,
        ...     temperature=298.15,  # K
        ...     viscosity=0.01,      # poise (water at 20°C)
        ...     temperature_units='K',
        ...     viscosity_units='p'
        ... )
        >>> zeno.run()
        >>> D = zeno.results.diffusion_coefficient.values
        >>> D_mean = D.mean()
        >>> D_std = np.sqrt(zeno.results.diffusion_coefficient.variance.mean())
    
    See Also
    --------
    MDAnalysis.analysis.base.AnalysisBase : Base class with run() method
    Property : Container for results with values and uncertainties
    
    Notes
    -----
    - Computation time scales linearly with n_walks and n_interior_samples
    - Statistical uncertainty decreases as 1/sqrt(N) for N samples
    - The electrostatic-hydrodynamic analogy relates capacitance C to hydrodynamic
      radius: R_h = C (exact for spheres, approximate for other shapes)
    - All properties include variance estimates via propagation of uncertainties
    - Setting a fixed seed ensures reproducible results
    
    References
    ----------
    .. [1] Douglas, J.F., Zhou, H.X., Hubbard, J.B. (1994)
           "Hydrodynamic friction and the capacitance of arbitrarily shaped objects"
           Phys. Rev. E 49: 5319-5331
    .. [2] ZENO Documentation: https://zeno.nist.gov/
    .. [3] ZENO Calculations: https://zeno.nist.gov/Calculations.html
    .. [4] ZENO Output Properties: https://zeno.nist.gov/Output.html
    """

    def __init__(
        self,
        atom_group: AtomGroup,
        type_radii: dict = {},
        n_walks: int = 1000000,
        min_n_walks: int = None,
        n_interior_samples: int = 100000,
        min_n_interior_samples: int = None,
        max_rsd_capacitance: float = None,
        max_rsd_polarizability: float = None,
        max_rsd_volume: float = None,
        max_run_time: float = None,
        temperature: float = None,
        size_scaling_factor: float = 1,
        launch_radius: float = None,
        skin_thickness: float = None,
        mass: str = None,
        viscosity: float = None,
        buoyancy_factor: float = None,
        temperature_units: str = "K",
        viscosity_units: str = "p",
        length_units: str = "L",
        mass_units: str = "kg",
        seed: int = -1,
        verbose: bool = False,
        **kwargs
    ):
        self.universe = atom_group.universe
        super().__init__(self.universe.trajectory, **kwargs)

        self.atom_group = atom_group

        self.length_units = length_units
        self.mass_units = mass_units
        self.temperature_units = temperature_units
        self.viscosity_units = viscosity_units
        self.verbose = verbose
        
        # Store attributes needed for optional results
        self.temperature = temperature
        self.mass = mass
        self.viscosity = viscosity
        self.buoyancy_factor = buoyancy_factor
        
        # Initialize Parameters for Walk on Spheres
        self.parametersWalkOnSpheres = zenolib.ParametersWalkOnSpheres()
        if n_walks is not None:
            self.parametersWalkOnSpheres.setTotalNumWalks(n_walks)
        if min_n_walks is not None:
            self.parametersWalkOnSpheres.setMinTotalNumWalks(min_n_walks)
        if max_rsd_capacitance is not None:
            self.parametersWalkOnSpheres.setMaxErrorCapacitance(max_rsd_capacitance)
        if max_rsd_polarizability is not None:
            self.parametersWalkOnSpheres.setMaxErrorPolarizability(max_rsd_polarizability)
        if max_run_time is not None:
            self.parametersWalkOnSpheres.setMaxRunTime(max_run_time)
        if seed != -1:
            self.parametersWalkOnSpheres.setSeed(seed)
        if skin_thickness is not None:
            self.parametersWalkOnSpheres.setSkinThickness(skin_thickness)
        if launch_radius is not None:
            self.parametersWalkOnSpheres.setLaunchRadius(launch_radius)
        
        # Initialize Parameters for Interior Sampling
        self.parametersInteriorSampling = zenolib.ParametersInteriorSampling()
        if min_n_interior_samples is not None:
            self.parametersInteriorSampling.setMinTotalNumSamples(min_n_interior_samples)
        if n_interior_samples is not None:
            self.parametersInteriorSampling.setTotalNumSamples(n_interior_samples)
        if max_rsd_volume is not None:
            self.parametersInteriorSampling.setMaxErrorVolume(max_rsd_volume)
        if max_run_time is not None:
            self.parametersInteriorSampling.setMaxRunTime(max_run_time)
        if seed != -1:
            self.parametersInteriorSampling.setSeed(seed)
        if launch_radius is not None:
            self.parametersInteriorSampling.setLaunchRadius(launch_radius)
        
        # Initialize Parameters for Parameter Results
        self.parametersResults = zenolib.ParametersResults()
        self.parametersResults.setLengthScale(1.0, _get_length_unit(length_units))
        if temperature is not None:
            self.parametersResults.setTemperature(temperature, _get_temperature_unit(temperature_units))
        if mass is not None:
            self.parametersResults.setMass(mass, _get_mass_unit(mass_units))
        if viscosity is not None:
            self.parametersResults.setSolventViscosity(viscosity, _get_viscosity_unit(viscosity_units))
        if buoyancy_factor is not None:
            self.parametersResults.setBuoyancyFactor(buoyancy_factor)
        
        # Check atom types
        if len(type_radii) == 0:
            raise ValueError(
                "Please specify radii for atom/bead types: {}".format(
                    ", ".join(self.atom_group.types)
                )
            )
        else:
            missing_radii = [x for x in self.atom_group.types if x not in type_radii]
            if missing_radii:
                raise ValueError(
                    "Missing radii for atom/bead types: {}".format(
                        ", ".join(missing_radii)
                    )
                )
            self.type_radii = np.array([type_radii[atom_type] * size_scaling_factor 
                                        for atom_type in self.atom_group.types])


    def _prepare(self):
        """
        Initialize result containers before trajectory analysis.
        
        Creates :class:`Property` objects for all computable quantities based on
        the provided parameters. Properties requiring optional parameters (temperature,
        viscosity, mass, buoyancy_factor) are only created if those parameters were
        specified during initialization.
        
        This method is called automatically by :meth:`run` before frame iteration.
        """
        
        self.results.capacitance = Property("capacitance", (self.n_frames), self.length_units)
        self.results.electric_polarizability_tensor = Property("electric_polarizability_tensor", (self.n_frames, 3, 3), f"{self.length_units}^3")
        self.results.electric_polarizability_eigenvalues = Property("electric_polarizability_eigenvalues", (self.n_frames, 3), f"{self.length_units}^3")
        self.results.electric_polarizability = Property("electric_polarizability", (self.n_frames), f"{self.length_units}^3")
        self.results.intrinsic_conductivity = Property("intrinsic_conductivity", (self.n_frames), None)
        self.results.volume = Property("volume", (self.n_frames), f"{self.length_units}^3")
        self.results.gyration_tensor = Property("gyration_tensor", (self.n_frames, 3, 3), f"{self.length_units}^2")
        self.results.gyration_eigenvalues = Property("gyration_eigenvalues", (self.n_frames, 3), f"{self.length_units}^2")
        self.results.capacitance_same_volume_sphere = Property("capacitance_same_volume_sphere", (self.n_frames), self.length_units)
        self.results.hydrodynamic_radius = Property("hydrodynamic_radius", (self.n_frames), self.length_units)
        self.results.prefactor_polarizability2intrinsic_viscosity = Property("prefactor_polarizability2intrinsic_viscosity", (self.n_frames), None)
        self.results.viscometric_radius = Property("viscometric_radius", (self.n_frames), self.length_units)
        self.results.intrinsic_viscosity = Property("intrinsic_viscosity", (self.n_frames), None)
        
        if self.viscosity is not None:
            unit_sed_coeff = f"{self.length_units} * {self.viscosity}"
            self.results.friction_coefficient = Property("friction_coefficient", (self.n_frames), unit_sed_coeff)
            if self.mass is not None and self.buoyancy_factor is not None:
                self.results.sedimentation_coefficient = Property("sedimentation_coefficient", (self.n_frames), f"{self.mass_units} / ({unit_sed_coeff})")
            if self.temperature is not None:
                self.results.diffusion_coefficient = Property("diffusion_coefficient", (self.n_frames), f"{self.temperature_units} / ({unit_sed_coeff})")
        
        if self.mass is not None:        
            self.results.mass_intrinsic_viscosity = Property("mass_intrinsic_viscosity", (self.n_frames), self.mass_units)

    def _single_frame(self):
        """
        Compute ZENO properties for the current trajectory frame.
        
        Executes the Monte Carlo calculations (Walk-on-Spheres and Interior Sampling)
        for the current frame's atomic coordinates. Results are stored in the
        corresponding :class:`Property` objects in :attr:`results`.
        
        This method is called automatically by :meth:`run` for each analyzed frame.
        All heavy computation occurs in the C++ layer via the zenolib extension.
        
        Notes
        -----
        The method extracts atomic positions from the current frame and passes them
        to the C++ ZENO implementation along with the pre-configured Monte Carlo
        parameters. Statistical uncertainties are computed automatically.
        """

        # Get values from python
        positions = self.atom_group.positions
        
        if self.verbose:
            print(f"Analyzing frame {self._frame_index}")
        
        # Call C++ function to compute ZENO results
        # All geometry building and computation happens in C++ layer
        results = zenolib.compute_zeno_single_frame(
            positions, 
            self.type_radii,
            self.parametersWalkOnSpheres,
            self.parametersInteriorSampling,
            self.parametersResults,
        )
        
        # Extract results - note the C++ returns flat arrays for tensors
        self.results.capacitance.add_value(self._frame_index, results.capacitance_mean)
        self.results.capacitance.add_variance(self._frame_index, results.capacitance_variance)
        
        # Polarizability tensor is returned as flat array (row-major)
        pol_tensor = np.array(results.polarizability_tensor_mean).reshape(3, 3)
        pol_tensor_var = np.array(results.polarizability_tensor_variance).reshape(3, 3)
        self.results.electric_polarizability_tensor.add_value(self._frame_index, pol_tensor)
        self.results.electric_polarizability_tensor.add_variance(self._frame_index, pol_tensor_var)
        
        pol_eigen = np.array(results.polarizability_eigenvalues_mean)
        pol_eigen_var = np.array(results.polarizability_eigenvalues_variance)
        self.results.electric_polarizability_eigenvalues.add_value(self._frame_index, pol_eigen)
        self.results.electric_polarizability_eigenvalues.add_variance(self._frame_index, pol_eigen_var)
        
        self.results.electric_polarizability.add_value(self._frame_index, results.mean_polarizability_mean)
        self.results.electric_polarizability.add_variance(self._frame_index, results.mean_polarizability_variance)
        
        self.results.intrinsic_conductivity.add_value(self._frame_index, results.intrinsic_conductivity_mean)
        self.results.intrinsic_conductivity.add_variance(self._frame_index, results.intrinsic_conductivity_variance)
        
        self.results.volume.add_value(self._frame_index, results.volume_mean)
        self.results.volume.add_variance(self._frame_index, results.volume_variance)
        
        # Gyration tensor
        gyr_tensor = np.array(results.gyration_tensor_mean).reshape(3, 3)
        gyr_tensor_var = np.array(results.gyration_tensor_variance).reshape(3, 3)
        self.results.gyration_tensor.add_value(self._frame_index, gyr_tensor)
        self.results.gyration_tensor.add_variance(self._frame_index, gyr_tensor_var)
        
        gyr_eigen = np.array(results.gyration_eigenvalues_mean)
        gyr_eigen_var = np.array(results.gyration_eigenvalues_variance)
        self.results.gyration_eigenvalues.add_value(self._frame_index, gyr_eigen)
        self.results.gyration_eigenvalues.add_variance(self._frame_index, gyr_eigen_var)
        
        self.results.capacitance_same_volume_sphere.add_value(self._frame_index, results.capacitance_sphere_mean)
        self.results.capacitance_same_volume_sphere.add_variance(self._frame_index, results.capacitance_sphere_variance)
        
        self.results.hydrodynamic_radius.add_value(self._frame_index, results.hydrodynamic_radius_mean)
        self.results.hydrodynamic_radius.add_variance(self._frame_index, results.hydrodynamic_radius_variance)
        
        self.results.prefactor_polarizability2intrinsic_viscosity.add_value(self._frame_index, results.q_eta_mean)
        self.results.prefactor_polarizability2intrinsic_viscosity.add_variance(self._frame_index, results.q_eta_variance)
        
        self.results.viscometric_radius.add_value(self._frame_index, results.viscometric_radius_mean)
        self.results.viscometric_radius.add_variance(self._frame_index, results.viscometric_radius_variance)
        
        self.results.intrinsic_viscosity.add_value(self._frame_index, results.intrinsic_viscosity_mean)
        self.results.intrinsic_viscosity.add_variance(self._frame_index, results.intrinsic_viscosity_variance)
        
        # Optional results
        if self.viscosity is not None:
            self.results.friction_coefficient.add_value(self._frame_index, results.friction_coefficient_mean)
            self.results.friction_coefficient.add_variance(self._frame_index, results.friction_coefficient_variance)
            
            if self.mass is not None and self.buoyancy_factor is not None:
                self.results.sedimentation_coefficient.add_value(self._frame_index, results.sedimentation_coefficient_mean)
                self.results.sedimentation_coefficient.add_variance(self._frame_index, results.sedimentation_coefficient_variance)
            
            if self.temperature is not None:
                self.results.diffusion_coefficient.add_value(self._frame_index, results.diffusion_coefficient_mean)
                self.results.diffusion_coefficient.add_variance(self._frame_index, results.diffusion_coefficient_variance)
        
        if self.mass is not None:        
            self.results.mass_intrinsic_viscosity.add_value(self._frame_index, results.mass_intrinsic_viscosity_mean)
            self.results.mass_intrinsic_viscosity.add_variance(self._frame_index, results.mass_intrinsic_viscosity_variance)

    def _conclude(self):
        """Calculate the result uncertainties of the analysis"""
        
        self.results.capacitance.compute_total_values()
        self.results.electric_polarizability_tensor.compute_total_values()
        self.results.electric_polarizability_eigenvalues.compute_total_values()
        self.results.electric_polarizability.compute_total_values()
        self.results.intrinsic_conductivity.compute_total_values()
        self.results.volume.compute_total_values()
        self.results.gyration_tensor.compute_total_values()
        self.results.gyration_eigenvalues.compute_total_values()
        self.results.capacitance_same_volume_sphere.compute_total_values()
        self.results.hydrodynamic_radius.compute_total_values()
        self.results.prefactor_polarizability2intrinsic_viscosity.compute_total_values()
        self.results.viscometric_radius.compute_total_values()
        self.results.intrinsic_viscosity.compute_total_values()
        
        if self.viscosity is not None:
            self.results.friction_coefficient.compute_total_values()
            if self.mass is not None and self.buoyancy_factor is not None:
                self.results.sedimentation_coefficient.compute_total_values()
            if self.temperature is not None:
                self.results.diffusion_coefficient.compute_total_values()
        
        if self.mass is not None:        
            self.results.mass_intrinsic_viscosity.compute_total_values()
