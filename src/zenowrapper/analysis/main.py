"""
ZenoWrapper --- :mod:`zenowrapper.analysis.ZenoWrapper`
===========================================================

This module contains the :class:`ZenoWrapper` class.

"""

#import zenolib
from typing import Union, TYPE_CHECKING

from MDAnalysis.analysis.base import AnalysisBase
import numpy as np

if TYPE_CHECKING:
    from MDAnalysis.core.universe import Universe, AtomGroup

class Property:
    def __init__(self, name, shape, unit):
        self.name = name
        self.values = np.nan * np.ones(shape, dtype=float)
        self.variance  = np.nan * np.ones(shape, dtype=float)
        self.unit = unit
        
    def add_value(self, index, value):
        self.values[index] = value
        
    def add_variance(self, index, value):
        self.values[index] = value
        
    def compute_total_values(self):
        if np.any(np.isnan(self.values)):
            raise ValueError("Values of NaN found in {}".format(self.name))
        self.overall_value = np.nanmean(self.values)
        self.overall_variance = np.nansum(self.variance)

class ZenoWrapper(AnalysisBase):
    """ZenoWrapper class.

    This class is used to perform analysis on a trajectory.

    Parameters
    ----------
    universe_or_atomgroup: :class:`~MDAnalysis.core.universe.Universe` or :class:`~MDAnalysis.core.groups.AtomGroup`
        Universe or group of atoms to apply this analysis to.
        If a trajectory is associated with the atoms,
        then the computation iterates over the trajectory.
    select: str
        Selection string for atoms to extract from the input Universe or
        AtomGroup
    buoyancy_factor : float, default=None
        Buoyancy factor, required for the computation of results: ``sedimentation coefficient``,
        ``friction_coefficient`` and ``sedimentation coefficient``
    mass : float, default=None
        Mass of system, required for the computation of results: ``mass_intrinsic viscosity``
        and the ``sedimentation coefficient``
    temperature : float, default=None
        Temperature of system, required for the computation of results: ``diffusion_coefficient``
    viscosity : float, default=None
        Viscosity of surrounding solvent, required for the computation of results: ``diffusion coefficient``,
        ``friction_coefficient`` and ``sedimentation coefficient``

    Attributes
    ----------
    universe: :class:`~MDAnalysis.core.universe.Universe`
        The universe to which this analysis is applied
    atomgroup: :class:`~MDAnalysis.core.groups.AtomGroup`
        The atoms to which this analysis is applied
    type_radii: numpy.ndarray
        List of radii corresponding to the atoms/beads in ``atomgroup``
    results: :class:`~MDAnalysis.analysis.base.Results`
        results of calculation are stored here, after calling
        :meth:`ZenoWrapper.run`
    start: Optional[int]
        The first frame of the trajectory used to compute the analysis
    stop: Optional[int]
        The frame to stop at for the analysis
    step: Optional[int]
        Number of frames to skip between each analyzed frame
    n_frames: int
        Number of frames analysed in the trajectory
    times: numpy.ndarray
        array of Timestep times. Only exists after calling
        :meth:`ZenoWrapper.run`
    frames: numpy.ndarray
        array of Timestep frame indices. Only exists after calling
        :meth:`ZenoWrapper.run`
    """

    def __init__(
        self,
        universe_or_atomgroup: Union["Universe", "AtomGroup"],
        select: str = "all",
        type_radii: dict = {},
        n_walks: int = None,
        min_n_walks: int = None,
        n_interior_samples: int = None,
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

        super().__init__(universe_or_atomgroup.trajectory, **kwargs)

        self.universe = universe_or_atomgroup.universe
        self.atomgroup = universe_or_atomgroup.select_atoms(select)

        self.length_units = length_units
        self.mass_units = mass_units
        self.temperature_units = temperature_units
        self.viscosity_units = viscosity_units
        self.verbose = verbose
        
        # Initialize Parameters for Walk on Spheres
        self.parametersWalkOnSpheres = zenolib.parametersWalkOnSpheres()
        if n_walks is not None:
            self.parametersWalkOnSpheres.seTotalNumWalks(n_walks)
        if min_n_walks is not None:
            self.parametersWalkOnSpheres.setMinTotalNumWalks(min_n_walks)
        if max_rsd_capacitance is not None:
            self.parametersWalkOnSpheres.setMaxErrorCapacitance(max_rsd_capacitance)
        if max_rsd_polarizability is not None:
            self.parametersWalkOnSpheres.setMaxErrorPolarizability(max_rsd_polarizability)
        if max_run_time is not None:
            self.parametersWalkOnSpheres.setMaxRunTime(max_run_time)
        if seed is not None:
            self.parametersWalkOnSpheres.setSeed(seed)
        if skin_thickness is not None:
            self.parametersWalkOnSpheres.setSkinThickness(skin_thickness)
        if launch_radius is not None:
            self.parametersWalkOnSpheres.setLaunchRadius(launch_radius)
        
#        # Initialize Parameters for Interior Sampling
#        self.parametersInteriorSampling = zenolib.parametersInteriorSampling()
#        if min_n_interior_samples is not None:
#            self.parametersInteriorSampling.setMinTotalNumSamples(min_n_interior_samples)
#        if n_interior_samples is not None:
#            self.parametersInteriorSampling.setTotalNumSamples(n_interior_samples)
#        if max_rsd_volume is not None:
#            self.parametersInteriorSampling.setMaxErrorVolume(max_rsd_volume)
#        if max_run_time is not None:
#            self.parametersInteriorSampling.setMaxRunTime(max_run_time)
#        if seed is not None:
#            self.parametersInteriorSampling.setSeed(seed)
#        if launch_radius is not None:
#            self.parametersInteriorSampling.setLaunchRadius(launch_radius)
#        
#        # Initialize Parameters for Parameter Results
#        self.parametersResults = zenolib.parametersResults()
#        self.parametersResults.setLengthScale(1, length_units)
#        if temperature is not None:
#            self.parametersResults.setTemperature(temperature, temperature_units)
#        if mass is not None:
#            self.parametersResults.setMass(mass, mass_units)
#        if viscosity is not None:
#            self.parametersResults.setMass(viscosity, viscosity_units)
#        if buoyancy_factor is not None:
#            self.parametersResults.setBuoyancyFactor(buoyancy_factor)
        
        # Check atom types
        if len(type_radii) == 0:
            raise ValueError(
                "Please specify radii for atom/bead types: {}".format(
                    ", ".join(self.atomgroup.types)
                )
            )
        else:
            missing_radii = [x for x in self.atomgroup.types if x not in type_radii]
            if missing_radii:
                raise ValueError(
                    "Missing radii for atom/bead types: {}".format(
                        ", ".join(missing_radii)
                    )
                )
            self.type_radii = np.array([type_radii[x] for x in type_radii])
        

    def _prepare(self):
        """Initialize Zeno Outputs"""
        
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
        """Calculate data from a single frame of trajectory"""
        
############ Do this
#        # Get values from python
#        positions = self.atomgroup.positions
#        
#        if self.verbose:
#            print("Building spatial data structure")
#        MixedModel = zenolib.MixedModel()
#        for i, pos in enumerate(positions):
#            vec = zenolib.Vector3(*pos)
#            sphere = zenolib.Sphere(vec, self.type_radii[i])
#            MixedModel.addSphere(sphere)
#        Zeno = zenolib.Zeno(MixedModel)            
#    
#        if self.verbose:
#            print("Walk on Spheres")    
#        doWalkOnSpheresStatus = Zeno.doWalkOnSpheres(
#            self.parametersWalkOnSpheres,
#            self.parametersResults
#        )
#        if not doWalkOnSpheresStatus:
#            raise ValueError("Error: no geometry loaded")
#            
#        if self.verbose:
#            print("Interior Samples")
#        doInteriorSamplingStatus = Zeno.doInteriorSampling(
#            self.parametersInteriorSampling,
#			self.parametersResults,
#        )
#        if not doInteriorSamplingStatus:
#            raise ValueError("Error: no geometry loaded")
#        
#        if Zeno.getTotalTime() > self.max_run_time:
#            warnings.warn(
#                "*** Warning *** Max run time was reached.  Not all requested"
#                " computations may have been performed."
#            )
#        
#        # TODO: Initialize Results class
#        results = zenolib.Results()
#        
#        Zeno.getResults(self.parametersResults, results)
#############
#        # All set
#        self.results.capacitance.add_value(self._frame_index) = results.capacitance.value.getMean()
#        self.results.electric_polarizability_tensor.add_value(self._frame_index) = results.polarizabilityTensor.value.getMean()
#        self.results.electric_polarizability_eigenvalues.add_value(self._frame_index) = results.polarizabilityEigenvalues.value.getMean()
#        self.results.electric_polarizability.add_value(self._frame_index) = results.meanPolarizability.value.getMean()
#        self.results.intrinsic_conductivity.add_value(self._frame_index) = results.intrinsicConductivity.value.getMean()
#        self.results.volume.add_value(self._frame_index) = results.volume.value.getMean()
#        self.results.gyration_tensor.add_value(self._frame_index) = results.gyrationTensor.value.getMean()
#        self.results.gyration_eigenvalues.add_value(self._frame_index) = results.gyrationEigenvalues.value.getMean()
#        self.results.capacitance_same_volume_sphere.add_value(self._frame_index) = results.capacitanceOfASphere.value.getMean()
#        self.results.hydrodynamic_radius.add_value(self._frame_index) = results.hydrodynamicRadius.value.getMean()
#        self.results.prefactor_polarizability2intrinsic_viscosity.add_value(self._frame_index) = results.q_eta.value.getMean()
#        self.results.viscometric_radius.add_value(self._frame_index) = results.viscometricRadius.value.getMean()
#        self.results.intrinsic_viscosity.add_value(self._frame_index) = results.intrinsicViscosity.value.getMean()
#        
#        self.results.capacitance.add_variance(self._frame_index) = results.capacitance.value.getVariance()
#        self.results.electric_polarizability_tensor.add_variance(self._frame_index) = results.polarizabilityTensor.value.getVariance()
#        self.results.electric_polarizability_eigenvalues.add_variance(self._frame_index) = results.polarizabilityEigenvalues.value.getVariance()
#        self.results.electric_polarizability.add_variance(self._frame_index) = results.meanPolarizability.value.getVariance()
#        self.results.intrinsic_conductivity.add_variance(self._frame_index) = results.intrinsicConductivity.value.getVariance()
#        self.results.volume.add_variance(self._frame_index) = results.volume.value.getVariance()
#        self.results.gyration_tensor.add_variance(self._frame_index) = results.gyrationTensor.value.getVariance()
#        self.results.gyration_eigenvalues.add_variance(self._frame_index) = results.gyrationEigenvalues.value.getVariance()
#        self.results.capacitance_same_volume_sphere.add_variance(self._frame_index) = results.capacitanceOfASphere.value.getVariance()
#        self.results.hydrodynamic_radius.add_variance(self._frame_index) = results.hydrodynamicRadius.value.getVariance()
#        self.results.prefactor_polarizability2intrinsic_viscosity.add_variance(self._frame_index) = results.q_eta.value.getVariance()
#        self.results.viscometric_radius.add_variance(self._frame_index) = results.viscometricRadius.value.getVariance()
#        self.results.intrinsic_viscosity.add_variance(self._frame_index) = results.intrinsicViscosity.value.getVariance()
#        
#        if self.viscosity is not None:
#            self.results.friction_coefficient.add_value(self._frame_index) = results.frictionCoefficient.value.getMean()
#            self.results.friction_coefficient.add_variance(self._frame_index) = results.frictionCoefficient.value.getVariance()
#            if self.mass is not None and self.buoyancy_factor is not None:
#                self.results.sedimentation_coefficient.add_value(self._frame_index) = results.sedimentationCoefficient.value.getMean()
#                self.results.sedimentation_coefficient.add_variance(self._frame_index) = results.sedimentationCoefficient.value.getVariance()
#            if self.temperature is not None:
#                self.results.diffusion_coefficient.add_value(self._frame_index) = results.diffusionCoefficient.value.getMean()
#                self.results.diffusion_coefficient.add_variance(self._frame_index) = results.diffusionCoefficient.value.getVariance()
#        
#        if self.mass is not None:        
#            self.results.mass_intrinsic_viscosity.add_value(self._frame_index) = results.intrinsicViscosityConventional.value.getMean()
#            self.results.mass_intrinsic_viscosity.add_variance(self._frame_index) = results.intrinsicViscosityConventional.value.getVariance()

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
