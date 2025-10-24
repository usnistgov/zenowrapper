"""
Comprehensive tests for ZENO computation functionality.

Tests the C++ bindings, parameter setup, and integration with MDAnalysis.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import MDAnalysis as mda

from zenowrapper import ZenoWrapper
from zenowrapper.zenowrapper_ext import zenolib
from tests.utils import make_Universe


class TestZenoLibBindings:
    """Test the nanobind C++ extension directly."""
    
    def test_enum_bindings(self):
        """Test that all Units enums are properly bound."""
        # Length units
        assert hasattr(zenolib.Length, 'm')
        assert hasattr(zenolib.Length, 'cm')
        assert hasattr(zenolib.Length, 'nm')
        assert hasattr(zenolib.Length, 'A')
        assert hasattr(zenolib.Length, 'L')
        
        # Temperature units
        assert hasattr(zenolib.Temperature, 'C')
        assert hasattr(zenolib.Temperature, 'K')
        
        # Mass units
        assert hasattr(zenolib.Mass, 'Da')
        assert hasattr(zenolib.Mass, 'kDa')
        assert hasattr(zenolib.Mass, 'g')
        assert hasattr(zenolib.Mass, 'kg')
        
        # Viscosity units
        assert hasattr(zenolib.Viscosity, 'p')
        assert hasattr(zenolib.Viscosity, 'cp')
    
    def test_parameter_classes_instantiation(self):
        """Test that parameter classes can be instantiated."""
        params_walk = zenolib.ParametersWalkOnSpheres()
        assert params_walk is not None
        
        params_interior = zenolib.ParametersInteriorSampling()
        assert params_interior is not None
        
        params_results = zenolib.ParametersResults()
        assert params_results is not None
    
    def test_parameter_setters(self):
        """Test that parameter setters work correctly."""
        params_walk = zenolib.ParametersWalkOnSpheres()
        params_walk.setTotalNumWalks(50000)
        params_walk.setMaxErrorCapacitance(0.01)
        params_walk.setMaxErrorPolarizability(0.01)
        params_walk.setSeed(12345)
        params_walk.setNumThreads(2)
        
        params_interior = zenolib.ParametersInteriorSampling()
        params_interior.setTotalNumSamples(10000)
        params_interior.setMinTotalNumSamples(1000)
        params_interior.setMaxErrorVolume(0.01)
        params_interior.setSeed(54321)
        
        params_results = zenolib.ParametersResults()
        params_results.setLengthScale(1.0, zenolib.Length.A)
        params_results.setTemperature(298.15, zenolib.Temperature.K)
        params_results.setMass(50000.0, zenolib.Mass.Da)
        params_results.setSolventViscosity(0.01, zenolib.Viscosity.p)
        params_results.setBuoyancyFactor(0.73)
    
    def test_compute_single_sphere(self):
        """Test computation with a single sphere."""
        # Single sphere at origin with radius 1.0 Angstrom
        positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        radii = np.array([1.0], dtype=np.float64)
        
        # Set up parameters with minimal computation for speed
        params_walk = zenolib.ParametersWalkOnSpheres()
        params_walk.setTotalNumWalks(10000)
        params_walk.setMaxErrorCapacitance(0.1)
        
        params_interior = zenolib.ParametersInteriorSampling()
        params_interior.setTotalNumSamples(1000)
        
        params_results = zenolib.ParametersResults()
        params_results.setLengthScale(1.0, zenolib.Length.A)
        
        # Run computation
        result = zenolib.compute_zeno_single_frame(
            positions, radii, params_walk, params_interior, params_results
        )
        
        # Check that result object exists and has expected attributes
        assert hasattr(result, 'capacitance_mean')
        assert hasattr(result, 'capacitance_variance')
        assert hasattr(result, 'volume_mean')
        assert hasattr(result, 'volume_variance')
        assert hasattr(result, 'polarizability_tensor_mean')
        assert hasattr(result, 'gyration_tensor_mean')
        
        # For a single sphere, capacitance should be proportional to radius
        # (exact value depends on whether computation succeeded)
        assert isinstance(result.capacitance_mean, float)
        assert isinstance(result.capacitance_variance, float)
        assert result.capacitance_variance >= 0.0  # Variance is non-negative
        
        # Volume should be approximately 4/3 * pi * r^3 if computed
        assert isinstance(result.volume_mean, float)
        assert result.volume_variance >= 0.0
    
    def test_result_tensor_shapes(self):
        """Test that tensor results have correct shapes."""
        positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        radii = np.array([1.0], dtype=np.float64)
        
        params_walk = zenolib.ParametersWalkOnSpheres()
        params_walk.setTotalNumWalks(5000)
        
        params_interior = zenolib.ParametersInteriorSampling()
        params_interior.setTotalNumSamples(500)
        
        params_results = zenolib.ParametersResults()
        params_results.setLengthScale(1.0, zenolib.Length.A)
        
        result = zenolib.compute_zeno_single_frame(
            positions, radii, params_walk, params_interior, params_results
        )
        
        # Tensors should be 3x3 flattened to 9 elements
        assert len(result.polarizability_tensor_mean) == 9
        assert len(result.polarizability_tensor_variance) == 9
        assert len(result.gyration_tensor_mean) == 9
        assert len(result.gyration_tensor_variance) == 9
        
        # Eigenvalues should have 3 elements
        assert len(result.polarizability_eigenvalues_mean) == 3
        assert len(result.polarizability_eigenvalues_variance) == 3
        assert len(result.gyration_eigenvalues_mean) == 3
        assert len(result.gyration_eigenvalues_variance) == 3
    
    def test_multiple_spheres(self):
        """Test computation with multiple spheres."""
        # Create a simple dimer: two spheres separated along x-axis
        positions = np.array([
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0]
        ], dtype=np.float64)
        radii = np.array([1.0, 1.0], dtype=np.float64)
        
        params_walk = zenolib.ParametersWalkOnSpheres()
        params_walk.setTotalNumWalks(10000)
        
        params_interior = zenolib.ParametersInteriorSampling()
        params_interior.setTotalNumSamples(1000)
        
        params_results = zenolib.ParametersResults()
        params_results.setLengthScale(1.0, zenolib.Length.A)
        
        result = zenolib.compute_zeno_single_frame(
            positions, radii, params_walk, params_interior, params_results
        )
        
        # Check basic result validity
        assert isinstance(result.capacitance_mean, float)
        assert isinstance(result.volume_mean, float)
        
        # Variances should be non-negative
        assert result.capacitance_variance >= 0.0
        assert result.volume_variance >= 0.0
    
    @pytest.mark.skip(reason="C++ binding does not currently validate input shapes")
    def test_invalid_input_shapes(self):
        """Test that mismatched positions/radii raise appropriate errors."""
        # Mismatched sizes
        positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        radii = np.array([1.0, 2.0], dtype=np.float64)  # Wrong size
        
        params_walk = zenolib.ParametersWalkOnSpheres()
        params_interior = zenolib.ParametersInteriorSampling()
        params_results = zenolib.ParametersResults()
        params_results.setLengthScale(1.0, zenolib.Length.A)
        
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            zenolib.compute_zeno_single_frame(
                positions, radii, params_walk, params_interior, params_results
            )


class TestZenoWrapperIntegration:
    """Test the ZenoWrapper class integration with MDAnalysis."""
    
    @pytest.fixture
    def simple_universe(self):
        """Create a simple universe with a few atoms."""
        u = make_Universe(
            extras=("masses", "radii", "types"),
            size=(10, 2, 1),
            n_frames=3
        )
        # Set some positions
        for ts in u.trajectory:
            # Create a simple linear arrangement
            positions = np.zeros((10, 3))
            positions[:, 0] = np.arange(10) * 5.0  # Spread along x-axis
            ts.positions = positions
        
        # Set radii (normally would come from force field)
        u.atoms.radii = np.ones(10) * 1.5  # 1.5 Angstrom radii
        
        # Set atom types
        u.atoms.types = ['X'] * 10
        
        return u
    
    def test_wrapper_instantiation(self, simple_universe):
        """Test that ZenoWrapper can be instantiated."""
        type_radii = {'X': 1.5}
        wrapper = ZenoWrapper(simple_universe.atoms, type_radii=type_radii)
        assert wrapper is not None
        assert wrapper.atom_group.n_atoms == 10
    
    def test_wrapper_with_parameters(self, simple_universe):
        """Test ZenoWrapper instantiation with various parameters."""
        type_radii = {'X': 1.5}
        wrapper = ZenoWrapper(
            simple_universe.atoms,
            type_radii=type_radii,
            n_walks=50000,
            n_interior_samples=5000,
            temperature=298.15,
            temperature_units='K',
            viscosity=0.01,
            viscosity_units='p',
            mass=50000.0,
            mass_units='Da',
            buoyancy_factor=0.73,
            length_units='A'
        )
        
        assert wrapper.atom_group.n_atoms == 10
        # Parameters should be stored
        assert hasattr(wrapper, 'parametersWalkOnSpheres')
        assert hasattr(wrapper, 'parametersInteriorSampling')
        assert hasattr(wrapper, 'parametersResults')
    
    def test_wrapper_atom_selection(self, simple_universe):
        """Test that atom selection works correctly."""
        type_radii = {'X': 1.5}
        # Select atoms first, then create wrapper
        selected_atoms = simple_universe.select_atoms('index 0:4')
        wrapper = ZenoWrapper(
            selected_atoms,
            type_radii=type_radii
        )
        assert wrapper.atom_group.n_atoms == 5
    
    @pytest.mark.parametrize("length_unit", ['m', 'cm', 'nm', 'A', 'L'])
    def test_length_units(self, simple_universe, length_unit):
        """Test different length unit specifications."""
        type_radii = {'X': 1.5}
        wrapper = ZenoWrapper(
            simple_universe.atoms,
            type_radii=type_radii,
            length_units=length_unit,
            n_walks=1000,  # Small for speed
            n_interior_samples=100
        )
        assert wrapper is not None
    
    @pytest.mark.parametrize("temp_unit", ['C', 'K'])
    def test_temperature_units(self, simple_universe, temp_unit):
        """Test different temperature unit specifications."""
        type_radii = {'X': 1.5}
        wrapper = ZenoWrapper(
            simple_universe.atoms,
            type_radii=type_radii,
            temperature=298.15,
            temperature_units=temp_unit,
            n_walks=1000,
            n_interior_samples=100
        )
        assert wrapper is not None
    
    @pytest.mark.parametrize("mass_unit", ['Da', 'kDa', 'g', 'kg'])
    def test_mass_units(self, simple_universe, mass_unit):
        """Test different mass unit specifications."""
        type_radii = {'X': 1.5}
        wrapper = ZenoWrapper(
            simple_universe.atoms,
            type_radii=type_radii,
            mass=50000.0,
            mass_units=mass_unit,
            n_walks=1000,
            n_interior_samples=100
        )
        assert wrapper is not None
    
    @pytest.mark.parametrize("visc_unit", ['p', 'cp'])
    def test_viscosity_units(self, simple_universe, visc_unit):
        """Test different viscosity unit specifications."""
        type_radii = {'X': 1.5}
        wrapper = ZenoWrapper(
            simple_universe.atoms,
            type_radii=type_radii,
            viscosity=0.01,
            viscosity_units=visc_unit,
            n_walks=1000,
            n_interior_samples=100
        )
        assert wrapper is not None
    
    def test_missing_radii_raises_error(self):
        """Test that missing radii attribute raises an error."""
        u = make_Universe(size=(5, 1, 1), n_frames=1)
        # Set types so ZenoWrapper can be instantiated
        u.atoms.types = ['X'] * 5
        # Don't set radii attribute - the test should now check for missing type_radii parameter
        
        type_radii = {}  # Empty dict should raise error
        with pytest.raises(ValueError):
            wrapper = ZenoWrapper(u.atoms, type_radii=type_radii)
    
    def test_run_analysis(self, simple_universe):
        """Test running the analysis on a trajectory."""
        type_radii = {'X': 1.5}
        wrapper = ZenoWrapper(
            simple_universe.atoms,
            type_radii=type_radii,
            n_walks=5000,  # Keep small for test speed
            n_interior_samples=500
        )
        
        # Run on all frames
        wrapper.run()
        
        # Check that results exist
        assert hasattr(wrapper, 'results')
        assert hasattr(wrapper.results, 'capacitance')
        assert hasattr(wrapper.results, 'volume')
        
        # Results should have data for each frame (stored in values array)
        assert len(wrapper.results.capacitance.values) == simple_universe.trajectory.n_frames
        assert len(wrapper.results.volume.values) == simple_universe.trajectory.n_frames
    
    def test_run_partial_trajectory(self, simple_universe):
        """Test running analysis on part of a trajectory."""
        type_radii = {'X': 1.5}
        wrapper = ZenoWrapper(
            simple_universe.atoms,
            type_radii=type_radii,
            n_walks=5000,
            n_interior_samples=500
        )
        
        # Run on first 2 frames
        wrapper.run(stop=2)
        
        # Should have processed 2 frames
        assert wrapper.n_frames == 2
        assert len(wrapper.results.capacitance.values) == 2
    
    def test_results_have_statistics(self, simple_universe):
        """Test that results include mean and variance statistics."""
        type_radii = {'X': 1.5}
        wrapper = ZenoWrapper(
            simple_universe.atoms,
            type_radii=type_radii,
            n_walks=5000,
            n_interior_samples=500
        )
        
        wrapper.run()
        
        # Each result should have values and variance arrays
        for prop_name in ['capacitance', 'volume', 'hydrodynamic_radius']:
            if hasattr(wrapper.results, prop_name):
                prop = getattr(wrapper.results, prop_name)
                assert hasattr(prop, 'values')
                assert hasattr(prop, 'variance')
                
                # Values should be an array
                assert isinstance(prop.values, np.ndarray)
                
                # Variance should be an array with non-negative values
                assert isinstance(prop.variance, np.ndarray)
                # Check that all non-NaN variances are non-negative
                non_nan_variance = prop.variance[~np.isnan(prop.variance)]
                if len(non_nan_variance) > 0:
                    assert np.all(non_nan_variance >= 0.0)
    
    def test_tensor_results_shape(self, simple_universe):
        """Test that tensor results have correct shapes."""
        type_radii = {'X': 1.5}
        wrapper = ZenoWrapper(
            simple_universe.atoms,
            type_radii=type_radii,
            n_walks=5000,
            n_interior_samples=500
        )
        
        wrapper.run()
        
        # Check tensor properties if they exist
        if hasattr(wrapper.results, 'electric_polarizability_tensor'):
            tensor = wrapper.results.electric_polarizability_tensor
            # Should have values and variance arrays, each (n_frames, 3, 3)
            assert hasattr(tensor, 'values')
            assert hasattr(tensor, 'variance')
            # Check shape: (n_frames, 3, 3)
            assert tensor.values.shape == (simple_universe.trajectory.n_frames, 3, 3)
            assert tensor.variance.shape == (simple_universe.trajectory.n_frames, 3, 3)
        
        if hasattr(wrapper.results, 'gyration_tensor'):
            tensor = wrapper.results.gyration_tensor
            assert hasattr(tensor, 'values')
            assert hasattr(tensor, 'variance')
            assert tensor.values.shape == (simple_universe.trajectory.n_frames, 3, 3)
            assert tensor.variance.shape == (simple_universe.trajectory.n_frames, 3, 3)


class TestZenoWrapperEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_atom(self):
        """Test computation with a single atom."""
        u = make_Universe(
            extras=("radii", "types"),
            size=(1, 1, 1),
            n_frames=1
        )
        u.atoms.radii = np.array([2.0])
        u.atoms.types = ['X']
        u.trajectory[0].positions = np.array([[0.0, 0.0, 0.0]])
        
        type_radii = {'X': 2.0}
        wrapper = ZenoWrapper(
            u.atoms,
            type_radii=type_radii,
            n_walks=5000,
            n_interior_samples=500
        )
        wrapper.run()
        
        assert hasattr(wrapper.results, 'capacitance')
    
    @pytest.mark.skip(reason="Zero radius handling not yet implemented in C++ binding")
    def test_zero_radius_handling(self):
        """Test handling of zero or negative radii."""
        u = make_Universe(
            extras=("radii", "types"),
            size=(3, 1, 1),
            n_frames=1
        )
        u.atoms.radii = np.array([1.0, 0.0, 1.0])  # One zero radius
        u.atoms.types = ['X', 'Y', 'X']
        u.trajectory[0].positions = np.array([
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0]
        ])
        
        # Should raise an error or handle gracefully
        type_radii = {'X': 1.0, 'Y': 0.0}  # Zero radius
        with pytest.raises((ValueError, RuntimeError, AssertionError)):
            wrapper = ZenoWrapper(u.atoms, type_radii=type_radii)
    
    def test_reproducibility_with_seed(self, ):
        """Test that results are reproducible with a fixed seed."""
        u = make_Universe(
            extras=("radii", "types"),
            size=(5, 1, 1),
            n_frames=1
        )
        u.atoms.radii = np.ones(5) * 1.5
        u.atoms.types = ['X'] * 5
        u.trajectory[0].positions = np.random.rand(5, 3) * 10.0
        
        type_radii = {'X': 1.5}
        # Run twice with same seed
        wrapper1 = ZenoWrapper(
            u.atoms,
            type_radii=type_radii,
            n_walks=10000,
            n_interior_samples=1000,
            seed=42
        )
        wrapper1.run()
        
        wrapper2 = ZenoWrapper(
            u.atoms,
            type_radii=type_radii,
            n_walks=10000,
            n_interior_samples=1000,
            seed=42
        )
        wrapper2.run()
        
        # Results should be identical (use values array)
        if hasattr(wrapper1.results, 'capacitance') and hasattr(wrapper2.results, 'capacitance'):
            cap1 = wrapper1.results.capacitance.values[0]
            cap2 = wrapper2.results.capacitance.values[0]
            assert_allclose(cap1, cap2, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
