"""
Additional tests to improve code coverage for main.py.

This test file specifically targets uncovered lines in the ZenoWrapper class
and helper functions.
"""

import numpy as np
import pytest

from tests.utils import make_Universe
from zenowrapper import ZenoWrapper
from zenowrapper.main import _get_length_unit, _get_mass_unit, _get_temperature_unit, _get_viscosity_unit


class TestUnitConversionFunctions:
    """Test the unit conversion helper functions with invalid inputs."""

    def test_get_length_unit_invalid(self):
        """Test _get_length_unit with invalid unit returns default."""
        from zenowrapper.zenowrapper_ext import zenolib

        # Invalid unit should return default (L)
        result = _get_length_unit("invalid_unit")
        assert result == zenolib.Length.L

    def test_get_temperature_unit_invalid(self):
        """Test _get_temperature_unit with invalid unit returns default."""
        from zenowrapper.zenowrapper_ext import zenolib

        # Invalid unit should return default (K)
        result = _get_temperature_unit("invalid_unit")
        assert result == zenolib.Temperature.K

    def test_get_mass_unit_invalid(self):
        """Test _get_mass_unit with invalid unit returns default."""
        from zenowrapper.zenowrapper_ext import zenolib

        # Invalid unit should return default (kg)
        result = _get_mass_unit("invalid_unit")
        assert result == zenolib.Mass.kg

    def test_get_viscosity_unit_invalid(self):
        """Test _get_viscosity_unit with invalid unit returns default."""
        from zenowrapper.zenowrapper_ext import zenolib

        # Invalid unit should return default (p)
        result = _get_viscosity_unit("invalid_unit")
        assert result == zenolib.Viscosity.p


class TestPropertyErrorHandling:
    """Test Property class error handling."""

    def test_property_with_nan_values_raises_error(self):
        """Test that Property.compute_total_values raises error with NaN values."""
        from zenowrapper.main import Property

        prop = Property("test_property", (3,), "test_units")
        prop.add_value(0, 1.0)
        prop.add_value(1, np.nan)  # Add NaN value
        prop.add_value(2, 3.0)
        prop.add_variance(0, 0.1)
        prop.add_variance(1, 0.1)
        prop.add_variance(2, 0.1)

        with pytest.raises(ValueError, match="Values of NaN found in test_property"):
            prop.compute_total_values()


class TestZenoWrapperMissingRadii:
    """Test ZenoWrapper error handling for missing radii."""

    def test_missing_some_radii_raises_error(self):
        """Test that missing radii for some atom types raises error."""
        u = make_Universe(size=(6, 1, 1), n_frames=1)
        # Create multiple atom types
        u.atoms.types = ["X", "X", "Y", "Y", "Z", "Z"]

        # Provide radii for only X and Y, missing Z
        type_radii = {"X": 1.5, "Y": 2.0}

        with pytest.raises(ValueError, match="Missing radii for atom/bead types: Z"):
            _ = ZenoWrapper(u.atoms, type_radii=type_radii)

    def test_no_radii_provided_raises_error(self):
        """Test that providing None for type_radii raises error."""
        u = make_Universe(size=(3, 1, 1), n_frames=1)
        u.atoms.types = ["X", "Y", "Z"]

        # Provide None for type_radii (will be converted to empty dict)
        with pytest.raises(ValueError, match="Please specify radii for atom/bead types"):
            _ = ZenoWrapper(u.atoms, type_radii=None)


class TestZenoWrapperOptionalParameters:
    """Test ZenoWrapper with various combinations of optional parameters."""

    @pytest.fixture
    def simple_universe(self):
        """Create a simple universe for testing."""
        u = make_Universe(extras=("types",), size=(5, 1, 1), n_frames=2)
        u.atoms.types = ["X"] * 5
        # Set simple positions
        for i, ts in enumerate(u.trajectory):
            positions = np.zeros((5, 3))
            positions[:, 0] = np.arange(5) * 5.0
            positions[:, 1] = i * 2.0  # Slight variation per frame
            ts.positions = positions
        return u

    def test_with_viscosity_only(self, simple_universe):
        """Test ZenoWrapper with viscosity but no mass or temperature."""
        type_radii = {"X": 1.5}
        wrapper = ZenoWrapper(
            simple_universe.atoms,
            type_radii=type_radii,
            viscosity=0.01,
            viscosity_units="p",
            n_walks=5000,
            n_interior_samples=500,
        )
        wrapper.run()

        # Should have friction_coefficient
        assert hasattr(wrapper.results, "friction_coefficient")
        # Should NOT have sedimentation or diffusion coefficients
        # (because mass and temperature are not provided)

    def test_with_viscosity_and_mass_and_buoyancy(self, simple_universe):
        """Test ZenoWrapper with viscosity, mass, and buoyancy factor."""
        type_radii = {"X": 1.5}
        wrapper = ZenoWrapper(
            simple_universe.atoms,
            type_radii=type_radii,
            viscosity=0.01,
            viscosity_units="cp",
            mass=50000.0,
            mass_units="Da",
            buoyancy_factor=0.73,
            n_walks=5000,
            n_interior_samples=500,
        )
        wrapper.run()

        # Should have friction_coefficient AND sedimentation_coefficient
        assert hasattr(wrapper.results, "friction_coefficient")
        assert hasattr(wrapper.results, "sedimentation_coefficient")
        assert wrapper.results.sedimentation_coefficient.values.shape[0] == 2

    def test_with_viscosity_and_temperature(self, simple_universe):
        """Test ZenoWrapper with viscosity and temperature."""
        type_radii = {"X": 1.5}
        wrapper = ZenoWrapper(
            simple_universe.atoms,
            type_radii=type_radii,
            viscosity=0.01,
            viscosity_units="p",
            temperature=298.15,
            temperature_units="K",
            n_walks=5000,
            n_interior_samples=500,
        )
        wrapper.run()

        # Should have friction_coefficient AND diffusion_coefficient
        assert hasattr(wrapper.results, "friction_coefficient")
        assert hasattr(wrapper.results, "diffusion_coefficient")
        assert wrapper.results.diffusion_coefficient.values.shape[0] == 2

    def test_with_mass_only(self, simple_universe):
        """Test ZenoWrapper with mass but no viscosity."""
        type_radii = {"X": 1.5}
        wrapper = ZenoWrapper(
            simple_universe.atoms,
            type_radii=type_radii,
            mass=50000.0,
            mass_units="kDa",
            n_walks=5000,
            n_interior_samples=500,
        )
        wrapper.run()

        # Should have mass_intrinsic_viscosity
        assert hasattr(wrapper.results, "mass_intrinsic_viscosity")
        assert wrapper.results.mass_intrinsic_viscosity.values.shape[0] == 2

    def test_with_all_optional_parameters(self, simple_universe):
        """Test ZenoWrapper with all optional parameters set."""
        type_radii = {"X": 1.5}
        wrapper = ZenoWrapper(
            simple_universe.atoms,
            type_radii=type_radii,
            viscosity=0.01,
            viscosity_units="cp",
            mass=50000.0,
            mass_units="Da",
            temperature=298.15,
            temperature_units="C",
            buoyancy_factor=0.73,
            n_walks=5000,
            n_interior_samples=500,
        )
        wrapper.run()

        # Should have all optional results
        assert hasattr(wrapper.results, "friction_coefficient")
        assert hasattr(wrapper.results, "sedimentation_coefficient")
        assert hasattr(wrapper.results, "diffusion_coefficient")
        assert hasattr(wrapper.results, "mass_intrinsic_viscosity")


class TestZenoWrapperParameterSettings:
    """Test ZenoWrapper with different parameter configurations."""

    @pytest.fixture
    def simple_universe(self):
        """Create a simple universe for testing."""
        u = make_Universe(extras=("types",), size=(3, 1, 1), n_frames=1)
        u.atoms.types = ["X"] * 3
        u.trajectory[0].positions = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        return u

    def test_min_n_walks_parameter(self, simple_universe):
        """Test setting min_n_walks parameter."""
        type_radii = {"X": 1.5}
        wrapper = ZenoWrapper(
            simple_universe.atoms,
            type_radii=type_radii,
            min_n_walks=1000,
            max_rsd_capacitance=0.05,
            n_interior_samples=500,
        )
        wrapper.run()
        assert hasattr(wrapper.results, "capacitance")

    def test_max_rsd_polarizability_parameter(self, simple_universe):
        """Test setting max_rsd_polarizability parameter."""
        type_radii = {"X": 1.5}
        wrapper = ZenoWrapper(
            simple_universe.atoms,
            type_radii=type_radii,
            n_walks=5000,
            max_rsd_polarizability=0.05,
            n_interior_samples=500,
        )
        wrapper.run()
        assert hasattr(wrapper.results, "electric_polarizability_tensor")

    def test_max_run_time_walk_parameter(self, simple_universe):
        """Test setting max_run_time for walk-on-spheres."""
        type_radii = {"X": 1.5}
        wrapper = ZenoWrapper(
            simple_universe.atoms,
            type_radii=type_radii,
            n_walks=5000,
            max_run_time=10.0,  # 10 seconds max
            n_interior_samples=500,
        )
        wrapper.run()
        assert hasattr(wrapper.results, "capacitance")

    def test_skin_thickness_parameter(self, simple_universe):
        """Test setting skin_thickness parameter."""
        type_radii = {"X": 1.5}
        wrapper = ZenoWrapper(
            simple_universe.atoms,
            type_radii=type_radii,
            n_walks=5000,
            skin_thickness=0.1,
            n_interior_samples=500,
        )
        wrapper.run()
        assert hasattr(wrapper.results, "capacitance")

    def test_launch_radius_walk_parameter(self, simple_universe):
        """Test setting launch_radius for walk-on-spheres."""
        type_radii = {"X": 1.5}
        wrapper = ZenoWrapper(
            simple_universe.atoms,
            type_radii=type_radii,
            n_walks=5000,
            launch_radius=50.0,
            n_interior_samples=500,
        )
        wrapper.run()
        assert hasattr(wrapper.results, "capacitance")

    def test_min_n_interior_samples_parameter(self, simple_universe):
        """Test setting min_n_interior_samples parameter."""
        type_radii = {"X": 1.5}
        wrapper = ZenoWrapper(
            simple_universe.atoms,
            type_radii=type_radii,
            n_walks=5000,
            min_n_interior_samples=100,
            max_rsd_volume=0.05,
        )
        wrapper.run()
        assert hasattr(wrapper.results, "volume")

    def test_max_rsd_volume_parameter(self, simple_universe):
        """Test setting max_rsd_volume parameter."""
        type_radii = {"X": 1.5}
        wrapper = ZenoWrapper(
            simple_universe.atoms,
            type_radii=type_radii,
            n_walks=5000,
            n_interior_samples=1000,
            max_rsd_volume=0.05,
        )
        wrapper.run()
        assert hasattr(wrapper.results, "volume")

    def test_max_run_time_interior_parameter(self, simple_universe):
        """Test setting max_run_time for interior sampling."""
        type_radii = {"X": 1.5}
        wrapper = ZenoWrapper(
            simple_universe.atoms,
            type_radii=type_radii,
            n_walks=5000,
            n_interior_samples=1000,
            max_run_time_interior=10.0,
        )
        wrapper.run()
        assert hasattr(wrapper.results, "volume")

    def test_launch_radius_interior_parameter(self, simple_universe):
        """Test setting launch_radius for interior sampling."""
        type_radii = {"X": 1.5}
        wrapper = ZenoWrapper(
            simple_universe.atoms,
            type_radii=type_radii,
            n_walks=5000,
            n_interior_samples=1000,
            launch_radius_interior=50.0,
        )
        wrapper.run()
        assert hasattr(wrapper.results, "volume")

    def test_seed_interior_parameter(self, simple_universe):
        """Test setting seed for interior sampling."""
        type_radii = {"X": 1.5}
        wrapper = ZenoWrapper(
            simple_universe.atoms,
            type_radii=type_radii,
            n_walks=5000,
            n_interior_samples=1000,
            seed_interior=42,
        )
        wrapper.run()
        assert hasattr(wrapper.results, "volume")

    def test_seed_walk_not_set(self, simple_universe):
        """Test that walk seed=-1 branch is covered (default, no seed set)."""
        type_radii = {"X": 1.5}
        wrapper = ZenoWrapper(
            simple_universe.atoms,
            type_radii=type_radii,
            n_walks=5000,
            n_interior_samples=1000,
            seed=-1,  # Explicitly set to -1 (default)
        )
        wrapper.run()
        assert hasattr(wrapper.results, "capacitance")

    def test_seed_interior_not_set(self, simple_universe):
        """Test that interior seed=-1 branch is covered (default, no seed set)."""
        type_radii = {"X": 1.5}
        wrapper = ZenoWrapper(
            simple_universe.atoms,
            type_radii=type_radii,
            n_walks=5000,
            n_interior_samples=1000,
            seed_interior=-1,  # Explicitly set to -1 (default)
        )
        wrapper.run()
        assert hasattr(wrapper.results, "volume")

    def test_verbose_mode(self, simple_universe, capsys):
        """Test verbose output."""
        type_radii = {"X": 1.5}
        wrapper = ZenoWrapper(
            simple_universe.atoms,
            type_radii=type_radii,
            n_walks=5000,
            n_interior_samples=500,
            verbose=True,
        )
        wrapper.run()

        # Check that verbose output was printed
        captured = capsys.readouterr()
        assert "Analyzing frame" in captured.out


class TestZenoWrapperConcludePhase:
    """Test the _conclude phase which computes overall statistics."""

    @pytest.fixture
    def universe_multiframe(self):
        """Create a universe with multiple frames."""
        u = make_Universe(extras=("types",), size=(5, 1, 1), n_frames=3)
        u.atoms.types = ["X"] * 5
        for i, ts in enumerate(u.trajectory):
            positions = np.zeros((5, 3))
            positions[:, 0] = np.arange(5) * 5.0 + i
            ts.positions = positions
        return u

    def test_conclude_computes_overall_values(self, universe_multiframe):
        """Test that _conclude properly computes overall_value and overall_variance."""
        type_radii = {"X": 1.5}
        wrapper = ZenoWrapper(
            universe_multiframe.atoms,
            type_radii=type_radii,
            n_walks=5000,
            n_interior_samples=500,
        )
        wrapper.run()

        # Check that overall_value and overall_variance are computed
        assert hasattr(wrapper.results.capacitance, "overall_value")
        assert hasattr(wrapper.results.capacitance, "overall_variance")
        assert wrapper.results.capacitance.overall_value is not None
        assert wrapper.results.capacitance.overall_variance is not None

    def test_conclude_with_all_optional_results(self, universe_multiframe):
        """Test _conclude with all optional results."""
        type_radii = {"X": 1.5}
        wrapper = ZenoWrapper(
            universe_multiframe.atoms,
            type_radii=type_radii,
            viscosity=0.01,
            mass=50000.0,
            temperature=298.15,
            buoyancy_factor=0.73,
            n_walks=5000,
            n_interior_samples=500,
        )
        wrapper.run()

        # All optional properties should have overall values computed
        assert hasattr(wrapper.results.friction_coefficient, "overall_value")
        assert hasattr(wrapper.results.sedimentation_coefficient, "overall_value")
        assert hasattr(wrapper.results.diffusion_coefficient, "overall_value")
        assert hasattr(wrapper.results.mass_intrinsic_viscosity, "overall_value")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
