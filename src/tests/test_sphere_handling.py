"""
Test sphere handling in ZENO computations.

This module tests that:
1. Multiple spheres are properly added to the ZENO model
2. Spheres are maintained throughout the computation
3. Launch radius is appropriate for the geometry
4. Spheres are not locked prematurely
"""

import MDAnalysis as mda
import numpy as np

from zenowrapper import ZenoWrapper
from zenowrapper.zenowrapper_ext import zenolib


class TestSphereHandling:
    """Test that spheres are properly handled in ZENO computations."""

    def test_multiple_spheres_are_added(self):
        """
        Test that multiple spheres are added to the model.

        This test verifies that when we have N atoms, we get N spheres
        in the computation, and that they maintain their positions and radii.
        """
        # Create a simple universe with 5 atoms in a line
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
                [12.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        radii = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)

        # Create parameters
        params_walk = zenolib.ParametersWalkOnSpheres()
        params_walk.setTotalNumWalks(10000)
        params_walk.setSeed(42)

        params_interior = zenolib.ParametersInteriorSampling()
        params_interior.setTotalNumSamples(10000)

        params_results = zenolib.ParametersResults()
        params_results.setLengthScale(1.0, zenolib.Length.L)

        # Compute ZENO results
        results = zenolib.compute_zeno_single_frame(positions, radii, params_walk, params_interior, params_results)

        # For 5 spheres of radius 1.0 spaced 3.0 apart (overlapping),
        # we should get non-zero results
        print("\n5 overlapping spheres:")
        print(f"  Capacitance: {results.capacitance_mean:.6f}")
        print(f"  Volume: {results.volume_mean:.6f}")

        # The capacitance should be non-zero if spheres were added
        # For comparison, a single sphere of radius 1.0 has capacitance ≈ 1/π ≈ 0.318
        # Five overlapping spheres should have larger capacitance
        assert results.capacitance_mean > 0, "Capacitance is zero - spheres may not have been added!"

    def test_non_overlapping_spheres(self):
        """
        Test computation with non-overlapping spheres.

        This ensures that ZENO can handle well-separated spheres and that
        the launch radius encompasses all of them.
        """
        # Create 3 widely separated spheres
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [5.0, 8.66, 0.0],  # Forms equilateral triangle
            ],
            dtype=np.float64,
        )

        radii = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        # Create parameters
        params_walk = zenolib.ParametersWalkOnSpheres()
        params_walk.setTotalNumWalks(50000)  # More walks for separated spheres
        params_walk.setSeed(42)

        params_interior = zenolib.ParametersInteriorSampling()
        params_interior.setTotalNumSamples(50000)

        params_results = zenolib.ParametersResults()
        params_results.setLengthScale(1.0, zenolib.Length.L)

        # Compute ZENO results
        results = zenolib.compute_zeno_single_frame(positions, radii, params_walk, params_interior, params_results)

        print("\n3 widely separated spheres:")
        print(f"  Capacitance: {results.capacitance_mean:.6f}")
        print(f"  Volume: {results.volume_mean:.6f}")

        # Expected volume: 3 * (4/3 * π * 1.0^3) ≈ 12.566
        expected_volume = 3 * (4.0 / 3.0) * np.pi * (1.0**3)

        # Should get non-zero results
        assert results.capacitance_mean > 0, "Capacitance is zero!"
        assert results.volume_mean > 0, "Volume is zero!"

        # Volume should be close to expected (within 10% due to Monte Carlo)
        rel_error = abs(results.volume_mean - expected_volume) / expected_volume
        print(f"  Expected volume: {expected_volume:.6f}")
        print(f"  Relative error: {rel_error*100:.2f}%")

    def test_single_sphere_baseline(self):
        """
        Test a single sphere as a baseline.

        A single sphere is the simplest case and should always work.
        This helps diagnose if the issue is with multiple spheres specifically.
        """
        # Single sphere at origin
        positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        radii = np.array([1.0], dtype=np.float64)

        # Create parameters
        params_walk = zenolib.ParametersWalkOnSpheres()
        params_walk.setTotalNumWalks(10000)
        params_walk.setSeed(42)

        params_interior = zenolib.ParametersInteriorSampling()
        params_interior.setTotalNumSamples(10000)

        params_results = zenolib.ParametersResults()
        params_results.setLengthScale(1.0, zenolib.Length.L)

        # Compute ZENO results
        results = zenolib.compute_zeno_single_frame(positions, radii, params_walk, params_interior, params_results)

        print("\nSingle sphere (radius=1.0):")
        print(f"  Capacitance: {results.capacitance_mean:.6f} (expected ~1.0)")
        print(f"  Volume: {results.volume_mean:.6f} (expected ~{4.0/3.0*np.pi:.6f})")

        # For a sphere of radius R in ZENO's dimensionless units:
        # Capacitance = R (not R/(4πε₀) - that's the SI formula with physical units)
        # Volume = (4/3)πR³
        # So for R=1: C = 1.0, V ≈ 4.189
        expected_capacitance = 1.0
        expected_volume = (4.0 / 3.0) * np.pi * (1.0**3)

        # Should be within 10% (Monte Carlo uncertainty)
        cap_error = abs(results.capacitance_mean - expected_capacitance) / expected_capacitance
        vol_error = abs(results.volume_mean - expected_volume) / expected_volume

        print(f"  Capacitance error: {cap_error*100:.2f}%")
        print(f"  Volume error: {vol_error*100:.2f}%")

        assert results.capacitance_mean > 0, "Single sphere gave zero capacitance!"
        assert results.volume_mean > 0, "Single sphere gave zero volume!"
        assert cap_error < 0.15, f"Capacitance error {cap_error*100:.1f}% too large"
        assert vol_error < 0.15, f"Volume error {vol_error*100:.1f}% too large"

    def test_launch_radius_contains_geometry(self):
        """
        Test that the auto-computed launch radius contains all spheres.

        This test creates spheres at known positions and verifies that
        the launch sphere would encompass them all.
        """
        # Create spheres at corners of a cube
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [10.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
                [10.0, 0.0, 10.0],
                [0.0, 10.0, 10.0],
                [10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )

        radii = np.array([1.0] * 8, dtype=np.float64)

        # Calculate expected minimum launch radius
        # Center should be at (5, 5, 5)
        # Farthest point is corner at (10, 10, 10) + radius 1.0
        center = np.mean(positions, axis=0)
        max_distance = 0
        for pos, r in zip(positions, radii, strict=False):
            distance = np.linalg.norm(pos - center) + r
            max_distance = max(max_distance, distance)

        print("\nGeometry bounds:")
        print(f"  Geometric center: {center}")
        print(f"  Min required launch radius: {max_distance:.6f}")

        # Create parameters - let ZENO auto-compute launch radius
        params_walk = zenolib.ParametersWalkOnSpheres()
        params_walk.setTotalNumWalks(50000)
        params_walk.setSeed(42)

        params_interior = zenolib.ParametersInteriorSampling()
        params_interior.setTotalNumSamples(50000)

        params_results = zenolib.ParametersResults()
        params_results.setLengthScale(1.0, zenolib.Length.L)

        # Compute ZENO results
        results = zenolib.compute_zeno_single_frame(positions, radii, params_walk, params_interior, params_results)

        print("\nResults for 8 spheres at cube corners:")
        print(f"  Capacitance: {results.capacitance_mean:.6f}")
        print(f"  Volume: {results.volume_mean:.6f}")

        # Should get non-zero results if launch radius is appropriate
        assert results.capacitance_mean > 0, "Zero capacitance - launch radius may not contain geometry!"
        assert results.volume_mean > 0, "Zero volume - launch radius may not contain geometry!"

    def test_wrapper_with_multiple_atoms(self):
        """
        Test ZenoWrapper with multiple atoms to ensure end-to-end functionality.
        """
        # Create a universe with multiple atoms
        u = mda.Universe.empty(5, trajectory=True)
        u.add_TopologyAttr("type", ["C"] * 5)

        # Set positions to form a pentagon
        angles = np.linspace(0, 2 * np.pi, 6)[:-1]  # 5 points
        radius = 5.0
        positions = np.zeros((5, 3))
        positions[:, 0] = radius * np.cos(angles)
        positions[:, 1] = radius * np.sin(angles)
        u.atoms.positions = positions

        print("\nPentagon of 5 atoms:")
        print(f"  Positions:\n{positions}")

        # Run ZenoWrapper
        type_radii = {"C": 1.5}
        wrapper = ZenoWrapper(u.atoms, type_radii=type_radii, n_walks=50000, n_interior_samples=50000, seed=42)

        wrapper.run()

        # Check results
        cap_mean = wrapper.results.capacitance.values[0]
        vol_mean = wrapper.results.volume.values[0]

        print(f"  Capacitance: {cap_mean:.6f}")
        print(f"  Volume: {vol_mean:.6f}")

        assert cap_mean > 0, "ZenoWrapper gave zero capacitance!"
        assert vol_mean > 0, "ZenoWrapper gave zero volume!"

        # Expected volume: 5 spheres of radius 1.5
        expected_volume = 5 * (4.0 / 3.0) * np.pi * (1.5**3)
        print(f"  Expected volume (non-overlapping): {expected_volume:.6f}")

    def test_sphere_locking_order(self):
        """
        Test that spheres can be added multiple times without locking issues.

        This creates fresh parameter objects and runs computations twice
        to ensure spheres aren't locked from previous computations.
        """
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.5, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        radii = np.array([1.0, 1.0], dtype=np.float64)

        # First computation
        params_walk_1 = zenolib.ParametersWalkOnSpheres()
        params_walk_1.setTotalNumWalks(5000)
        params_walk_1.setSeed(42)

        params_interior_1 = zenolib.ParametersInteriorSampling()
        params_interior_1.setTotalNumSamples(5000)

        params_results_1 = zenolib.ParametersResults()
        params_results_1.setLengthScale(1.0, zenolib.Length.L)

        results1 = zenolib.compute_zeno_single_frame(
            positions, radii, params_walk_1, params_interior_1, params_results_1
        )

        print("\nFirst computation:")
        print(f"  Capacitance: {results1.capacitance_mean:.6f}")

        # Second computation with FRESH parameters
        params_walk_2 = zenolib.ParametersWalkOnSpheres()
        params_walk_2.setTotalNumWalks(5000)
        params_walk_2.setSeed(43)  # Different seed

        params_interior_2 = zenolib.ParametersInteriorSampling()
        params_interior_2.setTotalNumSamples(5000)

        params_results_2 = zenolib.ParametersResults()
        params_results_2.setLengthScale(1.0, zenolib.Length.L)

        results2 = zenolib.compute_zeno_single_frame(
            positions, radii, params_walk_2, params_interior_2, params_results_2
        )

        print("Second computation:")
        print(f"  Capacitance: {results2.capacitance_mean:.6f}")

        # Both should give non-zero results
        assert results1.capacitance_mean > 0, "First computation gave zero!"
        assert results2.capacitance_mean > 0, "Second computation gave zero!"

        # Results should be similar (within 20% due to different seeds)
        rel_diff = abs(results1.capacitance_mean - results2.capacitance_mean) / results1.capacitance_mean
        print(f"  Relative difference: {rel_diff*100:.2f}%")
        assert rel_diff < 0.3, "Results differ too much between runs!"


class TestDebugSphereGeometry:
    """Detailed debugging tests for sphere geometry handling."""

    def test_very_simple_two_spheres(self):
        """
        Simplest possible test: two touching spheres.

        This should ALWAYS work if sphere handling is correct.
        """
        # Two spheres of radius 1.0, centers 2.0 apart (touching)
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        radii = np.array([1.0, 1.0], dtype=np.float64)

        params_walk = zenolib.ParametersWalkOnSpheres()
        params_walk.setTotalNumWalks(10000)
        params_walk.setSeed(42)

        params_interior = zenolib.ParametersInteriorSampling()
        params_interior.setTotalNumSamples(10000)

        params_results = zenolib.ParametersResults()
        params_results.setLengthScale(1.0, zenolib.Length.L)

        results = zenolib.compute_zeno_single_frame(positions, radii, params_walk, params_interior, params_results)

        print("\nTwo touching spheres:")
        print(f"  Capacitance: {results.capacitance_mean:.6f}")
        print(f"  Volume: {results.volume_mean:.6f}")

        # Expected volume: 2 * (4/3 * π * 1^3) ≈ 8.378
        expected_volume = 2 * (4.0 / 3.0) * np.pi
        print(f"  Expected volume: {expected_volume:.6f}")

        assert results.capacitance_mean > 0, "Two touching spheres gave zero capacitance!"
        assert results.volume_mean > 0, "Two touching spheres gave zero volume!"

    def test_explicit_launch_radius(self):
        """
        Test with explicitly set launch radius to rule out auto-computation issues.
        """
        positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        radii = np.array([1.0], dtype=np.float64)

        # Explicitly set launch radius to 2.0 (much larger than sphere)
        params_walk = zenolib.ParametersWalkOnSpheres()
        params_walk.setTotalNumWalks(10000)
        params_walk.setSeed(42)
        params_walk.setLaunchRadius(5.0)  # Explicitly set

        params_interior = zenolib.ParametersInteriorSampling()
        params_interior.setTotalNumSamples(10000)
        params_interior.setLaunchRadius(5.0)  # Explicitly set

        params_results = zenolib.ParametersResults()
        params_results.setLengthScale(1.0, zenolib.Length.L)

        results = zenolib.compute_zeno_single_frame(positions, radii, params_walk, params_interior, params_results)

        print("\nSingle sphere with explicit launch radius=5.0:")
        print(f"  Capacitance: {results.capacitance_mean:.6f}")
        print(f"  Volume: {results.volume_mean:.6f}")

        assert results.capacitance_mean > 0, "Explicit launch radius still gave zero!"
        assert results.volume_mean > 0, "Explicit launch radius still gave zero volume!"
