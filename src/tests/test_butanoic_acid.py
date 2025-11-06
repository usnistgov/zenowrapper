"""
Test module for ZENO calculations on butanoic acid structure.

Tests various parameter configurations including launch radius settings
and validates calculation accuracy for small organic molecules.
"""

import MDAnalysis.coordinates.memory
import numpy as np
import pytest
from MDAnalysis.core.topology import Topology
from MDAnalysis.core.topologyattrs import Atomtypes
from MDAnalysis.core.universe import Universe

from zenowrapper.main import ZenoWrapper


class TestButanoicAcid:
    """Test ZENO calculations on a butanoic acid molecule (C4H8O2)."""

    @pytest.fixture
    def butanoic_acid_universe(self):
        """
        Create a minimal Universe for butanoic acid: CH3-CH2-CH2-COOH.

        Returns
        -------
        MDAnalysis.Universe
            Universe containing butanoic acid structure with appropriate atom types.
        """
        # Approximate coordinates for a linear chain
        positions = np.array(
            [
                # Carbon chain: C1-C2-C3-C4
                [0.0, 0.0, 0.0],  # C1 (CH3)
                [1.54, 0.0, 0.0],  # C2 (CH2)
                [3.08, 0.0, 0.0],  # C3 (CH2)
                [4.62, 0.0, 0.0],  # C4 (COOH)
                # Oxygens on C4
                [5.2, 1.2, 0.0],  # O1 (double bond)
                [5.2, -1.2, 0.0],  # O2 (OH)
                # Hydrogens on C1
                [-0.5, 0.9, 0.0],
                [-0.5, -0.9, 0.0],
                [-0.5, 0.0, 0.9],
                # Hydrogens on C2
                [2.0, 0.9, 0.0],
                [2.0, -0.9, 0.0],
                # Hydrogens on C3
                [3.5, 0.9, 0.0],
                [3.5, -0.9, 0.0],
                # Hydrogen on O2
                [6.0, -1.5, 0.0],
            ],
            dtype=np.float64,
        )

        # Atom types
        types = ["C", "C", "C", "C", "O", "O", "H", "H", "H", "H", "H", "H", "H", "H"]

        # Create topology
        n_atoms = len(positions)
        topology_attrs = [Atomtypes(np.array(types))]
        top = Topology(n_atoms=n_atoms, attrs=topology_attrs)

        # Create universe with single frame
        u = Universe(top, positions[np.newaxis, :, :], format=MDAnalysis.coordinates.memory.MemoryReader)
        return u

    @pytest.fixture
    def type_radii(self):
        """Standard Van der Waals radii for C, O, H atoms."""
        return {"C": 1.7, "O": 1.52, "H": 1.2}

    @pytest.fixture
    def molecular_extent(self, butanoic_acid_universe, type_radii):
        """Calculate the maximum extent of the molecule from its centroid."""
        positions = butanoic_acid_universe.atoms.positions
        types = butanoic_acid_universe.atoms.types

        centroid = positions.mean(axis=0)
        radii = np.array([type_radii[t] for t in types])
        distances_to_centroid = np.linalg.norm(positions - centroid, axis=1)
        max_extent = np.max(distances_to_centroid + radii)
        return max_extent

    def test_universe_creation(self, butanoic_acid_universe):
        """Test that the butanoic acid universe is created correctly."""
        assert butanoic_acid_universe.atoms.n_atoms == 14
        assert len(butanoic_acid_universe.trajectory) == 1

        # Check atom types
        atom_types = set(butanoic_acid_universe.atoms.types)
        assert atom_types == {"C", "O", "H"}

    def test_default_parameters(self, butanoic_acid_universe, type_radii):
        """Test ZENO calculation with default parameters (no explicit launch_radius)."""
        zeno = ZenoWrapper(
            butanoic_acid_universe.atoms,
            type_radii=type_radii,
            n_walks=10000,
            n_interior_samples=1000,
            length_units="A",
            verbose=False,
        )
        zeno.run()

        # Check that results exist and are reasonable
        assert hasattr(zeno.results, "capacitance")
        assert hasattr(zeno.results, "volume")
        assert hasattr(zeno.results, "hydrodynamic_radius")

        assert len(zeno.results["capacitance"].values) == 1
        assert zeno.results["capacitance"].values[0] > 0
        assert zeno.results["volume"].values[0] > 0
        assert zeno.results["hydrodynamic_radius"].values[0] > 0

    @pytest.mark.parametrize(
        "launch_radius_factor, n_walks, n_interior_samples",
        [
            (1.2, 10000, 1000),  # Conservative launch radius
            (1.5, 10000, 1000),  # Larger safety margin
            (2.0, 50000, 5000),  # Very large margin with more samples
        ],
    )
    def test_launch_radius_variations(
        self, butanoic_acid_universe, type_radii, molecular_extent, launch_radius_factor, n_walks, n_interior_samples
    ):
        """Test ZENO with different launch_radius values and sample counts."""
        launch_radius = molecular_extent * launch_radius_factor

        zeno = ZenoWrapper(
            butanoic_acid_universe.atoms,
            type_radii=type_radii,
            n_walks=n_walks,
            n_interior_samples=n_interior_samples,
            launch_radius=launch_radius,
            length_units="A",
            verbose=False,
        )
        zeno.run()

        # Verify results are physically reasonable
        assert zeno.results["capacitance"].values[0] > 0
        assert zeno.results["volume"].values[0] > 0
        assert zeno.results["hydrodynamic_radius"].values[0] > 0

        # Volume should be reasonable for a small molecule (rough estimate)
        # Butanoic acid is small, volume should be on order of 100-1000 Ų
        assert 10 < zeno.results["volume"].values[0] < 10000

    def test_results_consistency(self, butanoic_acid_universe, type_radii, molecular_extent):
        """Test that multiple runs with same parameters give consistent results."""
        launch_radius = molecular_extent * 1.5

        results = []
        for _ in range(2):
            zeno = ZenoWrapper(
                butanoic_acid_universe.atoms,
                type_radii=type_radii,
                n_walks=50000,
                n_interior_samples=5000,
                launch_radius=launch_radius,
                length_units="A",
                verbose=False,
            )
            zeno.run()
            results.append(
                {
                    "capacitance": zeno.results["capacitance"].values[0],
                    "volume": zeno.results["volume"].values[0],
                    "hydrodynamic_radius": zeno.results["hydrodynamic_radius"].values[0],
                }
            )

        # Results should be similar (within 20% due to Monte Carlo nature)
        # Using higher tolerance since ZENO is stochastic
        for key in ["capacitance", "volume", "hydrodynamic_radius"]:
            mean_val = (results[0][key] + results[1][key]) / 2
            rel_diff = abs(results[0][key] - results[1][key]) / mean_val
            assert rel_diff < 0.2, f"{key} varies by {rel_diff*100:.1f}%"

    def test_molecular_extent_calculation(self, butanoic_acid_universe, type_radii, molecular_extent):
        """Test that molecular extent is calculated correctly."""
        # Molecular extent should be positive and reasonable for this molecule
        assert molecular_extent > 0
        # For a ~7 Å long molecule with ~1.7 Å radius atoms, extent should be < 10 Å
        assert molecular_extent < 15
