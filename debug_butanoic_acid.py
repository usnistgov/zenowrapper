"""
Diagnostic script to understand the ZENO warnings for butanoic acid.
"""
import numpy as np
import MDAnalysis as mda
from zenowrapper import ZenoWrapper

# Create a simple butanoic acid structure
# C4H8O2: CH3-CH2-CH2-COOH
# Approximate coordinates for a linear chain
positions = np.array([
    # Carbon chain: C1-C2-C3-C4
    [0.0, 0.0, 0.0],      # C1 (CH3)
    [1.54, 0.0, 0.0],     # C2 (CH2)
    [3.08, 0.0, 0.0],     # C3 (CH2)
    [4.62, 0.0, 0.0],     # C4 (COOH)
    # Oxygens on C4
    [5.2, 1.2, 0.0],      # O1 (double bond)
    [5.2, -1.2, 0.0],     # O2 (OH)
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
], dtype=np.float64)

# Atom types
types = ['C', 'C', 'C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H']

# Standard VdW radii
type_radii = {"C": 1.7, "O": 1.52, "H": 1.2}

print("Butanoic Acid Structure Analysis")
print("=" * 50)
print(f"Number of atoms: {len(positions)}")
print(f"Atom types: {', '.join(set(types))}")
print()

# Calculate molecular extent
centroid = positions.mean(axis=0)
print(f"Centroid: {centroid}")

# Get radii for each atom
radii = np.array([type_radii[t] for t in types])
print(f"Radii range: {radii.min():.2f} - {radii.max():.2f} Å")

# Calculate maximum extent (distance from centroid to farthest atom surface)
distances_to_centroid = np.linalg.norm(positions - centroid, axis=1)
max_extent = np.max(distances_to_centroid + radii)
print(f"Maximum extent from centroid: {max_extent:.2f} Å")
print()

# Bounding box
min_coords = positions.min(axis=0) - radii.max()
max_coords = positions.max(axis=0) + radii.max()
bbox_size = max_coords - min_coords
print(f"Bounding box size: {bbox_size}")
print(f"Diagonal: {np.linalg.norm(bbox_size):.2f} Å")
print()

# Recommended launch radius
recommended_launch_radius = max_extent * 1.2
print(f"Recommended launch radius: {recommended_launch_radius:.2f} Å")
print(f"  (1.2× max extent for safety margin)")
print()

# Create a minimal Universe
from MDAnalysis.core.universe import Universe
from MDAnalysis.core.topology import Topology
from MDAnalysis.core.topologyattrs import Atomtypes
import MDAnalysis.coordinates.memory

# Create topology
n_atoms = len(positions)
topology_attrs = []
topology_attrs.append(Atomtypes(np.array(types)))
top = Topology(n_atoms=n_atoms, attrs=topology_attrs)

# Create universe
u = Universe(top, positions[np.newaxis, :, :], format=MDAnalysis.coordinates.memory.MemoryReader)

print("Testing ZENO with default parameters:")
print("-" * 50)

try:
    # Test 1: Default (no launch_radius specified)
    print("\nTest 1: Default parameters (no launch_radius)")
    zeno1 = ZenoWrapper(
        u.atoms,
        type_radii=type_radii,
        n_walks=10000,  # Reduced for speed
        n_interior_samples=1000,
        length_units='A',
        verbose=True
    )
    zeno1.run()
    print(f"  Capacitance: {zeno1.results.capacitance.values[0]:.4f}")
    print(f"  Volume: {zeno1.results.volume.values[0]:.4f}")
    print(f"  Hydrodynamic radius: {zeno1.results.hydrodynamic_radius.values[0]:.4f}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n" + "=" * 50)
print("\nTest 2: With explicit launch_radius")
print("-" * 50)

try:
    zeno2 = ZenoWrapper(
        u.atoms,
        type_radii=type_radii,
        n_walks=10000,
        n_interior_samples=1000,
        launch_radius=recommended_launch_radius,
        length_units='A',
        verbose=True
    )
    zeno2.run()
    print(f"  Capacitance: {zeno2.results.capacitance.values[0]:.4f}")
    print(f"  Volume: {zeno2.results.volume.values[0]:.4f}")
    print(f"  Hydrodynamic radius: {zeno2.results.hydrodynamic_radius.values[0]:.4f}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n" + "=" * 50)
print("\nTest 3: With larger launch_radius and more samples")
print("-" * 50)

try:
    zeno3 = ZenoWrapper(
        u.atoms,
        type_radii=type_radii,
        n_walks=100000,  # Increase walks
        n_interior_samples=10000,  # Increase interior samples
        launch_radius=max_extent * 1.5,  # Larger safety margin
        length_units='A',
        verbose=True
    )
    zeno3.run()
    print(f"  Capacitance: {zeno3.results.capacitance.values[0]:.4f}")
    print(f"  Volume: {zeno3.results.volume.values[0]:.4f}")
    print(f"  Hydrodynamic radius: {zeno3.results.hydrodynamic_radius.values[0]:.4f}")
except Exception as e:
    print(f"  ERROR: {e}")
