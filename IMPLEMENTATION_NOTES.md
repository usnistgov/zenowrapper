# ZenoWrapper Implementation - Data Transfer Optimization

## Overview

This implementation minimizes data transfer between Python and C++ layers by performing all ZENO computations (geometry building, Walk on Spheres, Interior Sampling) entirely in the C++ layer, with only a single function call per frame from Python.

## Architecture

### Python Layer (`main.py`)

**ZenoWrapper Class Structure:**

1. **`__init__`**: Initializes ZENO parameter objects once
   - Creates `ParametersWalkOnSpheres`, `ParametersInteriorSampling`, and `ParametersResults` objects
   - Configures all parameters (number of walks, error tolerances, physical properties, etc.)
   - Validates atom types and creates radii array

2. **`_prepare`**: Initializes output data structures
   - Creates `Property` objects for all ZENO results (capacitance, polarizability, volume, etc.)
   - Allocates numpy arrays for storing results across all frames
   - Handles optional results (friction coefficient, diffusion coefficient, etc.) based on provided parameters

3. **`_single_frame`**: Analyzes a single frame
   - Extracts atomic positions from MDAnalysis
   - **Single C++ call**: `zenolib.compute_zeno_single_frame(positions, radii, params_walk, params_interior, params_results)`
   - Unpacks flat C++ arrays into numpy tensors
   - Stores mean and variance values for this frame

4. **`_conclude`**: Summarizes results
   - Computes overall mean and total variance across all frames
   - Produces final results for the user

### C++ Layer (`zenolib.cpp`)

**Key Components:**

1. **ZenoResults Struct**: Efficient result container
   - Contains all ZENO outputs as simple doubles and double arrays
   - Mean and variance for each property
   - Flat arrays for tensors (row-major order) to minimize marshalling overhead

2. **compute_zeno_single_frame Function**: Main computation entry point
   ```cpp
   ZenoResults compute_zeno_single_frame(
       positions,  // Nx3 numpy array
       radii,      // N numpy array
       params_walk,
       params_interior,
       params_results
   )
   ```

   **Internal workflow:**
   - Builds `MixedModel<double>` from positions and radii
   - Creates spheres using ZENO's `Vector3` and `Sphere` classes
   - Instantiates `Zeno` object with the model
   - Calls `doWalkOnSpheres()` and `doInteriorSampling()`
   - Extracts results via `getResults()`
   - Packs all means and variances into `ZenoResults` struct
   - Returns struct to Python (nanobind handles conversion)

3. **Nanobind Bindings**: Expose C++ classes to Python
   - `ParametersWalkOnSpheres`: Walk-on-Spheres configuration
   - `ParametersInteriorSampling`: Interior sampling configuration
   - `ParametersResults`: Physical parameters and units
   - `ZenoResults`: Read-only result struct with all properties exposed
   - `compute_zeno_single_frame`: Main computation function

## Data Transfer Optimization

### Before (Inefficient):
- Create empty `MixedModel` in Python
- Loop over all atoms in Python
- For each atom: create `Vector3`, create `Sphere`, call `addSphere()` → N Python→C++ calls
- Call `doWalkOnSpheres()` → 1 call
- Call `doInteriorSampling()` → 1 call
- Access each result property individually → 26+ C++→Python calls
- **Total: ~N+30 calls across Python/C++ boundary**

### After (Optimized):
- Pass positions and radii as numpy arrays → 1 Python→C++ call
- All geometry building in C++
- All computations in C++
- All result extraction in C++
- Return single packed structure → 1 C++→Python call
- **Total: 2 calls across Python/C++ boundary per frame**

## Benefits

1. **Performance**: Minimized Python/C++ crossing overhead
2. **Simplicity**: Clean API with single function call
3. **Memory Efficiency**: Direct numpy array access via nanobind
4. **Maintainability**: Clear separation of concerns
5. **Type Safety**: Strongly typed C++ with automatic Python bindings

## ZENO Workflow

The implementation follows ZENO's standard workflow:

1. **Geometry Construction**: Build `MixedModel` from spheres
2. **Preprocessing**: ZENO automatically preprocesses the model
3. **Walk on Spheres**: Monte Carlo method for exterior properties (capacitance, polarizability)
4. **Interior Sampling**: Monte Carlo method for interior properties (volume, gyration)
5. **Results Compilation**: ZENO computes derived properties from raw Monte Carlo results

All steps 1-5 occur in a single C++ function call, with geometry built from numpy arrays.

## Usage Example

```python
from zenowrapper import ZenoWrapper
import MDAnalysis as mda

# Load trajectory
u = mda.Universe("topology.pdb", "trajectory.xtc")

# Define radii for each atom type
type_radii = {'CA': 2.0, 'CB': 1.5}

# Create analyzer
zw = ZenoWrapper(
    u.atoms,
    type_radii=type_radii,
    n_walks=100000,
    n_interior_samples=100000,
    temperature=298.15,
    viscosity=0.01
)

# Run analysis (automatically calls _prepare, _single_frame for each frame, _conclude)
zw.run()

# Access results
print(f"Hydrodynamic radius: {zw.results.hydrodynamic_radius.overall_value} ± "
      f"{np.sqrt(zw.results.hydrodynamic_radius.overall_variance)}")
```

## Notes

- All ZENO computation parameters are configured once during `__init__`
- The same parameter objects are reused for all frames (thread-safe)
- Results are accumulated frame-by-frame in Python numpy arrays
- Final statistics computed in `_conclude` using standard uncertainty propagation
