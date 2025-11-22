============================
Parallelization Guide
============================

ZenoWrapper provides two independent levels of parallelization that can be combined for optimal performance on multi-core systems.

Two-Level Parallelism
=====================

1. **Frame-Level Parallelism** (MDAnalysis)

   Distributes trajectory frames across multiple Python processes using MDAnalysis's parallel analysis framework.

2. **Within-Frame Parallelism** (ZENO C++)

   Parallelizes Monte Carlo walks within each frame using ZENO's native C++ threading.

Architecture
============

.. code-block:: text

    ┌───────────────────────────────────────────────────────────────┐
    │          MDAnalysis Multiprocessing (Frame Level)             │
    │  Distributes FRAMES across Python processes                   │
    ├───────────────────────────────────────────────────────────────┤
    │  Process 1    │  Process 2    │  Process 3    │  Process 4    │
    │  Frames 0-24  │  Frames 25-49 │  Frames 50-74 │  Frames 75-99 │
    └───────┬───────┴───────┬───────┴───────┬───────┴───────┬───────┘
            │               │               │               │
            ▼               ▼               ▼               ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │ ZENO C++      │ │ ZENO C++      │ │ ZENO C++      │ │ ZENO C++      │
    │ Threading     │ │ Threading     │ │ Threading     │ │ Threading     │
    │ (Within Frame)│ │ (Within Frame)│ │ (Within Frame)│ │ (Within Frame)│
    ├───────────────┤ ├───────────────┤ ├───────────────┤ ├───────────────┤
    │ Thread 1      │ │ Thread 1      │ │ Thread 1      │ │ Thread 1      │
    │ Thread 2      │ │ Thread 2      │ │ Thread 2      │ │ Thread 2      │
    │ Thread 3      │ │ Thread 3      │ │ Thread 3      │ │ Thread 3      │
    │ Thread 4      │ │ Thread 4      │ │ Thread 4      │ │ Thread 4      │
    └───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘

    Total Parallelism: 4 processes × 4 threads = 16 parallel computations

Choosing a Parallelization Strategy
====================================

The optimal strategy depends on your workload characteristics:

Many Frames, Fast Computation
------------------------------

**Use frame-level parallelism only**

- **Configuration**: ``backend='multiprocessing'``, ``n_workers=N_CORES``, ``num_threads=1``
- **Best for**: Trajectories with >100 frames, ``n_walks`` < 100,000
- **Memory**: N_CORES × base memory (each worker loads full trajectory)
- **Scaling**: Near-linear up to ~physical cores

.. code-block:: python

    import MDAnalysis as mda
    from zenowrapper import ZenoWrapper

    u = mda.Universe('topology.pdb', 'trajectory.dcd')  # 1000 frames

    zeno = ZenoWrapper(
        u.atoms,
        type_radii={'C': 1.7, 'N': 1.55, 'O': 1.52},
        n_walks=50000,       # Moderate computation per frame
        n_interior_samples=5000,
        num_threads=1        # Single-threaded per frame
    )

    # Distribute frames across 16 workers
    zeno.run(backend='multiprocessing', n_workers=16)

Few Frames, Expensive Computation
----------------------------------

**Use within-frame parallelism only**

- **Configuration**: ``backend='serial'``, ``num_threads=N_CORES``
- **Best for**: <20 frames, ``n_walks`` > 1,000,000
- **Memory**: 1× base memory (shared across threads)
- **Scaling**: 90-95% efficiency (ZENO's C++ threading is very efficient)

.. code-block:: python

    u = mda.Universe('protein.pdb', 'single_frame.pdb')  # Single frame

    zeno = ZenoWrapper(
        u.atoms,
        type_radii={'C': 1.7, 'N': 1.55, 'O': 1.52},
        n_walks=10000000,    # Very expensive: 10M walks!
        n_interior_samples=1000000,
        num_threads=16       # Multi-threaded ZENO computation
    )

    # Process serially but with multi-threaded frames
    zeno.run(backend='serial')

Balanced Workload (Hybrid)
---------------------------

**Use both levels of parallelism**

- **Configuration**: ``backend='multiprocessing'``, ``n_workers=K``, ``num_threads=M`` where K×M ≤ N_CORES
- **Best for**: Medium trajectories (20-200 frames), moderate computation
- **Memory**: K × base memory
- **Scaling**: 60-75% efficiency (overhead from both levels)

.. code-block:: python

    u = mda.Universe('topology.pdb', 'trajectory.dcd')  # 100 frames

    zeno = ZenoWrapper(
        u.atoms,
        type_radii={'C': 1.7, 'N': 1.55, 'O': 1.52},
        n_walks=500000,      # Moderate computation
        n_interior_samples=50000,
        num_threads=4        # 4 threads per frame
    )

    # 4 workers × 4 threads = 16 cores total
    zeno.run(backend='multiprocessing', n_workers=4)

Performance Comparison
======================

Example: 100 frames, 1,000,000 walks per frame, 16-core machine

+---------------------+----------------+----------------+--------------+------------------+
| Configuration       | n_workers      | num_threads    | Total Time   | Memory Usage     |
+=====================+================+================+==============+==================+
| Serial              | 1              | 1              | ~1000s       | 1× (baseline)    |
+---------------------+----------------+----------------+--------------+------------------+
| Frame-parallel only | 16             | 1              | ~65s         | 16×              |
+---------------------+----------------+----------------+--------------+------------------+
| Thread-parallel only| 1              | 16             | ~100s        | 1×               |
+---------------------+----------------+----------------+--------------+------------------+
| Hybrid              | 4              | 4              | ~30s         | 4×               |
+---------------------+----------------+----------------+--------------+------------------+

.. note::
   Performance numbers are approximate and depend on system architecture,
   memory bandwidth, and workload specifics.

Backend Selection
=================

Serial Backend
--------------

.. code-block:: python

    zeno.run(backend='serial')

- Single-process execution
- Always available
- Use with high ``num_threads`` for within-frame parallelism
- Best for: debugging, single frames, small systems

Multiprocessing Backend
-----------------------

.. code-block:: python

    zeno.run(backend='multiprocessing', n_workers=8)

- Standard Python multiprocessing
- No additional dependencies
- Good for local multi-core machines
- Each worker gets independent Python process
- **Limitation**: Cannot use with streaming readers (e.g., IMDReader)

Dask Backend
------------

.. code-block:: python

    zeno.run(backend='dask', n_workers=8)

- Requires ``dask`` and ``dask.distributed`` packages
- Supports distributed computing across multiple machines
- More sophisticated scheduling
- Better for very large workloads or clusters

.. code-block:: bash

    # Install dask support
    pip install "dask[distributed]"

Limitations
===========

Trajectory Reader Compatibility
--------------------------------

Frame-level parallelization requires trajectory readers that support:

1. **Random access**: Ability to seek to arbitrary frames
2. **Pickling**: Serialization for inter-process communication
3. **Independent copies**: Each worker creates its own reader instance

**Compatible readers** (most file-based formats):
- DCD, XTC, TRR, NetCDF, HDF5, PDB, etc.

**Incompatible readers**:
- :class:`~MDAnalysis.coordinates.IMD.IMDReader` (streaming, no random access)
- Any custom readers without pickle support

For incompatible readers, use serial backend with within-frame threading:

.. code-block:: python

    # IMDReader example (streaming data)
    u = mda.Universe('topology.tpr', 'imd://localhost:8889')

    zeno = ZenoWrapper(
        u.atoms,
        type_radii=type_radii,
        num_threads=8  # Use threading only
    )

    # Must use serial backend
    zeno.run(backend='serial')

Memory Considerations
---------------------

Each worker in frame-level parallelization loads a complete copy of the trajectory:

.. code-block:: python

    # Memory usage ≈ n_workers × trajectory_size
    memory_needed = n_workers * trajectory_memory_footprint

For large trajectories, consider:
- Using fewer workers with more threads per worker
- Processing trajectory in chunks
- Using memory-efficient trajectory formats (e.g., XTC instead of DCD)

Best Practices
==============

1. **Start with profiling**: Run a few frames serially to estimate per-frame cost
2. **Match strategy to workload**: Use guidelines above based on frame count and computation cost
3. **Monitor memory**: Ensure ``n_workers × trajectory_size`` fits in RAM
4. **Test scaling**: Verify speedup with small tests before full production runs
5. **Use fixed seeds**: Set ``seed`` parameter for reproducible parallel results
6. **Check results**: Compare serial vs parallel runs on small dataset to verify correctness

Example: Adaptive Strategy
---------------------------

.. code-block:: python

    import MDAnalysis as mda
    from zenowrapper import ZenoWrapper
    import multiprocessing

    u = mda.Universe('topology.pdb', 'trajectory.dcd')
    n_cores = multiprocessing.cpu_count()
    n_frames = len(u.trajectory)

    type_radii = {'C': 1.7, 'N': 1.55, 'O': 1.52}
    n_walks = 1000000

    # Adaptive strategy based on workload
    if n_frames > 100 and n_walks < 100000:
        # Many frames, fast computation: maximize frame parallelism
        config = {
            'backend': 'multiprocessing',
            'n_workers': n_cores,
            'num_threads': 1
        }
    elif n_frames < 20 and n_walks > 1000000:
        # Few frames, expensive: maximize thread parallelism
        config = {
            'backend': 'serial',
            'n_workers': None,
            'num_threads': n_cores
        }
    else:
        # Balanced: hybrid approach
        n_workers = max(1, n_cores // 4)
        threads_per_worker = n_cores // n_workers
        config = {
            'backend': 'multiprocessing',
            'n_workers': n_workers,
            'num_threads': threads_per_worker
        }

    print(f"Using strategy: {config}")

    zeno = ZenoWrapper(
        u.atoms,
        type_radii=type_radii,
        n_walks=n_walks,
        num_threads=config['num_threads']
    )

    if config['backend'] == 'serial':
        zeno.run(backend='serial')
    else:
        zeno.run(backend=config['backend'], n_workers=config['n_workers'])

See Also
========

- :ref:`parallel-analysis` : MDAnalysis parallel analysis framework
- :class:`~MDAnalysis.analysis.base.AnalysisBase` : Base class documentation
- `ZENO Documentation <https://zeno.nist.gov/>`_ : Algorithm and implementation details
