"""
Tests for ZenoWrapper parallelization support.

Tests both frame-level (MDAnalysis) and within-frame (ZENO) parallelization.
"""

import MDAnalysis as mda
import pytest
from numpy.testing import assert_allclose

from zenowrapper import ZenoWrapper

from .utils import make_Universe


@pytest.fixture
def parallel_universe():
    """Create a test universe with multiple frames for parallel testing."""
    return make_Universe(
        size=(50, 10, 2),
        n_frames=10,
    )


class TestParallelizationSupport:
    """Test that ZenoWrapper declares parallelization support."""

    def test_parallelizable_flag(self):
        """Test that the class is marked as parallelizable."""
        assert ZenoWrapper._analysis_algorithm_is_parallelizable is True

    def test_supported_backends(self):
        """Test that all expected backends are supported."""
        backends = ZenoWrapper.get_supported_backends()
        assert "serial" in backends
        assert "multiprocessing" in backends
        assert "dask" in backends

    def test_has_aggregator_method(self):
        """Test that the aggregator method exists."""
        assert hasattr(ZenoWrapper, "_get_aggregator")


class TestSerialVsParallel:
    """Test that parallel execution gives same results as serial."""

    def test_multiprocessing_vs_serial(self, parallel_universe):
        """Test that multiprocessing backend gives same results as serial."""
        type_radii = {"X": 1.5}

        # Serial run
        zeno_serial = ZenoWrapper(
            parallel_universe.atoms,
            type_radii=type_radii,
            n_walks=5000,  # Small for test speed
            n_interior_samples=500,
            num_threads=1,
            seed=42,  # Fixed seed for reproducibility
        )
        zeno_serial.run(backend="serial")

        # Parallel run with 2 workers
        zeno_parallel = ZenoWrapper(
            parallel_universe.atoms,
            type_radii=type_radii,
            n_walks=5000,
            n_interior_samples=500,
            num_threads=1,
            seed=42,
        )
        zeno_parallel.run(backend="multiprocessing", n_workers=2)

        # Results should be identical (same seed, same algorithm)
        # Note: Due to different chunking, results may differ slightly
        # so we use a relaxed tolerance
        assert_allclose(
            zeno_serial.results["capacitance"].values,
            zeno_parallel.results["capacitance"].values,
            rtol=1e-10,
            atol=1e-10,
        )
        assert_allclose(
            zeno_serial.results["volume"].values,
            zeno_parallel.results["volume"].values,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_results_shape_with_parallel(self, parallel_universe):
        """Test that parallel run produces correctly shaped results."""
        type_radii = {"X": 1.5}

        zeno = ZenoWrapper(
            parallel_universe.atoms,
            type_radii=type_radii,
            n_walks=5000,
            n_interior_samples=500,
            num_threads=1,
        )
        zeno.run(backend="multiprocessing", n_workers=2)

        # Check that all frames are present
        assert len(zeno.results["capacitance"].values) == parallel_universe.trajectory.n_frames
        assert len(zeno.results["volume"].values) == parallel_universe.trajectory.n_frames

        # Check tensor shapes
        assert zeno.results.electric_polarizability_tensor.values.shape == (
            parallel_universe.trajectory.n_frames,
            3,
            3,
        )


class TestTwoLevelParallelism:
    """Test combining frame-level and within-frame parallelization."""

    def test_hybrid_parallelization(self, parallel_universe):
        """Test that hybrid approach (n_workers > 1, num_threads > 1) works."""
        type_radii = {"X": 1.5}

        # This uses 2 workers × 2 threads = 4 cores total
        zeno = ZenoWrapper(
            parallel_universe.atoms,
            type_radii=type_radii,
            n_walks=5000,
            n_interior_samples=500,
            num_threads=2,  # Within-frame threading
        )
        zeno.run(backend="multiprocessing", n_workers=2)  # Frame-level parallelism

        # Should complete successfully and produce correct shape
        assert len(zeno.results.capacitance.values) == parallel_universe.trajectory.n_frames

    def test_high_thread_count(self, parallel_universe):
        """Test that high num_threads works (within-frame parallelism)."""
        type_radii = {"X": 1.5}

        zeno = ZenoWrapper(
            parallel_universe.atoms,
            type_radii=type_radii,
            n_walks=10000,
            n_interior_samples=1000,
            num_threads=4,  # High thread count per frame
        )
        # Run serially but with multi-threaded frames
        zeno.run(backend="serial")

        assert len(zeno.results.capacitance.values) == parallel_universe.trajectory.n_frames


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_frame_parallel(self, parallel_universe):
        """Test that parallel backend works even with single frame."""
        type_radii = {"X": 1.5}

        zeno = ZenoWrapper(
            parallel_universe.atoms,
            type_radii=type_radii,
            n_walks=5000,
            n_interior_samples=500,
        )
        # Run only first frame with parallel backend
        zeno.run(stop=1, backend="multiprocessing", n_workers=2)

        assert len(zeno.results.capacitance.values) == 1

    def test_partial_trajectory_parallel(self, parallel_universe):
        """Test parallel execution on partial trajectory."""
        type_radii = {"X": 1.5}

        zeno = ZenoWrapper(
            parallel_universe.atoms,
            type_radii=type_radii,
            n_walks=5000,
            n_interior_samples=500,
        )
        # Run frames 2-7 (5 frames total)
        zeno.run(start=2, stop=7, backend="multiprocessing", n_workers=2)

        assert zeno.n_frames == 5
        assert len(zeno.results.capacitance.values) == 5


@pytest.mark.skipif(
    not hasattr(mda.analysis.base, "BackendDask"),
    reason="Dask not available",
)
class TestDaskBackend:
    """Test Dask backend if available."""

    def test_dask_backend(self, parallel_universe):
        """Test that Dask backend works."""
        type_radii = {"X": 1.5}

        zeno = ZenoWrapper(
            parallel_universe.atoms,
            type_radii=type_radii,
            n_walks=5000,
            n_interior_samples=500,
        )

        try:
            zeno.run(backend="dask", n_workers=2)
            assert len(zeno.results.capacitance.values) == parallel_universe.trajectory.n_frames
        except (ImportError, ValueError) as e:
            # ValueError raised by MDAnalysis if dask not installed
            # ImportError for direct import failures
            if "dask" in str(e).lower():
                pytest.skip("Dask not installed")
            else:
                raise
