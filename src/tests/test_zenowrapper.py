import pytest
from numpy.testing import assert_allclose
import sys

import zenowrapper
from zenowrapper.main import ZenoWrapper
from tests.utils import make_Universe


def test_zenowrapper_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "zenowrapper" in sys.modules

class TestZenoWrapper:

    # fixtures are helpful functions that set up a test
    # See more at https://docs.pytest.org/en/stable/how-to/fixtures.html
    @pytest.fixture
    def universe(self):
        u = make_Universe(
            extras=("names", "resnames",),
            n_frames=3,
        )
        # create toy data to test assumptions
        for ts in u.trajectory:
            ts.positions[:ts.frame] *= -1
        return u

    @pytest.fixture
    def analysis(self, universe):
        # ZenoWrapper requires type_radii dictionary
        type_radii = {'X': 1.5}  # Default type 'X' with radius 1.5 Angstrom
        return ZenoWrapper(universe.atoms, type_radii=type_radii)

    @pytest.mark.parametrize(
        "select, n_atoms",  # argument names
        [  # argument values in a tuple, in order
            ("all", 125),
            ("index 0:9", 10),
            ("segindex 3:4", 50),
        ]
    )
    def test_atom_selection(self, universe, select, n_atoms):
        # `universe` here is the fixture defined above
        type_radii = {'X': 1.5}  # Default type 'X' with radius 1.5 Angstrom
        # Select atoms first, then pass to ZenoWrapper
        selected_atoms = universe.select_atoms(select)
        analysis = ZenoWrapper(selected_atoms, type_radii=type_radii)
        assert analysis.atom_group.n_atoms == n_atoms

    @pytest.mark.parametrize(
        "stop, expected_frames",
        [
            (1, 1),
            (2, 2),
            (3, 3)
        ]
    )
    def test_run_frames(self, analysis, stop, expected_frames):
        # assert we haven't run yet
        assert not hasattr(analysis, 'n_frames') or analysis.n_frames is None
        analysis.run(stop=stop)
        assert analysis.n_frames == expected_frames

        # Check that capacitance results exist
        assert hasattr(analysis.results, 'capacitance')
        assert analysis.results.capacitance.values.shape[0] == expected_frames
