""""Tests for the mt_density_dyn_pulse_localized_qs module."""

import pytest
from ruscs.mt_density_dyn_pulse_localized_qs import *

import numpy as np

class TestMTDensityDynPulseLocalizedQS:
    """Test cases for mt_density_dyn_pulse_localized_qs functions."""

    def test_placeholder(self):
        """Placeholder test to ensure test discovery works."""
        assert True


    # Initial checks tests:
    def test_negative_rate(self):
        """Test that a ValueError is raised when rate is negative."""
        # Initialize network for test
        neighbours = [[1], [3, 2], [1]]
        kmax = 2
        maxiter = 10
        with pytest.raises(ValueError, match="Parameter rate must be non-negative."):
            mt_density_dynamic_qs_pulse_loc(
                infected_vector = np.zeros(maxiter),
                rate = -0.1,
                alpha = 0.5,
                delta = 0.5,
                maxiter = maxiter,
                neighbors = neighbours,
                kmax = kmax
            )
    def test_negative_alpha(self):
        """Test that a ValueError is raised when alpha is negative."""
        # Initialize network for test
        neighbours = [[1], [3, 2], [1]]
        kmax = 2
        maxiter = 10
        with pytest.raises(ValueError, match="Parameter alpha must be non-negative."):
            mt_density_dynamic_qs_pulse_loc(
                infected_vector = np.zeros(maxiter),
                rate = 0.1,
                alpha = -0.5,
                delta = 0.5,
                maxiter = maxiter,
                neighbors = neighbours,
                kmax = kmax
            )
    def test_negative_delta(self):
        """Test that a ValueError is raised when delta is negative."""
        # Initialize network for test
        neighbours = [[1], [3, 2], [1]]
        kmax = 2
        maxiter = 10
        with pytest.raises(ValueError, match="Parameter delta must be non-negative."):
            mt_density_dynamic_qs_pulse_loc(
                infected_vector = np.zeros(maxiter),
                rate = 0.1,
                alpha = 0.5,
                delta = -0.5,
                maxiter = maxiter,
                neighbors = neighbours,
                kmax = kmax
            )