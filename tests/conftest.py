"""Pytest configuration and fixtures for tabe tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)
    yield
    # Reset random state after test
    np.random.seed()


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    n = 50
    t = np.linspace(0, 1, n)

    # Clean signals
    linear = 2 * t + 1
    quadratic = 1 + 2 * t + 0.5 * t**2
    sinusoidal = np.sin(4 * np.pi * t)
    step = np.where(t < 0.5, 0, 1)

    # Noise
    noise = 0.1 * np.random.randn(n)

    return {
        't': t,
        'linear': linear,
        'quadratic': quadratic,
        'sinusoidal': sinusoidal,
        'step': step,
        'noise': noise,
        'noisy_linear': linear + noise,
        'noisy_quadratic': quadratic + noise,
        'noisy_sinusoidal': sinusoidal + noise,
        'noisy_step': step + noise,
    }


@pytest.fixture
def missing_data_pattern():
    """Generate missing data patterns for testing."""
    n = 30

    patterns = {
        'random': np.random.choice([True, False], n, p=[0.2, 0.8]),
        'block': np.array([False] * 10 + [True] * 10 + [False] * 10),
        'edges': np.array([True] * 3 + [False] * 24 + [True] * 3),
        'alternating': np.array([i % 2 == 0 for i in range(n)]),
    }

    return patterns


@pytest.fixture
def outlier_data():
    """Generate data with outliers for robustness testing."""
    n = 40
    t = np.linspace(0, 1, n)
    clean_signal = np.sin(2 * np.pi * t)
    noise = 0.05 * np.random.randn(n)

    # Add outliers
    outlier_signal = clean_signal + noise
    outlier_indices = [10, 25, 35]
    outlier_values = [2.0, -1.5, 1.8]

    for idx, val in zip(outlier_indices, outlier_values, strict=False):
        outlier_signal[idx] += val

    return {
        't': t,
        'clean': clean_signal,
        'with_outliers': outlier_signal,
        'outlier_indices': outlier_indices,
        'outlier_values': outlier_values,
    }


@pytest.fixture(scope='session')
def performance_data():
    """Generate larger datasets for performance testing."""
    sizes = [100, 500, 1000]
    data = {}

    for n in sizes:
        t = np.linspace(0, 1, n)
        signal = np.sin(2 * np.pi * t) + 0.5 * np.sin(8 * np.pi * t)
        noise = 0.1 * np.random.randn(n)

        data[f'n_{n}'] = {
            't': t,
            'signal': signal,
            'noisy_signal': signal + noise,
        }

    return data


# Test markers for different test categories
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        'markers',
        'slow: marks tests as slow (deselect with \'-m "not slow"\')',
    )
    config.addinivalue_line(
        'markers', 'integration: marks tests as integration tests'
    )
    config.addinivalue_line('markers', 'unit: marks tests as unit tests')


# Skip tests if optional dependencies are not available
def pytest_runtest_setup(item):
    """Setup function to handle optional dependencies."""
    # Check for specific test requirements here if needed
    pass


# Custom assertion helpers
class Helpers:
    """Helper functions for tests."""

    @staticmethod
    def assert_smooth_signal(signal, tolerance=0.1):
        """Assert that a signal is smooth (low second derivative variance)."""
        second_diff = np.diff(signal, 2)
        smoothness = np.var(second_diff)
        assert smoothness < tolerance, (
            f'Signal not smooth enough: variance={smoothness}'
        )

    @staticmethod
    def assert_preserves_trend(original, fitted, correlation_threshold=0.8):
        """Assert that fitted signal preserves the trend of original."""
        correlation = np.corrcoef(original, fitted)[0, 1]
        assert correlation > correlation_threshold, (
            f'Trend not preserved: correlation={correlation}'
        )

    @staticmethod
    def assert_valid_weights(weights):
        """Assert that weights are valid (non-negative, finite)."""
        assert np.all(weights >= 0), 'Weights must be non-negative'
        assert np.all(np.isfinite(weights)), 'Weights must be finite'
        assert np.any(weights > 0), 'At least some weights must be positive'

    @staticmethod
    def assert_residuals_reasonable(residuals, signal, max_ratio=0.5):
        """Assert that residuals are reasonable compared to signal."""
        residual_std = np.std(residuals)
        signal_std = np.std(signal)
        ratio = residual_std / signal_std if signal_std > 0 else np.inf
        assert ratio < max_ratio, f'Residuals too large: ratio={ratio}'


@pytest.fixture
def helpers():
    """Provide helper functions for tests."""
    return Helpers
