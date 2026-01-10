"""
Pytest configuration for CIRISOssicle tests.

License: BSL 1.1
Author: CIRIS L3C
"""

import pytest
import sys


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow-running"
    )


@pytest.fixture(scope="session")
def gpu_available():
    """Check if GPU is available."""
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        return True
    except:
        return False


@pytest.fixture(scope="session")
def gpu_info():
    """Get GPU information."""
    try:
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(0)
        return {
            'name': props['name'].decode(),
            'sm_count': props['multiProcessorCount'],
            'memory_gb': props['totalGlobalMem'] / (1024**3),
        }
    except:
        return None
