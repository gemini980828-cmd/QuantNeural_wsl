# tests/conftest.py
"""
Pytest configuration for QUANT-NEURAL test suite.

Centralizes sys.path setup so all test files can import project modules
(e.g., `from src.factors import ...`, `from utils.math_tools import ...`)
without per-file sys.path hacks.

Also filters third-party deprecation warnings that we cannot fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so imports work.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
root_str = str(PROJECT_ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


def pytest_configure(config):
    """
    Configure pytest warning filters for third-party noise.
    
    These warnings originate from TensorFlow/Keras using deprecated numpy 
    attributes (np.object, etc.) - a known issue with numpy 2.0 migration.
    See: https://numpy.org/devdocs/numpy_2_0_migration_guide.html
    
    Filters are message-pinned AND module-restricted to tensorflow/keras/tf2onnx.
    Warnings from our own code (src/*, main_executor.py, tests/*) will NOT be filtered.
    """
    # np.object deprecation warnings (tensorflow/keras/tf2onnx internal use)
    # Message: "In the future `np.object` will be defined as the corresponding NumPy scalar"
    # This appears as both DeprecationWarning and FutureWarning
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*`np.object`.*:DeprecationWarning:.*tensorflow.*"
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*`np.object`.*:DeprecationWarning:.*keras.*"
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*`np.object`.*:DeprecationWarning:.*tf2onnx.*"
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*`np.object`.*:FutureWarning:.*tensorflow.*"
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*`np.object`.*:FutureWarning:.*keras.*"
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*`np.object`.*:FutureWarning:.*tf2onnx.*"
    )
    
    # Also catch: "`np.object` is a deprecated alias"
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*np.object.*deprecated.*:DeprecationWarning:.*tensorflow.*"
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*np.object.*deprecated.*:DeprecationWarning:.*keras.*"
    )
    
    # __array__ copy parameter deprecation (numpy 2.0)
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*__array__.*copy.*:DeprecationWarning:.*tensorflow.*"
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*__array__.*copy.*:DeprecationWarning:.*keras.*"
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*__array__.*copy.*:FutureWarning:.*tensorflow.*"
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*__array__.*copy.*:FutureWarning:.*keras.*"
    )
