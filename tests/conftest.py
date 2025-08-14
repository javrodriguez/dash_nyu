"""
Pytest configuration for DASH_NYU tests.
"""

import pytest
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

@pytest.fixture(scope="session")
def test_data_dir():
    """Return the test data directory."""
    return Path(__file__).parent.parent / "data" / "sample_data"

@pytest.fixture(scope="session")
def src_dir():
    """Return the source code directory."""
    return Path(__file__).parent.parent / "src"

@pytest.fixture
def sample_config():
    """Return a sample configuration dictionary."""
    return {
        "Training": {
            "epochs": 10,
            "neurons_per_layer": 20,
            "batch_type": "trajectory",
            "method": "dopri5",
            "pruning": False
        },
        "Data": {
            "noise": 0.05,
            "train_split": 0.8
        }
    }
