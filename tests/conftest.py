import pytest
import shutil
from pathlib import Path


@pytest.fixture
def shared_datadir(tmp_path):
    original = Path(__file__).parent / "data"
    dest = tmp_path / "data"
    shutil.copytree(original, dest)
    return dest
