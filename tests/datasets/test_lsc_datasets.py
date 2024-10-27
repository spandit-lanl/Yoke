"""Unit tests for the *lsc_dataset* classes.

We use the *mock* submodule of *unittest* to allow fake files, directories, and
data for testing. This avoids a lot of costly sample file storage.

"""

import pytest
from typing import Callable
import numpy as np
import numpy.typing as npt
import torch
from unittest.mock import patch, mock_open
from yoke.datasets.lsc_dataset import LSC_rho2rho_temporal_DataSet


# Mock for np.load to simulate .npz file loading behavior
class MockNpzFile:
    """Mock datafile loader class."""

    def __init__(self, data: dict[str, np.ndarray]) -> None:
        """Setup mock data."""
        self.data = data

    def __getitem__(self, item: int) -> dict[str, np.ndarray]:
        """Return single mock data sample."""
        return self.data[item]

    def close(self) -> None:
        """Close the file."""
        pass


# Mock LSCread_npz function
def mock_LSCread_npz(npz_file: str, hfield: str) -> npt.NDArray[np.float64]:
    """Test function to read data."""
    return np.ones((10, 10))  # Return a simple array for testing


@pytest.fixture
def dataset() -> LSC_rho2rho_temporal_DataSet:
    """Setup an instance of the dataset.

    Mock arguments are used for testing.

    """
    # Setup the necessary arguments
    LSC_NPZ_DIR = "/mock/path/"
    file_prefix_list = "mock_file_prefix_list.txt"
    max_timeIDX_offset = 3

    # Mock file prefix list
    mock_file_list = "mock_prefix_1\nmock_prefix_2\nmock_prefix_3\n"

    with patch("builtins.open", mock_open(read_data=mock_file_list)):
        ds = LSC_rho2rho_temporal_DataSet(
            LSC_NPZ_DIR, file_prefix_list, max_timeIDX_offset
        )

    return ds


def test_dataset_init(dataset: LSC_rho2rho_temporal_DataSet) -> None:
    """Test that the dataset is initialized correctly."""
    assert dataset.LSC_NPZ_DIR == "/mock/path/"
    assert dataset.max_timeIDX_offset == 3
    assert dataset.Nsamples == 3
    assert dataset.hydro_fields == [
        "density_case",
        "density_cushion",
        "density_maincharge",
        "density_outside_air",
        "density_striker",
        "density_throw",
        "Uvelocity",
        "Wvelocity",
    ]


def test_len(dataset: LSC_rho2rho_temporal_DataSet) -> None:
    """Test that the dataset length is correctly returned."""
    assert len(dataset) == 3


@patch("yoke.datasets.lsc_dataset.LSCread_npz", side_effect=mock_LSCread_npz)
@patch(
    "numpy.load", side_effect=lambda _: MockNpzFile({"dummy_field": np.ones((10, 10))})
)
def test_getitem(
    mock_npz_load: Callable[[str], MockNpzFile],
    mock_LSCread_npz: Callable[[MockNpzFile, str], np.ndarray],
    dataset: LSC_rho2rho_temporal_DataSet,
) -> None:
    """Test the retrieval of items from the dataset."""
    idx = 0
    start_img, end_img, Dt = dataset[idx]

    # Test shapes and types of returned values
    assert isinstance(start_img, torch.Tensor)
    assert isinstance(end_img, torch.Tensor)
    assert isinstance(Dt, float)

    # Ensure Dt is within expected bounds
    assert 0 < Dt <= 0.75

    # Test tensor shapes
    #
    # 8 channels, 10x20 image due to flipping and concatenation
    assert start_img.shape == (8, 10, 20)
    assert end_img.shape == (8, 10, 20)


def test_file_prefix_list_loading(dataset: LSC_rho2rho_temporal_DataSet) -> None:
    """Test that the file prefix list is loaded correctly."""
    expected_prefixes = ["mock_prefix_1", "mock_prefix_2", "mock_prefix_3"]
    assert dataset.file_prefix_list == expected_prefixes
