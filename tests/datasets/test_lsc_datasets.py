"""Unit tests for the *lsc_dataset* classes.

We use the *mock* submodule of *unittest* to allow fake files, directories, and
data for testing. This avoids a lot of costly sample file storage.

"""

import os
import pytest
import tempfile
import numpy as np
import torch
from typing import List
from unittest.mock import patch, mock_open, MagicMock
from yoke.datasets.lsc_dataset import LSC_rho2rho_temporal_DataSet
from yoke.datasets.lsc_dataset import LSC_cntr2hfield_DataSet


# Mock np.load to simulate loading .npz files
class MockNpzFile:
    """Set up mock file load."""

    def __init__(self, data: dict[str, np.ndarray]) -> None:
        """Setup mock data."""
        self.data = data

    def __getitem__(self, item: str) -> np.ndarray:
        """Return single mock data sample."""
        return self.data[item]

    def close(self) -> None:
        """Close the file."""
        pass


# Mock LSCread_npz function
def mock_LSCread_npz(npz_file: MockNpzFile, hfield: str) -> np.ndarray:
    """Test function to read data."""
    return np.ones((10, 10))  # Return a simple array for testing


@pytest.fixture
def dataset() -> LSC_rho2rho_temporal_DataSet:
    """Setup an instance of the dataset.

    Mock arguments are used for testing.

    """
    LSC_NPZ_DIR = "/mock/path/"
    file_prefix_list = "mock_file_prefix_list.txt"
    max_timeIDX_offset = 3
    max_file_checks = 5

    mock_file_list = "mock_prefix_1\nmock_prefix_2\nmock_prefix_3\n"
    with patch("builtins.open", mock_open(read_data=mock_file_list)):
        with patch("random.shuffle") as mock_shuffle:
            ds = LSC_rho2rho_temporal_DataSet(
                LSC_NPZ_DIR, file_prefix_list, max_timeIDX_offset, max_file_checks
            )
            mock_shuffle.assert_called_once()

    return ds


def test_dataset_init(dataset: LSC_rho2rho_temporal_DataSet) -> None:
    """Test that the dataset is initialized correctly."""
    assert dataset.LSC_NPZ_DIR == "/mock/path/"
    assert dataset.max_timeIDX_offset == 3
    assert dataset.max_file_checks == 5
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
@patch("pathlib.Path.is_file", return_value=True)
def test_getitem(
    mock_is_file: MagicMock,
    mock_npz_load: MagicMock,
    mock_LSCread_npz: MagicMock,
    dataset: LSC_rho2rho_temporal_DataSet,
) -> None:
    """Test the retrieval of items from the dataset."""
    idx = 0
    start_img, end_img, Dt = dataset[idx]

    assert isinstance(start_img, torch.Tensor)
    assert isinstance(end_img, torch.Tensor)
    assert isinstance(Dt, float)

    assert 0 < Dt <= 0.75
    assert start_img.shape == (8, 10, 20)
    assert end_img.shape == (8, 10, 20)


def test_file_prefix_list_loading(dataset: LSC_rho2rho_temporal_DataSet) -> None:
    """Test that the file prefix list is loaded correctly."""
    expected_prefixes = ["mock_prefix_1", "mock_prefix_2", "mock_prefix_3"]
    assert sorted(dataset.file_prefix_list) == sorted(expected_prefixes)


@patch("pathlib.Path.is_file", return_value=False)
def test_getitem_max_file_checks(
    mock_is_file: MagicMock, dataset: LSC_rho2rho_temporal_DataSet
) -> None:
    """Test that max_file_checks is respected.

    Ensure FileNotFoundError is raised if files are not found.

    """
    err_msg = (
        r"\[Errno 2\] No such file or directory: "
        r"'/mock/path/mock_prefix_2_pvi_idx\d{5}\.npz'"
    )
    with pytest.raises(FileNotFoundError, match=err_msg):
        dataset[0]


@patch("numpy.load", side_effect=OSError("File could not be loaded"))
def test_getitem_load_error(
    mock_npz_load: MagicMock, dataset: LSC_rho2rho_temporal_DataSet
) -> None:
    """Test error thrown if load unsuccessful."""
    with pytest.raises(IOError, match="File could not be loaded"):
        dataset[0]


# Tests for cntr2field dataset
@pytest.fixture
def create_mock_files():
    """Create temporary files and directories for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_dir = os.path.join(tmpdir, "npz_files/")
        os.makedirs(npz_dir, exist_ok=True)

        npz_file = os.path.join(npz_dir, "test_file.npz")
        np.savez(npz_file, dummy_data=np.array([1, 2, 3]))

        filelist_path = os.path.join(tmpdir, "filelist.txt")
        with open(filelist_path, "w") as f:
            f.write("test_file.npz\n")

        design_file = os.path.join(tmpdir, "design.csv")
        with open(design_file, "w") as f:
            f.write("sim_key,bspline_node1,bspline_node2\n")
            f.write("test_key,0.1,0.2\n")

        yield {
            "npz_dir": npz_dir,
            "filelist": filelist_path,
            "design_file": design_file,
        }


@patch("yoke.datasets.lsc_dataset.LSCread_npz")
@patch("yoke.datasets.lsc_dataset.LSCnpz2key")
@patch("yoke.datasets.lsc_dataset.LSCcsv2bspline_pts")
def test_dataset_length(
    mock_lsc_csv2bspline_pts: MagicMock,
    mock_lsc_npz2key: MagicMock,
    mock_lsc_read_npz: MagicMock,
    create_mock_files,
):
    """Test that the dataset length matches the number of samples."""
    files = create_mock_files
    dataset = LSC_cntr2hfield_DataSet(
        LSC_NPZ_DIR=files["npz_dir"],
        filelist=files["filelist"],
        design_file=files["design_file"],
    )

    assert len(dataset) == 1


@patch("yoke.datasets.lsc_dataset.LSCread_npz")
@patch("yoke.datasets.lsc_dataset.LSCnpz2key")
@patch("yoke.datasets.lsc_dataset.LSCcsv2bspline_pts")
def test_dataset_getitem(
    mock_lsc_csv2bspline_pts: MagicMock,
    mock_lsc_npz2key: MagicMock,
    mock_lsc_read_npz: MagicMock,
    create_mock_files,
):
    """Test that the __getitem__ method returns the correct data format."""
    files = create_mock_files

    # Mock return values
    mock_lsc_read_npz.return_value = np.array([0.0, np.nan, 1.0])
    mock_lsc_npz2key.return_value = "test_key"
    mock_lsc_csv2bspline_pts.return_value = np.array([0.1, 0.2])

    dataset = LSC_cntr2hfield_DataSet(
        LSC_NPZ_DIR=files["npz_dir"],
        filelist=files["filelist"],
        design_file=files["design_file"],
    )

    geom_params, hfield = dataset[0]

    # Check types
    assert isinstance(geom_params, torch.Tensor)
    assert isinstance(hfield, torch.Tensor)

    # Check values
    assert geom_params.shape == (2,)
    assert hfield.shape == (1, 3)  # Assuming single channel

    # Validate NaN handling
    assert torch.equal(hfield, torch.tensor([[0.0, 0.0, 1.0]]).to(torch.float32))


def test_invalid_filelist(create_mock_files):
    """Test behavior with an invalid file list."""
    files = create_mock_files
    invalid_filelist = os.path.join(tempfile.gettempdir(), "invalid_filelist.txt")

    with pytest.raises(FileNotFoundError):
        LSC_cntr2hfield_DataSet(
            LSC_NPZ_DIR=files["npz_dir"],
            filelist=invalid_filelist,
            design_file=files["design_file"],
        )


def test_empty_dataset(create_mock_files):
    """Test behavior when file list is empty."""
    files = create_mock_files

    # Create an empty filelist
    with open(files["filelist"], "w") as f:
        f.truncate(0)

    dataset = LSC_cntr2hfield_DataSet(
        LSC_NPZ_DIR=files["npz_dir"],
        filelist=files["filelist"],
        design_file=files["design_file"],
    )

    assert len(dataset) == 0
