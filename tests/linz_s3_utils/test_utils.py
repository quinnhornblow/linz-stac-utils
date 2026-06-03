import numpy as np
import pytest
import xarray as xr

from linz_s3_utils.utils import last


def test_last_returns_last_non_null_value_and_coordinate():
    array = xr.DataArray(
        [[1.0, np.nan, 3.0], [np.nan, 5.0, np.nan]],
        dims=("y", "time"),
        coords={"time": [10, 20, 30], "y": [0, 1]},
    )

    result = last(array, dim="time", index_name="time_index", drop=False)

    assert result.dims == ("y",)
    assert result.sel(y=0).item() == 3.0
    assert result.sel(y=1).item() == 5.0
    assert result["time"].sel(y=0).item() == 30
    assert result["time"].sel(y=1).item() == 20
    assert result["time_index"].sel(y=0).item() == -1
    assert result["time_index"].sel(y=1).item() == -2


def test_last_masks_all_null_slices():
    array = xr.DataArray(
        [[np.nan, np.nan, np.nan], [1.0, np.nan, 2.0]],
        dims=("y", "time"),
        coords={"time": [10, 20, 30], "y": [0, 1]},
    )

    result = last(array, dim="time", index_name="time_index", drop=False)

    assert np.isnan(result.sel(y=0).item())
    assert np.isnan(result["time"].sel(y=0).item())
    assert np.isnan(result["time_index"].sel(y=0).item())
    assert result.sel(y=1).item() == 2.0


def test_last_raises_for_unknown_dimension():
    array = xr.DataArray([1.0, 2.0, 3.0], dims=("time",))

    with pytest.raises(ValueError, match="Dimension 'x' not found"):
        last(array, dim="x")
