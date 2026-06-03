import pytest
import xarray as xr

from linz_s3_utils.elevation import ElevationClient


@pytest.mark.skip(reason="Requires network access.")
def test_lidar_dem_loading():
    client = ElevationClient()
    ds = client.load_lidar_dem(resolution=1000)
    assert isinstance(ds, xr.Dataset)
