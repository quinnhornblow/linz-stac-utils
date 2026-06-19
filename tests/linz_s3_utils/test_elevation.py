import numpy as np
import pytest
import xarray as xr
from types import SimpleNamespace

from linz_s3_utils.elevation import (
    LIDAR_1M_DEM_COLLECTION_ID,
    ElevationClient,
    latest_elevation_surface,
)


class FakeCatalogClient:
    pass


def test_latest_elevation_surface_returns_last_non_null_time_slice():
    dataset = xr.Dataset(
        {
            "elevation": xr.DataArray(
                [[1.0, np.nan, 3.0], [np.nan, 5.0, np.nan]],
                dims=("y", "time"),
                coords={"time": [10, 20, 30], "y": [0, 1]},
            )
        }
    )

    result = latest_elevation_surface(dataset)

    assert isinstance(result, xr.DataArray)
    assert result.dims == ("y",)
    assert result.sel(y=0).item() == 3.0
    assert result.sel(y=1).item() == 5.0


def test_load_lidar_dem_uses_lidar_collection_and_returns_data_array(monkeypatch):
    dataset = xr.Dataset(
        {
            "elevation": xr.DataArray(
                [[1.0, np.nan, 3.0]],
                dims=("y", "time"),
                coords={"time": [10, 20, 30], "y": [0]},
            )
        }
    )

    captured: dict[str, object] = {}

    def fake_load(self, **kwargs):
        captured.update(kwargs)
        return dataset

    monkeypatch.setattr(ElevationClient, "load", fake_load)

    fake_catalog_client = FakeCatalogClient()
    client = ElevationClient(client=fake_catalog_client)
    result = client.load_lidar_dem(resolution=1000)

    assert isinstance(result, xr.DataArray)
    assert client.client is fake_catalog_client
    assert captured == {
        "collections": [LIDAR_1M_DEM_COLLECTION_ID],
        "resampling": "bilinear",
        "resolution": 1000,
        "progress": pytest.importorskip("tqdm").tqdm,
    }
    assert result.sel(y=0).item() == 3.0


@pytest.mark.integration
def test_lidar_dem_loading():
    client = ElevationClient()
    data = client.load_lidar_dem(resolution=1000)
    assert isinstance(data, xr.DataArray)
