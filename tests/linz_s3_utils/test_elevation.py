from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from odc.geo import CRS, Geometry
from odc.geo.geobox import GeoBox
from odc.geo.xr import wrap_xr
from shapely.geometry import Polygon

from linz_s3_utils.elevation import (
    LIDAR_1M_DEM_COLLECTION_ID,
    ElevationClient,
    latest_elevation_surface,
    load_elevation,
)

CONTOUR_8M_DEM_COLLECTION_ID = "01JE7NNKVY3QP5FPX2Q08DJQX5"


class FakeCatalogClient:
    pass


class FakeGeoInterface:
    def __init__(self, geometry, crs):
        self.__geo_interface__ = geometry
        self.crs = crs


def make_georegistered_array() -> xr.DataArray:
    geobox = GeoBox.from_bbox(
        (0.0, 0.0, 2.0, 2.0),
        crs="EPSG:4326",
        shape=(2, 2),
        tight=True,
    )
    return wrap_xr(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        geobox,
    ).rename("elevation")


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


def test_latest_elevation_surface_uses_the_latest_timestamp():
    dataset = xr.Dataset(
        {
            "elevation": xr.DataArray(
                [[10.0], [20.0]],
                dims=("time", "x"),
                coords={"time": ["2024-02-01", "2024-01-01"], "x": [0]},
            )
        }
    )

    result = latest_elevation_surface(dataset)

    assert result.item() == 10.0


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
    result = client.load_lidar_dem(
        chunks={"x": 128, "y": 128},
        crs="EPSG:4326",
        resolution=1000,
        bbox=(172.0, -43.0, 173.0, -42.0),
    )

    assert isinstance(result, xr.DataArray)
    assert client.client is fake_catalog_client
    assert captured == {
        "collections": [LIDAR_1M_DEM_COLLECTION_ID],
        "resampling": "bilinear",
        "chunks": {"x": 128, "y": 128},
        "crs": "EPSG:4326",
        "resolution": 1000,
        "bbox": (172.0, -43.0, 173.0, -42.0),
        "intersects": None,
        "progress": None,
    }
    assert result.sel(y=0).item() == 3.0


def test_load_lidar_dem_defaults_to_nztm_and_bilinear(monkeypatch):
    dataset = xr.Dataset(
        {
            "elevation": xr.DataArray(
                [[1.0]],
                dims=("y", "time"),
                coords={"time": [10], "y": [0]},
            )
        }
    )
    captured: dict[str, object] = {}

    def fake_load(self, **kwargs):
        captured.update(kwargs)
        return dataset

    monkeypatch.setattr(ElevationClient, "load", fake_load)

    ElevationClient(client=FakeCatalogClient()).load_lidar_dem()

    assert captured == {
        "collections": [LIDAR_1M_DEM_COLLECTION_ID],
        "resampling": "bilinear",
        "chunks": None,
        "crs": "EPSG:2193",
        "resolution": None,
        "bbox": None,
        "intersects": None,
        "progress": None,
    }


def test_load_lidar_dem_accepts_positional_resolution(monkeypatch):
    dataset = xr.Dataset(
        {
            "elevation": xr.DataArray(
                [[1.0]],
                dims=("y", "time"),
                coords={"time": [10], "y": [0]},
            )
        }
    )
    captured: dict[str, object] = {}

    def fake_load(self, **kwargs):
        captured.update(kwargs)
        return dataset

    monkeypatch.setattr(ElevationClient, "load", fake_load)

    ElevationClient(client=FakeCatalogClient()).load_lidar_dem(1000)

    assert captured["resolution"] == 1000


def test_load_dem_prefers_lidar_and_uses_contour_for_nulls(monkeypatch):
    captured: dict[str, object] = {}
    surfaces = {
        LIDAR_1M_DEM_COLLECTION_ID: (
            np.datetime64("2009-01-01"),
            np.array([100.0, np.nan, 0.0, np.nan]),
        ),
        f"{LIDAR_1M_DEM_COLLECTION_ID}-newer": (
            np.datetime64("2020-01-01"),
            np.array([np.nan, 300.0, np.nan, np.nan]),
        ),
        CONTOUR_8M_DEM_COLLECTION_ID: (
            np.datetime64("2011-12-31"),
            np.array([10.0, 20.0, 30.0, 40.0]),
        ),
    }

    def fake_load(self, **kwargs):
        captured.update(kwargs)
        groupby = kwargs["groupby"]
        ordered = sorted(
            surfaces.items(),
            key=lambda entry: groupby(
                type("Item", (), {"collection_id": entry[0].removesuffix("-newer")})(),
                type("ParsedItem", (), {"nominal_datetime": entry[1][0]})(),
                0,
            ),
        )
        return xr.Dataset(
            {
                "elevation": xr.DataArray(
                    np.stack([surface for _, (_, surface) in ordered]),
                    dims=("time", "x"),
                    coords={"time": [timestamp for _, (timestamp, _) in ordered]},
                )
            }
        )

    monkeypatch.setattr(ElevationClient, "load", fake_load)

    result = ElevationClient(client=FakeCatalogClient()).load_dem(
        10,
        bbox=(172.0, -43.0, 173.0, -42.0),
    )

    assert np.array_equal(
        result.values,
        np.array([100.0, 300.0, 0.0, 40.0]),
        equal_nan=True,
    )
    assert captured["collections"] == [
        LIDAR_1M_DEM_COLLECTION_ID,
        CONTOUR_8M_DEM_COLLECTION_ID,
    ]
    assert captured["resolution"] == 10


def test_load_dem_requires_resolution():
    client = ElevationClient(client=FakeCatalogClient())

    with pytest.raises(ValueError, match="Provide resolution."):
        client.load_dem()


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"output_path": Path("elevation.tif")},
    ],
)
def test_load_elevation_requires_bbox_or_intersects(monkeypatch, kwargs):
    def fake_load_dem(self, **kwargs):
        pytest.fail("load_dem should not be called")

    monkeypatch.setattr(ElevationClient, "load_dem", fake_load_dem, raising=False)
    monkeypatch.setattr(ElevationClient, "__init__", lambda self: None)

    with pytest.raises(ValueError, match="Provide bbox or intersects."):
        load_elevation(**kwargs)


def test_load_elevation_requires_resolution(monkeypatch):
    def fail_init(self):
        pytest.fail("ElevationClient should not be initialized")

    monkeypatch.setattr(ElevationClient, "__init__", fail_init)

    with pytest.raises(ValueError, match="Provide resolution."):
        load_elevation(bbox=(172.0, -43.0, 173.0, -42.0))


@pytest.mark.parametrize(
    ("intersects", "expected_crs"),
    [
        (Geometry(Polygon([(0, 0), (1, 0), (0, 1)]), "EPSG:2193"), CRS("EPSG:2193")),
        (Polygon([(0, 0), (1, 0), (0, 1)]), CRS("EPSG:4326")),
        (
            {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [0, 1], [0, 0]]],
            },
            CRS("EPSG:4326"),
        ),
        (
            FakeGeoInterface(
                {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [0, 1], [0, 0]]],
                },
                "EPSG:2193",
            ),
            CRS("EPSG:2193"),
        ),
    ],
)
def test_load_elevation_normalizes_intersects_and_delegates(
    monkeypatch, intersects, expected_crs
):
    data = make_georegistered_array()
    captured: dict[str, object] = {}

    def fake_load_dem(self, **kwargs):
        captured.update(kwargs)
        return data

    def fail_load_lidar_dem(self, **kwargs):
        pytest.fail("load_lidar_dem should not be called")

    def fake_crop(array, polygon, apply_mask, all_touched):
        captured["crop"] = (array, polygon, apply_mask, all_touched)
        return array

    monkeypatch.setattr(ElevationClient, "load_dem", fake_load_dem, raising=False)
    monkeypatch.setattr(ElevationClient, "load_lidar_dem", fail_load_lidar_dem)
    monkeypatch.setattr(ElevationClient, "__init__", lambda self: None)
    monkeypatch.setattr("linz_s3_utils.elevation.crop", fake_crop)

    result = load_elevation(
        intersects=intersects,
        crs="EPSG:2193",
        resolution=25,
    )

    assert result is data
    assert isinstance(captured["intersects"], Geometry)
    assert captured["intersects"].crs == expected_crs
    assert captured["crs"] == "EPSG:2193"
    assert captured["resolution"] == 25
    assert captured["crop"] == (data, captured["intersects"], True, True)


def test_load_elevation_masks_intersects_and_writes_cog(monkeypatch, tmp_path: Path):
    data = make_georegistered_array()
    polygon = Polygon(
        [(0.0, 0.0), (2.0, 0.0), (2.0, 0.1), (0.1, 0.1), (0.1, 2.0), (0.0, 2.0)]
    )
    load_kwargs: dict[str, object] = {}
    cog_call: dict[str, object] = {}

    def fake_load_dem(self, **kwargs):
        load_kwargs.update(kwargs)
        return data

    def fail_load_lidar_dem(self, **kwargs):
        pytest.fail("load_lidar_dem should not be called")

    def fake_write_cog(array, output_path, overwrite=False):
        cog_call["array"] = array
        cog_call["output_path"] = output_path
        cog_call["overwrite"] = overwrite
        return output_path

    monkeypatch.setattr(ElevationClient, "load_dem", fake_load_dem, raising=False)
    monkeypatch.setattr(ElevationClient, "load_lidar_dem", fail_load_lidar_dem)
    monkeypatch.setattr(ElevationClient, "__init__", lambda self: None)
    monkeypatch.setattr("linz_s3_utils.elevation.write_cog", fake_write_cog)

    output_path = tmp_path / "elevation.tif"
    result = load_elevation(
        intersects=polygon,
        crs="EPSG:4326",
        resolution=10,
        output_path=output_path,
        overwrite=True,
    )

    assert isinstance(load_kwargs["intersects"], Geometry)
    assert np.array_equal(
        np.isnan(result.values),
        np.array([[False, True], [False, False]]),
    )
    assert np.array_equal(
        result.values,
        np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32),
        equal_nan=True,
    )
    assert cog_call["array"] is result
    assert cog_call["output_path"] == output_path
    assert cog_call["overwrite"] is True


def test_load_elevation_computes_delayed_cog_write(monkeypatch, tmp_path: Path):
    data = make_georegistered_array().chunk({"latitude": 1, "longitude": 1})
    computed = False

    class FakeDelayedWrite:
        def compute(self):
            nonlocal computed
            computed = True

    def fake_load_dem(self, **kwargs):
        return data

    def fail_load_lidar_dem(self, **kwargs):
        pytest.fail("load_lidar_dem should not be called")

    def fake_write_cog(array, output_path, overwrite=False):
        assert array.chunks is not None
        return FakeDelayedWrite()

    monkeypatch.setattr(ElevationClient, "load_dem", fake_load_dem, raising=False)
    monkeypatch.setattr(ElevationClient, "load_lidar_dem", fail_load_lidar_dem)
    monkeypatch.setattr(ElevationClient, "__init__", lambda self: None)
    monkeypatch.setattr("linz_s3_utils.elevation.write_cog", fake_write_cog)

    load_elevation(
        bbox=(172.0, -43.0, 173.0, -42.0),
        chunks={"x": 1, "y": 1},
        resolution=10,
        output_path=tmp_path / "elevation.tif",
    )

    assert computed


def test_load_elevation_returns_unmasked_result_without_intersects(monkeypatch):
    data = make_georegistered_array()
    data[:] = np.nan
    captured: dict[str, object] = {}

    def fake_load_dem(self, **kwargs):
        captured.update(kwargs)
        return data

    def fail_load_lidar_dem(self, **kwargs):
        pytest.fail("load_lidar_dem should not be called")

    monkeypatch.setattr(ElevationClient, "load_dem", fake_load_dem, raising=False)
    monkeypatch.setattr(ElevationClient, "load_lidar_dem", fail_load_lidar_dem)
    monkeypatch.setattr(ElevationClient, "__init__", lambda self: None)

    result = load_elevation(
        bbox=(172.0, -43.0, 173.0, -42.0),
        resolution=10,
    )

    assert result is data
    assert np.isnan(result.values).all()
    assert captured == {
        "resampling": "bilinear",
        "chunks": None,
        "crs": "EPSG:2193",
        "resolution": 10,
        "bbox": (172.0, -43.0, 173.0, -42.0),
        "intersects": None,
        "progress": None,
    }


@pytest.mark.integration
def test_lidar_dem_loading():
    client = ElevationClient()
    data = client.load_lidar_dem(
        bbox=(173.5264, -41.3096, 173.5364, -41.2996),
        resolution=1000,
    )
    assert isinstance(data, xr.DataArray)


@pytest.mark.integration
def test_dem_loading_falls_back_to_contour_when_lidar_is_unavailable():
    bbox = (167.1030, -46.0470, 167.1080, -46.0430)
    client = ElevationClient()

    assert not list(client.search(collections=[LIDAR_1M_DEM_COLLECTION_ID], bbox=bbox))

    data = client.load_dem(bbox=bbox, resolution=8)

    assert data.notnull().any().item()
