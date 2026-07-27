# ruff: noqa: D103
import gc
import weakref
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests_cache
import xarray as xr
from odc.geo import Geometry
from platformdirs import user_cache_path
from pystac import Collection, Item
from pystac_client.stac_api_io import StacApiIO
from shapely.geometry import Point, Polygon

import linz_stac_utils.stac as stac_module
from linz_stac_utils.stac import StacCatalogClient, build_stac_io

REGIONAL_POLYGON = Polygon(
    [
        (172.4, -43.6),
        (172.6, -43.6),
        (172.6, -43.4),
        (172.4, -43.4),
    ]
)


class FakeCollection:
    def __init__(self, items):
        self._items = {item.id: item for item in items}

    def get_items(self):
        return iter(self._items.values())

    def get_item(self, item_id):
        return self._items.get(item_id)


class FakeCatalogClient:
    def __init__(self, collections):
        self._collections = collections

    def get_collection(self, collection_id):
        return self._collections[collection_id]


class FalseyCatalogClient(FakeCatalogClient):
    def __bool__(self):
        return False


class FakeGeoInterface:
    def __init__(self, geometry, crs=None):
        self.__geo_interface__ = geometry
        self.crs = crs


def make_item(item_id, *, geometry=None, bbox=None):
    return Item(
        id=item_id,
        geometry=geometry,
        bbox=bbox,
        datetime=datetime(2024, 1, 1, tzinfo=UTC),
        properties={},
    )


def test_stac_search_returns_items_from_requested_collections():
    item_a = SimpleNamespace(id="AS21")
    item_b = SimpleNamespace(id="AS22")
    client = StacCatalogClient(
        client=FakeCatalogClient({"lidar": FakeCollection([item_a, item_b])})
    )

    search_result = client.search(collections=["lidar"])

    assert isinstance(search_result, Iterator)
    assert list(search_result) == [item_a, item_b]


def test_stac_search_applies_global_limit_after_item_filters():
    item_a = make_item(
        "AS21",
        geometry=Point(3.0, 3.0).__geo_interface__,
        bbox=[3.0, 3.0, 3.0, 3.0],
    )
    item_b = make_item(
        "AS22",
        geometry=Point(1.0, 1.0).__geo_interface__,
        bbox=[1.0, 1.0, 1.0, 1.0],
    )
    item_c = make_item(
        "AS23",
        geometry=Point(1.5, 1.5).__geo_interface__,
        bbox=[1.5, 1.5, 1.5, 1.5],
    )
    client = StacCatalogClient(
        client=FakeCatalogClient(
            {
                "first": FakeCollection([item_a, item_b]),
                "second": FakeCollection([item_c]),
            }
        )
    )

    result = client.search(
        collections=["first", "second"],
        ids=["AS21", "AS22", "AS23"],
        bbox=(0.0, 0.0, 2.0, 2.0),
        limit=1,
    )

    assert list(result) == [item_b]


def test_stac_search_returns_empty_iterator_for_unknown_ids():
    item = SimpleNamespace(id="AS21")
    client = StacCatalogClient(
        client=FakeCatalogClient({"lidar": FakeCollection([item])})
    )

    result = client.search(collections=["lidar"], ids=["missing"])

    assert list(result) == []


@pytest.mark.parametrize("limit", [0, -1])
def test_stac_search_requires_positive_limit(limit):
    client = StacCatalogClient(client=FakeCatalogClient({}))

    with pytest.raises(ValueError, match="limit must be positive"):
        list(client.search(limit=limit))


def test_stac_search_filters_bbox_using_geometry_and_bbox_fallback():
    inside = make_item(
        "inside",
        geometry=Point(1.0, 1.0).__geo_interface__,
        bbox=[1.0, 1.0, 1.0, 1.0],
    )
    outside = make_item(
        "outside",
        geometry=Point(3.0, 3.0).__geo_interface__,
        bbox=[3.0, 3.0, 3.0, 3.0],
    )
    boundary = make_item(
        "boundary",
        geometry=Point(2.0, 1.0).__geo_interface__,
        bbox=[2.0, 1.0, 2.0, 1.0],
    )
    bbox_fallback = make_item("bbox-fallback", bbox=[0.5, 0.5, 1.5, 1.5])
    invalid_geometry = make_item(
        "invalid-geometry",
        geometry={"type": "Invalid", "coordinates": []},
        bbox=[0.5, 0.5, 0.0, 1.5, 1.5, 10.0],
    )
    empty_geometry = make_item(
        "empty-geometry",
        geometry=Polygon().__geo_interface__,
        bbox=[0.5, 0.5, 1.5, 1.5],
    )
    geometry_preferred = make_item(
        "geometry-preferred",
        geometry=Point(3.0, 3.0).__geo_interface__,
        bbox=[0.5, 0.5, 1.5, 1.5],
    )
    missing = make_item("missing")
    client = StacCatalogClient(
        client=FakeCatalogClient(
            {
                "lidar": FakeCollection(
                    [
                        inside,
                        outside,
                        boundary,
                        bbox_fallback,
                        invalid_geometry,
                        empty_geometry,
                        geometry_preferred,
                        missing,
                    ]
                )
            }
        )
    )

    result = client.search(collections=["lidar"], bbox=(0.0, 0.0, 2.0, 2.0))

    assert [item.id for item in result] == [
        "inside",
        "boundary",
        "bbox-fallback",
        "invalid-geometry",
        "empty-geometry",
    ]


def test_stac_search_includes_items_without_spatial_metadata_without_spatial_filter():
    missing = make_item("missing")
    client = StacCatalogClient(
        client=FakeCatalogClient({"lidar": FakeCollection([missing])})
    )

    result = client.search(collections=["lidar"])

    assert list(result) == [missing]


def test_stac_search_uses_exact_intersects_geometry():
    query = Polygon(
        shell=[(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)],
        holes=[[(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)]],
    )
    matching = make_item(
        "matching",
        geometry=Point(0.5, 0.5).__geo_interface__,
        bbox=[0.5, 0.5, 0.5, 0.5],
    )
    in_hole = make_item(
        "in-hole",
        geometry=Point(2.0, 2.0).__geo_interface__,
        bbox=[2.0, 2.0, 2.0, 2.0],
    )
    client = StacCatalogClient(
        client=FakeCatalogClient({"lidar": FakeCollection([matching, in_hole])})
    )

    result = client.search(collections=["lidar"], intersects=query)

    assert list(result) == [matching]


@pytest.mark.parametrize(
    "intersects",
    [
        REGIONAL_POLYGON,
        REGIONAL_POLYGON.__geo_interface__,
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": REGIONAL_POLYGON.__geo_interface__,
                }
            ],
        },
        Geometry(REGIONAL_POLYGON, "EPSG:4326"),
        Geometry(REGIONAL_POLYGON, "EPSG:4326").to_crs("EPSG:2193"),
        FakeGeoInterface(
            REGIONAL_POLYGON.__geo_interface__,
            "EPSG:4326",
        ),
        FakeGeoInterface(
            Geometry(REGIONAL_POLYGON, "EPSG:4326")
            .to_crs("EPSG:2193")
            .__geo_interface__,
            "EPSG:2193",
        ),
    ],
)
def test_stac_search_accepts_load_elevation_intersects_types(intersects):
    item = make_item(
        "inside",
        geometry=Point(172.5, -43.5).__geo_interface__,
        bbox=[172.5, -43.5, 172.5, -43.5],
    )
    client = StacCatalogClient(
        client=FakeCatalogClient({"lidar": FakeCollection([item])})
    )

    result = client.search(collections=["lidar"], intersects=intersects)

    assert list(result) == [item]


def test_stac_search_rejects_bbox_with_intersects():
    client = StacCatalogClient(client=FakeCatalogClient({}))

    with pytest.raises(ValueError, match="bbox and intersects are mutually exclusive"):
        list(
            client.search(
                bbox=(0.0, 0.0, 1.0, 1.0),
                intersects=Point(0.5, 0.5),
            )
        )


def test_stac_search_rejects_unrecognized_intersects():
    client = StacCatalogClient(client=FakeCatalogClient({}))

    with pytest.raises(ValueError, match="Can't interpret intersects as geometry"):
        list(client.search(intersects=object()))


def test_stac_load_passes_selected_items_to_odc_stac_load(monkeypatch):
    item = SimpleNamespace(id="AS21")
    dataset = xr.Dataset({"visual": xr.DataArray([1.0], dims=("y",))})
    captured: dict[str, object] = {}

    def fake_load(items, **kwargs):
        captured["items"] = list(items)
        captured.update(kwargs)
        return dataset

    monkeypatch.setattr(stac_module.odc.stac, "load", fake_load)

    client = StacCatalogClient(
        client=FakeCatalogClient({"lidar": FakeCollection([item])})
    )
    result = client.load(collections=["lidar"], resolution=250)

    assert captured["items"] == [item]
    assert captured["resolution"] == 250
    assert isinstance(result, xr.Dataset)
    assert list(result.data_vars) == ["elevation"]


def test_stac_load_passes_groupby_to_odc_stac_load(monkeypatch):
    item = SimpleNamespace(id="AS21")
    dataset = xr.Dataset({"visual": xr.DataArray([1.0], dims=("y",))})
    captured: dict[str, object] = {}

    def fake_load(items, **kwargs):
        captured["items"] = list(items)
        captured.update(kwargs)
        return dataset

    monkeypatch.setattr(stac_module.odc.stac, "load", fake_load)
    client = StacCatalogClient(
        client=FakeCatalogClient({"lidar": FakeCollection([item])})
    )

    client.load(collections=["lidar"], groupby="id")

    assert captured["items"] == [item]
    assert captured["groupby"] == "id"


@pytest.mark.parametrize(
    ("parameter_name", "parameter_value"),
    [
        ("bbox", (172.0, -43.0, 173.0, -42.0)),
        ("intersects", {"type": "Point", "coordinates": [172.5, -42.5]}),
    ],
)
def test_stac_load_passes_output_bounds_to_odc_stac_load(
    monkeypatch, parameter_name, parameter_value
):
    item = make_item(
        "inside",
        geometry=Point(172.5, -42.5).__geo_interface__,
        bbox=[172.5, -42.5, 172.5, -42.5],
    )
    outside = make_item(
        "outside",
        geometry=Point(174.0, -41.0).__geo_interface__,
        bbox=[174.0, -41.0, 174.0, -41.0],
    )
    dataset = xr.Dataset({"visual": xr.DataArray([1.0], dims=("y",))})
    captured: dict[str, object] = {}

    def fake_load(items, **kwargs):
        captured["items"] = list(items)
        captured.update(kwargs)
        return dataset

    monkeypatch.setattr(stac_module.odc.stac, "load", fake_load)

    client = StacCatalogClient(
        client=FakeCatalogClient({"lidar": FakeCollection([item, outside])})
    )
    client.load(collections=["lidar"], **{parameter_name: parameter_value})

    assert captured["items"] == [item]
    assert captured[parameter_name] == parameter_value


def test_stac_load_rejects_bbox_with_intersects_before_loading(monkeypatch):
    def fail_load(*args, **kwargs):
        pytest.fail("odc.stac.load should not be called")

    monkeypatch.setattr(stac_module.odc.stac, "load", fail_load)
    client = StacCatalogClient(client=FakeCatalogClient({}))

    with pytest.raises(ValueError, match="bbox and intersects are mutually exclusive"):
        client.load(
            bbox=(0.0, 0.0, 1.0, 1.0),
            intersects=Point(0.5, 0.5),
        )


def test_stac_load_does_not_load_when_spatial_filter_matches_no_items(monkeypatch):
    item = make_item(
        "outside",
        geometry=Point(3.0, 3.0).__geo_interface__,
        bbox=[3.0, 3.0, 3.0, 3.0],
    )

    def fail_load(*args, **kwargs):
        pytest.fail("odc.stac.load should not be called")

    monkeypatch.setattr(stac_module.odc.stac, "load", fail_load)
    client = StacCatalogClient(
        client=FakeCatalogClient({"lidar": FakeCollection([item])})
    )

    with pytest.raises(ValueError, match="No items match"):
        client.load(collections=["lidar"], bbox=(0.0, 0.0, 1.0, 1.0))


def test_stac_load_requires_explicit_item_selection():
    client = StacCatalogClient(client=FakeCatalogClient({}))

    with pytest.raises(ValueError, match="No items match"):
        client.load()


@pytest.mark.parametrize(
    ("parameter_name", "parameter_value"),
    [
        ("datetime", "2024-01-01/2024-12-31"),
    ],
)
def test_stac_search_rejects_unsupported_parameters(parameter_name, parameter_value):
    client = StacCatalogClient(client=FakeCatalogClient({}))

    with pytest.raises(NotImplementedError, match=parameter_name):
        list(client.search(**{parameter_name: parameter_value}))


def test_stac_invalid_catalog():
    with pytest.raises(KeyError):
        StacCatalogClient(catalog="invalid")  # ty:ignore[invalid-argument-type]


def test_default_cache_path_uses_user_cache_directory():
    assert stac_module.DEFAULT_CACHE_PATH == (
        user_cache_path("linz-stac-utils", appauthor=False) / "stac.sqlite"
    )


def test_build_stac_io_creates_cached_session_and_parent_directory(tmp_path: Path):
    cache_path = tmp_path / "cache" / "stac.sqlite"

    stac_io = build_stac_io(cache_path=cache_path, expire_after=60)

    assert isinstance(stac_io, StacApiIO)
    assert isinstance(stac_io.session, requests_cache.CachedSession)
    assert cache_path.parent.is_dir()


def test_build_stac_io_expands_cache_path_before_creating_parent(
    monkeypatch, tmp_path: Path
):
    home_path = tmp_path / "home"
    work_path = tmp_path / "work"
    home_path.mkdir()
    work_path.mkdir()
    monkeypatch.setenv("HOME", str(home_path))
    monkeypatch.setenv("USERPROFILE", str(home_path))
    monkeypatch.chdir(work_path)

    stac_io = build_stac_io(cache_path=Path("~/cache/stac.sqlite"), expire_after=60)

    assert isinstance(stac_io.session, requests_cache.CachedSession)
    assert (home_path / "cache").is_dir()
    assert not (work_path / "~").exists()


def test_build_stac_io_can_disable_caching():
    stac_io = build_stac_io(cache=False)

    assert isinstance(stac_io, StacApiIO)
    assert not isinstance(stac_io.session, requests_cache.CachedSession)


def test_stac_catalog_client_builds_default_stac_io_when_initialized(monkeypatch):
    default_stac_io = StacApiIO()

    monkeypatch.setattr(stac_module, "build_stac_io", lambda: default_stac_io)

    client = StacCatalogClient(client=FakeCatalogClient({}))

    assert client.stac_io is default_stac_io


def test_stac_catalog_client_uses_injected_stac_io(monkeypatch):
    injected_stac_io = build_stac_io(cache_path=Path("in-memory"), expire_after=60)

    opened: dict[str, object] = {}

    def fake_open(url, stac_io):
        opened["url"] = url
        opened["stac_io"] = stac_io
        return SimpleNamespace(_stac_io=stac_io)

    monkeypatch.setattr(stac_module.Client, "open", fake_open)

    client = StacCatalogClient(stac_io=injected_stac_io)

    assert client.client._stac_io is injected_stac_io
    assert opened == {
        "url": stac_module.CatalogURLs.ELEVATION.value,
        "stac_io": injected_stac_io,
    }


def test_stac_catalog_client_uses_falsey_injected_client(monkeypatch):
    injected_client = FalseyCatalogClient({})

    def fail_open(*args, **kwargs):
        pytest.fail("Client.open should not be called for an injected client")

    monkeypatch.setattr(stac_module.Client, "open", fail_open)

    client = StacCatalogClient(client=injected_client)

    assert client.client is injected_client


def test_stac_get_collection_uses_injected_client():
    collection = FakeCollection([])
    client = StacCatalogClient(client=FakeCatalogClient({"lidar": collection}))

    metadata = client._get_collection("lidar")

    assert metadata is collection


def test_stac_metadata_caches_do_not_keep_client_alive():
    item = SimpleNamespace(id="AS21")
    collection = FakeCollection([item])
    client = StacCatalogClient(client=FakeCatalogClient({"lidar": collection}))
    client_ref = weakref.ref(client)

    client._get_collection("lidar")
    client._get_item(collection, "AS21")
    del client
    gc.collect()

    assert client_ref() is None


def test_stac_get_item_reads_from_collection():
    item = SimpleNamespace(id="AS21")
    collection = FakeCollection([item])
    client = StacCatalogClient(client=FakeCatalogClient({"lidar": collection}))

    metadata = client._get_item(collection, "AS21")

    assert metadata is item


@pytest.mark.integration
def test_stac_collection_metadata():
    client = StacCatalogClient()
    metadata = client._get_collection("01JE4ZZWAG19KPKRHYJJP02HC9")
    assert isinstance(metadata, Collection)
    assert metadata.id == "01JE4ZZWAG19KPKRHYJJP02HC9"


@pytest.mark.integration
def test_stac_item_metadata():
    client = StacCatalogClient()
    metadata = client._get_item(
        client._get_collection("01JE4ZZWAG19KPKRHYJJP02HC9"), "AS21"
    )
    assert metadata is not None
    assert metadata.id == "AS21"
