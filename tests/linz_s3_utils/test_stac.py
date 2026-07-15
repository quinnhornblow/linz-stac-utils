# ruff: noqa: D103
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests_cache
import xarray as xr
from platformdirs import user_cache_path
from pystac import Collection
from pystac_client.stac_api_io import StacApiIO

import linz_s3_utils.stac as stac_module
from linz_s3_utils.stac import StacCatalogClient, build_stac_io


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


def test_stac_search_returns_items_from_requested_collections():
    item_a = SimpleNamespace(id="AS21")
    item_b = SimpleNamespace(id="AS22")
    client = StacCatalogClient(
        client=FakeCatalogClient({"lidar": FakeCollection([item_a, item_b])})
    )

    search_result = client.search(collections=["lidar"])

    assert isinstance(search_result, Iterator)
    assert list(search_result) == [item_a, item_b]


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
    client.load(collections=["lidar"], **{parameter_name: parameter_value})

    assert captured["items"] == [item]
    assert captured[parameter_name] == parameter_value


def test_stac_load_requires_explicit_item_selection():
    client = StacCatalogClient(client=FakeCatalogClient({}))

    with pytest.raises(ValueError, match="No items selected"):
        client.load()


@pytest.mark.parametrize(
    ("parameter_name", "parameter_value"),
    [
        ("limit", 1),
        ("bbox", (0.0, 0.0, 1.0, 1.0)),
        ("datetime", "2024-01-01/2024-12-31"),
        ("intersects", {"type": "Point", "coordinates": [0.0, 0.0]}),
        ("ids", ["AS21"]),
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
        user_cache_path("linz-s3-utils", appauthor=False) / "stac.sqlite"
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
