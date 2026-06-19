# ruff: noqa: D103
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import linz_s3_utils.stac as stac_module
import pytest
import requests_cache
import xarray as xr
from pystac import Collection
from pystac_client.stac_api_io import StacApiIO

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

    client = StacCatalogClient(client=FakeCatalogClient({"lidar": FakeCollection([item])}))
    result = client.load(collections=["lidar"], resolution=250)

    assert captured["items"] == [item]
    assert captured["resolution"] == 250
    assert isinstance(result, xr.Dataset)
    assert list(result.data_vars) == ["elevation"]


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


def test_build_stac_io_creates_cached_session(tmp_path: Path):
    stac_io = build_stac_io(cache_path=tmp_path / "stac.sqlite", expire_after=60)

    assert isinstance(stac_io, StacApiIO)
    assert isinstance(stac_io.session, requests_cache.CachedSession)


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
    metadata = client._get_item(client._get_collection("01JE4ZZWAG19KPKRHYJJP02HC9"), "AS21")
    assert metadata is not None
    assert metadata.id == "AS21"
