# ruff: noqa: D103
from collections.abc import Iterator
from pathlib import Path

import pytest
import requests_cache
import xarray as xr
from pystac import Collection
from pystac_client.stac_api_io import StacApiIO

from linz_s3_utils.stac import StacCatalogClient, build_stac_io


def test_stac_catalog_client_instance():
    client = StacCatalogClient()
    assert isinstance(client.search(), Iterator)


def test_stac_load_requires_explicit_item_selection():
    client = StacCatalogClient()

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
    client = StacCatalogClient()

    with pytest.raises(NotImplementedError, match=parameter_name):
        list(client.search(**{parameter_name: parameter_value}))


def test_stac_invalid_catalog():
    with pytest.raises(KeyError):
        StacCatalogClient(catalog="invalid")  # ty:ignore[invalid-argument-type]


def test_build_stac_io_creates_cached_session(tmp_path: Path):
    stac_io = build_stac_io(cache_path=tmp_path / "stac.sqlite", expire_after=60)

    assert isinstance(stac_io, StacApiIO)
    assert isinstance(stac_io.session, requests_cache.CachedSession)


def test_stac_catalog_client_uses_injected_stac_io():
    injected_stac_io = build_stac_io(cache_path=Path("in-memory"), expire_after=60)

    client = StacCatalogClient(stac_io=injected_stac_io)

    assert client.client._stac_io is injected_stac_io


def test_stac_collection_metadata():
    client = StacCatalogClient()
    metadata = client._get_collection("01JE4ZZWAG19KPKRHYJJP02HC9")
    assert isinstance(metadata, Collection)
    assert metadata.id == "01JE4ZZWAG19KPKRHYJJP02HC9"


def test_stac_item_metadata():
    client = StacCatalogClient()
    metadata = client._get_item(
        client._get_collection("01JE4ZZWAG19KPKRHYJJP02HC9"), "AS21"
    )
    assert metadata is not None
    assert metadata.id == "AS21"
