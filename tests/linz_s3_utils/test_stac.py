# ruff: noqa: D103
from collections.abc import Iterator

import pytest
import xarray as xr
from pystac import Collection

from linz_s3_utils.stac import StacCatalogClient


def test_stac_catalog_client_instance():
    client = StacCatalogClient()
    assert isinstance(client.search(), Iterator)
    assert isinstance(client.load(), xr.Dataset)


def test_stac_invalid_catalog():
    with pytest.raises(KeyError):
        StacCatalogClient(catalog="invalid")  # ty:ignore[invalid-argument-type]


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
