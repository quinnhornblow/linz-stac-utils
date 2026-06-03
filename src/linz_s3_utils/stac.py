from collections.abc import Iterator
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from warnings import filterwarnings

import odc.stac
import requests_cache
import xarray as xr
from pydantic import BaseModel
from pystac import Collection
from pystac.item import Item
from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO
from pystac_client.warnings import FallbackToPystac, NoConformsTo

filterwarnings("ignore", category=NoConformsTo)
filterwarnings("ignore", category=FallbackToPystac)


class CatalogURLs(Enum):  # noqa: D101
    ELEVATION = "https://nz-elevation.s3-ap-southeast-2.amazonaws.com/catalog.json"


class LINZCollection(BaseModel):  # noqa: D101
    id: str
    title: str
    linz_geospatial_category: Literal["dem"]


stac_io = StacApiIO()
cache_file = Path(__file__).parent / "stac_cache.sqlite"
stac_io.session = requests_cache.CachedSession(
    cache_name=str(cache_file),
    expire_after=86400,  # Cache expires after 1 day
)


class StacCatalogClient:
    """Search a STAC catalog with simple local filtering."""

    def __init__(self, catalog: Literal["elevation"] = "elevation"):  # noqa: D107
        self.catalog = catalog
        self.client = Client.open(CatalogURLs[catalog.upper()].value, stac_io=stac_io)

    def search(
        self,
        limit: int | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        datetime: str | None = None,
        intersects: dict | None = None,
        ids: list[str] | None = None,
        collections: list[str] | None = None,
    ) -> Iterator[Item]:
        """Mimic `pystac_client.Client.search` on a STAC catalog.

        Additional references: https://github.com/radiantearth/stac-api-spec/tree/release/v1.0.0/item-search

        Args:
            limit: Maximum number of items to return.
            bbox: Requested bounding box.
            datetime: Single date+time, or a range ('/' separator). Use double dots .. for open date ranges.
            intersects: Searches items by performing intersection between their geometry and provided GeoJSON geometry.
            ids: Array of Item ids to return.
            collections: Array of one or more Collection IDs that each matching Item must be in.

        Returns:
            An iterator of `pystac.Item` objects that match the search criteria.
        """
        items = []
        if collections:
            for collection_id in collections:
                collection = self._get_collection(collection_id)
                items.extend(collection.get_items())
        return iter(items)

    def load(
        self,
        limit: int | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        datetime: str | None = None,
        intersects: Any = None,
        ids: list[str] | None = None,
        collections: list[str] | None = None,
        resampling: str | dict[str, str] | None = None,
        chunks: dict[str, int | Literal["auto"]] | None = None,
        crs: str = "EPSG:2193",
        resolution: int = 100,
        progress: Any = None,
    ) -> xr.Dataset:
        """Mimic `odc.stac.load` on a STAC catalog."""
        items = self.search(
            limit=limit,
            bbox=bbox,
            datetime=datetime,
            intersects=intersects,
            ids=ids,
            collections=collections,
        )
        ds = odc.stac.load(
            items,
            resampling=resampling,
            chunks=chunks,
            crs=crs,
            resolution=resolution,
            bbox=bbox,
            intersects=intersects,
            progress=progress,
        )
        ds = ds.rename({"visual": self.catalog})
        return ds

    @lru_cache(maxsize=None)
    def _get_collection(self, collection_id: str) -> Collection:
        """Get metadata for a collection."""
        return self.client.get_collection(collection_id)

    @lru_cache(maxsize=None)
    def _get_item(self, collection: Collection, item_id: str) -> Item | None:
        """Get metadata for an item."""
        return collection.get_item(item_id)
