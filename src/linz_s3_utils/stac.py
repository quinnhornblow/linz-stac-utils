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

DEFAULT_CACHE_PATH = Path(__file__).parent / "stac_cache.sqlite"
DEFAULT_CACHE_EXPIRY_SECONDS = 86400


class CatalogURLs(Enum):  # noqa: D101
    ELEVATION = "https://nz-elevation.s3-ap-southeast-2.amazonaws.com/catalog.json"


class LINZCollection(BaseModel):  # noqa: D101
    id: str
    title: str
    linz_geospatial_category: Literal["dem"]


def build_stac_io(
    cache_path: Path = DEFAULT_CACHE_PATH,
    expire_after: int = DEFAULT_CACHE_EXPIRY_SECONDS,
) -> StacApiIO:
    """Build a STAC IO instance backed by a cached requests session."""
    stac_io = StacApiIO()
    stac_io.session = requests_cache.CachedSession(
        cache_name=str(cache_path),
        expire_after=expire_after,
    )
    return stac_io


DEFAULT_STAC_IO = build_stac_io()


class StacCatalogClient:
    """Search a STAC catalog with simple local filtering."""

    def __init__(
        self,
        catalog: Literal["elevation"] = "elevation",
        stac_io: StacApiIO | None = None,
        client: Any | None = None,
    ):
        """Initialize a static LINZ STAC catalog client.

        Args:
            catalog: LINZ catalog to open.
            stac_io: STAC IO configuration used when opening the catalog.
            client: Preconfigured catalog client, primarily for testing.
        """
        self.catalog = catalog
        self.stac_io = DEFAULT_STAC_IO if stac_io is None else stac_io
        self.client = (
            Client.open(CatalogURLs[catalog.upper()].value, stac_io=self.stac_io)
            if client is None
            else client
        )

    def search(
        self,
        limit: int | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        datetime: str | None = None,
        intersects: dict | None = None,
        ids: list[str] | None = None,
        collections: list[str] | None = None,
    ) -> Iterator[Item]:
        """Return items from explicitly selected collections.

        This client currently supports loading full collections from a static
        catalog. Other STAC search parameters are accepted for API
        compatibility but are not implemented.

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
        unsupported_parameters = {
            "limit": limit,
            "bbox": bbox,
            "datetime": datetime,
            "intersects": intersects,
            "ids": ids,
        }
        for parameter_name, parameter_value in unsupported_parameters.items():
            if parameter_value is not None:
                msg = f"{parameter_name} is not implemented for static catalog search."
                raise NotImplementedError(msg)

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
        items = list(
            self.search(
                limit=limit,
                datetime=datetime,
                ids=ids,
                collections=collections,
            )
        )
        if not items:
            msg = "No items selected for loading. Provide one or more collections."
            raise ValueError(msg)

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
