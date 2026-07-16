from collections.abc import Iterator
from enum import Enum
from functools import lru_cache
from itertools import chain, islice
from pathlib import Path
from typing import Any, Literal
from warnings import filterwarnings

import odc.stac
import requests_cache
import xarray as xr
from odc.geo import Geometry, MaybeCRS, SomeResolution
from platformdirs import user_cache_path
from pydantic import BaseModel
from pystac import Collection
from pystac.item import Item
from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO
from pystac_client.warnings import FallbackToPystac, NoConformsTo
from shapely.errors import ShapelyError
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry

filterwarnings("ignore", category=NoConformsTo)
filterwarnings("ignore", category=FallbackToPystac)

DEFAULT_CACHE_PATH = user_cache_path("linz-s3-utils", appauthor=False) / "stac.sqlite"
DEFAULT_CACHE_EXPIRY_SECONDS = 86400


class CatalogURLs(Enum):  # noqa: D101
    ELEVATION = "https://nz-elevation.s3-ap-southeast-2.amazonaws.com/catalog.json"


class LINZCollection(BaseModel):  # noqa: D101
    id: str
    title: str
    linz_geospatial_category: Literal["dem"]


def _geometry_from_intersects(intersects: Any) -> BaseGeometry:
    """Convert a supported intersects value to WGS84 Shapely geometry."""
    if isinstance(intersects, Geometry):
        return intersects.to_crs("EPSG:4326").geom
    if isinstance(intersects, BaseGeometry):
        return intersects
    if isinstance(intersects, dict):
        return Geometry(intersects, "EPSG:4326").geom

    geometry = getattr(intersects, "__geo_interface__", None)
    if geometry is None:
        msg = "Can't interpret intersects as geometry."
        raise ValueError(msg)
    crs = getattr(intersects, "crs", None) or "EPSG:4326"
    return Geometry(geometry, crs).to_crs("EPSG:4326").geom


def _geometry_from_item(item: Item) -> BaseGeometry | None:
    """Read an item's WGS84 geometry, falling back to its bounding box."""
    if item.geometry is not None:
        try:
            item_geometry = shape(item.geometry)
            if item_geometry.is_valid and not item_geometry.is_empty:
                return item_geometry
        except (ShapelyError, KeyError, TypeError, ValueError):
            pass

    if item.bbox is None:
        return None
    if len(item.bbox) == 4:
        west, south, east, north = item.bbox
    elif len(item.bbox) == 6:
        west, south, _, east, north, _ = item.bbox
    else:
        return None
    return box(west, south, east, north)


def build_stac_io(
    cache_path: Path = DEFAULT_CACHE_PATH,
    expire_after: int = DEFAULT_CACHE_EXPIRY_SECONDS,
    *,
    cache: bool = True,
) -> StacApiIO:
    """Build a STAC IO instance, optionally backed by a cached requests session."""
    stac_io = StacApiIO()
    if cache:
        cache_path = cache_path.expanduser()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        stac_io.session = requests_cache.CachedSession(
            cache_name=str(cache_path),
            expire_after=expire_after,
        )
    return stac_io


class StacCatalogClient:
    """Search and load explicitly selected static-catalog collections."""

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
        self.stac_io = build_stac_io() if stac_io is None else stac_io
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
        intersects: Any = None,
        ids: list[str] | None = None,
        collections: list[str] | None = None,
    ) -> Iterator[Item]:
        """Return locally filtered items from explicitly selected collections.

        Item geometry is preferred for spatial filtering, with item bounding
        boxes used as a fallback. Items without either are excluded from
        spatial searches. Boundary contact counts as an intersection.

        Args:
            limit: Positive maximum number of filtered items to return.
            bbox: WGS84 bounds as ``(west, south, east, north)``. Mutually
                exclusive with ``intersects``.
            datetime: Unsupported date or date range filter.
            intersects: WGS84 GeoJSON or Shapely geometry, or a CRS-aware ODC
                geometry or object implementing ``__geo_interface__``.
                Mutually exclusive with ``bbox``.
            ids: Item IDs to include.
            collections: Collection IDs whose items should be searched.

        Returns:
            An iterator of `pystac.Item` objects that match the search criteria.

        Raises:
            NotImplementedError: If ``datetime`` is provided.
            ValueError: If the spatial selectors conflict, ``limit`` is not
                positive, or ``intersects`` cannot be interpreted.
        """
        if bbox is not None and intersects is not None:
            msg = "bbox and intersects are mutually exclusive."
            raise ValueError(msg)

        if datetime is not None:
            msg = "datetime is not implemented for static catalog search."
            raise NotImplementedError(msg)

        if limit is not None and limit <= 0:
            msg = "limit must be positive."
            raise ValueError(msg)

        query_geometry = None
        if bbox is not None:
            query_geometry = box(*bbox)
        elif intersects is not None:
            query_geometry = _geometry_from_intersects(intersects)

        items = chain.from_iterable(
            self._get_collection(collection_id).get_items()
            for collection_id in collections or []
        )
        if ids is not None:
            item_ids = set(ids)
            items = (item for item in items if item.id in item_ids)
        if query_geometry is not None:
            items = (
                item
                for item in items
                if (item_geometry := _geometry_from_item(item)) is not None
                and item_geometry.intersects(query_geometry)
            )
        return items if limit is None else islice(items, limit)

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
        crs: MaybeCRS = "EPSG:2193",
        resolution: SomeResolution | None = 100,
        progress: Any = None,
    ) -> xr.Dataset:
        """Filter static-catalog items and load them with ``odc.stac.load``.

        Spatial selectors reduce the source items locally and are also passed
        to ODC to constrain the output grid.
        """
        items = list(
            self.search(
                limit=limit,
                bbox=bbox,
                datetime=datetime,
                intersects=intersects,
                ids=ids,
                collections=collections,
            )
        )
        if not items:
            msg = "No items match the selected collections and filters."
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
