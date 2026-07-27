from pathlib import Path
from typing import Any, Literal

import xarray as xr
from odc.geo import MaybeCRS, SomeResolution
from odc.geo.cog import write_cog
from odc.geo.geom import Geometry
from odc.geo.xr import crop
from shapely.geometry.base import BaseGeometry

from linz_s3_utils.stac import StacCatalogClient
from linz_s3_utils.utils import last

LIDAR_1M_DEM_COLLECTION_ID = "01JE4ZZWAG19KPKRHYJJP02HC9"
CONTOUR_8M_DEM_COLLECTION_ID = "01JE7NNKVY3QP5FPX2Q08DJQX5"


def latest_elevation_surface(dataset: xr.Dataset) -> xr.DataArray:
    """Reduce a loaded elevation dataset to the latest surface."""
    return last(dataset["elevation"].sortby("time"), dim="time")


def _elevation_groupby(item: Any, parsed: Any, _index: int) -> tuple[int, Any]:
    """Order contour data before LiDAR while retaining source chronology."""
    priority = 1 if item.collection_id == LIDAR_1M_DEM_COLLECTION_ID else 0
    return priority, parsed.nominal_datetime


class ElevationClient(StacCatalogClient):
    """Client for accessing elevation data from a STAC catalog."""

    def __init__(self, client: Any | None = None):  # noqa: D107
        super().__init__(catalog="elevation", client=client)

    def load_lidar_dem(
        self,
        resolution: SomeResolution | None = None,
        *,
        resampling: str | dict[str, str] | None = "bilinear",
        chunks: dict[str, int | Literal["auto"]] | None = None,
        crs: MaybeCRS = "EPSG:2193",
        bbox: tuple[float, float, float, float] | None = None,
        intersects: Any = None,
        progress: Any = None,
    ) -> xr.DataArray:
        """Load a dataset from the New Zealand LiDAR 1m DEM collection."""
        ds = self.load(
            collections=[LIDAR_1M_DEM_COLLECTION_ID],
            resampling=resampling,
            chunks=chunks,
            crs=crs,
            resolution=resolution,
            bbox=bbox,
            intersects=intersects,
            progress=progress,
        )

        return latest_elevation_surface(ds)

    def load_dem(
        self,
        resolution: SomeResolution | None = None,
        *,
        resampling: str | dict[str, str] | None = "bilinear",
        chunks: dict[str, int | Literal["auto"]] | None = None,
        crs: MaybeCRS = "EPSG:2193",
        bbox: tuple[float, float, float, float] | None = None,
        intersects: Any = None,
        progress: Any = None,
    ) -> xr.DataArray:
        """Load LiDAR elevation, filling its missing pixels with the 8m contour DEM."""
        if resolution is None:
            msg = "Provide resolution."
            raise ValueError(msg)

        ds = self.load(
            collections=[LIDAR_1M_DEM_COLLECTION_ID, CONTOUR_8M_DEM_COLLECTION_ID],
            groupby=_elevation_groupby,
            resampling=resampling,
            chunks=chunks,
            crs=crs,
            resolution=resolution,
            bbox=bbox,
            intersects=intersects,
            progress=progress,
        )

        return last(ds["elevation"], dim="time")


def _normalize_intersects(intersects: Any) -> Geometry:
    """Normalize supported ODC geometry inputs to a CRS-aware geometry."""
    if isinstance(intersects, Geometry):
        return intersects
    if isinstance(intersects, BaseGeometry):
        return Geometry(intersects, "EPSG:4326")
    if isinstance(intersects, dict):
        return Geometry(intersects, "EPSG:4326")

    geometry = getattr(intersects, "__geo_interface__", None)
    if geometry is None:
        msg = "Can't interpret intersects as geometry."
        raise ValueError(msg)
    return Geometry(geometry, getattr(intersects, "crs", "EPSG:4326"))


def load_elevation(
    *,
    resampling: str | dict[str, str] | None = "bilinear",
    chunks: dict[str, int | Literal["auto"]] | None = None,
    crs: MaybeCRS = "EPSG:2193",
    resolution: SomeResolution | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    intersects: Any = None,
    progress: Any = None,
    output_path: str | Path | None = None,
    overwrite: bool = False,
) -> xr.DataArray:
    """Load LiDAR elevation with contour fallback, optionally masking and exporting it.

    Args:
        resampling: Resampling method passed to ``odc.stac.load``.
        chunks: Chunk sizes passed to ``odc.stac.load``.
        crs: Output coordinate reference system.
        resolution: Required output resolution in the output CRS units.
        bbox: Longitude/latitude bounds passed to ``odc.stac.load``.
        intersects: Geometry limiting the output extent and mask.
        progress: Progress callback passed to ``odc.stac.load``.
        output_path: Optional Cloud Optimized GeoTIFF output path.
        overwrite: Whether to replace an existing output file.

    Returns:
        The loaded elevation surface.
    """
    if bbox is None and intersects is None:
        msg = "Provide bbox or intersects."
        raise ValueError(msg)
    if resolution is None:
        msg = "Provide resolution."
        raise ValueError(msg)
    geometry = None if intersects is None else _normalize_intersects(intersects)
    elevation = ElevationClient().load_dem(
        resampling=resampling,
        chunks=chunks,
        crs=crs,
        resolution=resolution,
        bbox=bbox,
        intersects=geometry,
        progress=progress,
    )
    if geometry is not None:
        elevation = crop(elevation, geometry, apply_mask=True, all_touched=True)
    if output_path is not None:
        cog_write = write_cog(elevation, output_path, overwrite=overwrite)
        if callable(compute := getattr(cog_write, "compute", None)):
            compute()
    return elevation
