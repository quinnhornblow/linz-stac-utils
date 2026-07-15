# linz-s3-utils

`linz-s3-utils` is a small Python library for querying and loading public LINZ elevation datasets from the `nz-elevation` STAC catalog hosted in S3.

It exists to make LINZ elevation access simpler in Python scripts and notebooks. Instead of repeatedly wiring together `pystac-client`, cached catalog requests, and `odc.stac` loading logic, this package provides a thin reusable wrapper around that workflow.

## What It Does

- opens the public LINZ elevation STAC catalog
- fetches collection and item metadata
- loads STAC results into `xarray` objects with `odc.stac`
- loads the latest New Zealand LiDAR 1 m DEM surface for an ODC-style spatial query
- optionally exports the loaded surface as a Cloud Optimized GeoTIFF

## Current Scope

- Python API only
- focused on the public `nz-elevation` catalog
- aimed at data access, loading, and optional DEM export rather than a general CLI workflow

## Installation

Requirements:

- Python 3.13+
- `uv`

Install the project and development dependencies:

```bash
uv sync
```

## Usage

Load the latest New Zealand LiDAR 1 m DEM surface for a bounding box, at a chosen output resolution:

```python
from linz_s3_utils.elevation import load_elevation

lidar = load_elevation(
    bbox=(172.6300, -43.5350, 172.6400, -43.5250),
    resolution=10,
    output_path="christchurch-dem.tif",
)
```

`load_elevation()` follows the spatial portion of `odc.stac.load`:

- Provide exactly one of `bbox` or `intersects`; calls with neither or both are rejected.
- `bbox` is `(min_longitude, min_latitude, max_longitude, max_latitude)` in `EPSG:4326`.
- `intersects` accepts an ODC geometry, Shapely geometry, GeoJSON mapping, or an object with `__geo_interface__`; Shapely and GeoJSON inputs are interpreted as `EPSG:4326`.
- `crs` defaults to `EPSG:2193`; `resolution` is in the output CRS units and defaults to ODC's source-grid resolution.
- `intersects` crops and masks the output polygon with all touched pixels retained.
- Set `overwrite=True` to replace an existing output file.

Use a polygon when the rectangular `bbox` is not precise enough:

```python
from shapely.geometry import Polygon

lidar = load_elevation(
    intersects=Polygon(
        [
            (172.6300, -43.5350),
            (172.6400, -43.5350),
            (172.6350, -43.5250),
        ]
    ),
    resolution=10,
)
```

For lower-level catalog access, use `StacCatalogClient` directly:

```python
from linz_s3_utils.stac import StacCatalogClient

client = StacCatalogClient()
dataset = client.load(
    collections=["01JE4ZZWAG19KPKRHYJJP02HC9"],
    bbox=(172.6300, -43.5350, 172.6400, -43.5250),
    resolution=1000,
)
```

`load()` filters static-catalog items locally before loading them. It supports
`bbox`, `intersects`, item IDs, and a positive result limit. It defaults to
`EPSG:2193`, and resolutions are specified in metres.

See `src/examples/elevation.ipynb` for an interactive example.

## Notes

- network access is required to read remote catalog and raster data
- STAC API responses are cached locally with `requests-cache` for one day by default. The cache is created when a client is initialized in your platform's user cache directory, rather than in the installed package directory.

Configure caching by creating and injecting a STAC IO instance:

```python
from pathlib import Path

from linz_s3_utils.stac import StacCatalogClient, build_stac_io

client = StacCatalogClient(
    stac_io=build_stac_io(
        cache_path=Path("data/stac.sqlite"),
        expire_after=3600,
    )
)
```

Pass `cache=False` to `build_stac_io()` to use an uncached STAC session.

## Development

Run tests:

```bash
uv run pytest
```

Run linting:

```bash
uv run ruff check .
```
