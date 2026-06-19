# linz-s3-utils

`linz-s3-utils` is a small Python library for querying and loading public LINZ elevation datasets from the `nz-elevation` STAC catalog hosted in S3.

It exists to make LINZ elevation access simpler in Python scripts and notebooks. Instead of repeatedly wiring together `pystac-client`, cached catalog requests, and `odc.stac` loading logic, this package provides a thin reusable wrapper around that workflow.

## What It Does

- opens the public LINZ elevation STAC catalog
- fetches collection and item metadata
- loads STAC results into `xarray` objects with `odc.stac`
- provides an `ElevationClient` helper for the New Zealand LiDAR 1 m DEM collection

## Current Scope

- Python API only
- focused on the public `nz-elevation` catalog
- aimed at data access and loading rather than DEM export or a general CLI workflow

## Installation

Requirements:

- Python 3.13+
- `uv`

Install the project and development dependencies:

```bash
uv sync
```

## Usage

Load the New Zealand LiDAR 1 m DEM collection at a chosen output resolution:

```python
from linz_s3_utils.elevation import ElevationClient

client = ElevationClient()
lidar = client.load_lidar_dem(resolution=1000)
```

For lower-level catalog access, use `StacCatalogClient` directly:

```python
from linz_s3_utils.stac import StacCatalogClient

client = StacCatalogClient()
dataset = client.load(
    collections=["01JE4ZZWAG19KPKRHYJJP02HC9"],
    resolution=1000,
)
```

`load()` defaults to `EPSG:2193`, and resolutions are specified in metres.

See `src/examples/elevation.ipynb` for an interactive example.

## Notes

- network access is required to read remote catalog and raster data
- STAC API responses are cached locally with `requests-cache`

## Development

Run tests:

```bash
uv run pytest
```

Run linting:

```bash
uv run ruff check .
```
