# Architecture

## Purpose And Scope

`linz-stac-utils` is a small Python library for accessing public LINZ elevation
data in scripts and notebooks. Its supported workflow is to select a bounded
region, load an elevation surface, optionally mask it to a polygon, and
optionally write a Cloud Optimized GeoTIFF.

The package is deliberately elevation-focused. It is not a general STAC client,
hosted service, CLI, or GDAL/VRT processing system.

## Boundaries

```text
load_elevation / ElevationClient
    -> StacCatalogClient
        -> LINZ static STAC catalog and odc.stac
    -> xarray transformations
        -> optional COG output
```

- `elevation.py` owns the user-facing elevation workflow, source priority, and
  product defaults.
- `stac.py` owns catalog access, request caching, local item filtering, and
  delegation to `odc.stac.load`.
- `utils.py` owns pure xarray array transformations.

Dependencies point from the elevation facade to the STAC adapter and utility
module. The lower-level modules do not depend on the elevation facade.

## Decisions

- Keep the public API small and use direct function or method calls.
- Keep external effects at the catalog, raster-loading, and output boundaries.
- Keep raster transformations compatible with NumPy and Dask-backed xarray
  arrays.
- Define the LiDAR surface as the last non-null value per pixel after sorting
  by the `time` coordinate.
- Define the composite surface by loading contour observations before LiDAR
  observations, then taking the last non-null value per pixel. This ensures
  LiDAR fills take precedence regardless of product timestamps.
- Require an explicit composite output resolution because the catalog items do
  not provide enough projection metadata for reliable automatic grid inference.
- Use ordinary constructor injection for catalog clients and STAC IO. Do not
  introduce a dependency-injection container.
- Add a small elevation-product descriptor only when collection discovery
  supports multiple products with different provider metadata or asset rules.

## Rejected Patterns

The current scope does not justify repositories, units of work, domain events,
CQRS, a plugin system, a general LINZ catalog framework, or separate services.
Those abstractions would add indirection without addressing a current change or
operational boundary.

## Roadmap And Triggers

- [#9](https://github.com/quinnhornblow/linz-stac-utils/issues/9): make the
  package release-ready with accurate metadata, declared dependencies, and
  build verification.
- [#11](https://github.com/quinnhornblow/linz-stac-utils/issues/11): expose
  elevation collection discovery and stable friendly names. Keep raw provider
  IDs supported.
- [#12](https://github.com/quinnhornblow/linz-stac-utils/issues/12): provide a
  runnable regional loading example and document network, CRS, resolution, and
  temporal behavior.
- [#16](https://github.com/quinnhornblow/linz-stac-utils/issues/16): centralize
  spatial-selector validation and geometry normalization before external work.

Measure static-catalog traversal before adding search infrastructure. A larger
catalog or repeated slow regional searches would justify a focused performance
boundary. Supporting non-elevation catalogs or products with materially
different asset/time semantics would justify expanding the product descriptor,
not a general framework by default.
