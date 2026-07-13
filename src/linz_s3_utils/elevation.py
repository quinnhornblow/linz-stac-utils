from typing import Any

import xarray as xr
from tqdm import tqdm

from linz_s3_utils.stac import StacCatalogClient
from linz_s3_utils.utils import last

LIDAR_1M_DEM_COLLECTION_ID = "01JE4ZZWAG19KPKRHYJJP02HC9"


def latest_elevation_surface(dataset: xr.Dataset) -> xr.DataArray:
    """Reduce a loaded elevation dataset to the latest surface."""
    return last(dataset["elevation"], dim="time")


class ElevationClient(StacCatalogClient):
    """Client for accessing elevation data from a STAC catalog."""

    def __init__(self, client: Any | None = None):  # noqa: D107
        super().__init__(catalog="elevation", client=client)

    def load_lidar_dem(self, resolution: int) -> xr.DataArray:
        """Load a dataset from the New Zealand LiDAR 1m DEM collection."""
        ds = self.load(
            collections=[LIDAR_1M_DEM_COLLECTION_ID],
            resampling="bilinear",
            resolution=resolution,
            progress=tqdm,
        )

        return latest_elevation_surface(ds)
