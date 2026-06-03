import xarray as xr
from tqdm import tqdm

from linz_s3_utils.stac import StacCatalogClient
from linz_s3_utils.utils import last


class ElevationClient(StacCatalogClient):
    """Client for accessing elevation data from a STAC catalog."""

    def __init__(self):  # noqa: D107
        super().__init__(catalog="elevation")

    def load_lidar_dem(self, resolution: int) -> xr.DataArray:
        """Load a dataset from the New Zealand LiDAR 1m DEM collection."""
        collection_id = (
            "01JE4ZZWAG19KPKRHYJJP02HC9"  # Collection ID for New Zealand LiDAR 1m DEM
        )
        ds = self.load(
            collections=[collection_id],
            resampling="bilinear",
            resolution=resolution,
            progress=tqdm,
        )

        return last(ds["elevation"], dim="time")
