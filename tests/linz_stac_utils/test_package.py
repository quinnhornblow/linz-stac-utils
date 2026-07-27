import linz_stac_utils
from linz_stac_utils.elevation import ElevationClient, load_elevation
from linz_stac_utils.stac import StacCatalogClient, build_stac_io


def test_package_exports_supported_public_api():
    assert linz_stac_utils.__all__ == [
        "ElevationClient",
        "StacCatalogClient",
        "build_stac_io",
        "load_elevation",
    ]
    assert linz_stac_utils.ElevationClient is ElevationClient
    assert linz_stac_utils.StacCatalogClient is StacCatalogClient
    assert linz_stac_utils.build_stac_io is build_stac_io
    assert linz_stac_utils.load_elevation is load_elevation
