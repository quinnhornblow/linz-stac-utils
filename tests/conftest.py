import pytest
from pystac_client.stac_api_io import StacApiIO

import linz_stac_utils.stac as stac_module


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run tests marked as integration",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return

    skip_integration = pytest.mark.skip(
        reason="integration test; use --run-integration to include it"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture(autouse=True)
def disable_default_stac_cache_for_unit_tests(monkeypatch, request):
    """Keep unit tests from creating a cache in the developer's home directory."""
    if "integration" not in request.node.keywords:
        monkeypatch.setattr(stac_module, "build_stac_io", lambda: StacApiIO())
