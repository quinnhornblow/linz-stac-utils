"""Adapted from dea_tools."""

import numpy as np
import xarray as xr


def _select_along_axis(values: np.ndarray, idx: np.ndarray, axis: int) -> np.ndarray:
    """Select values from an array using indices along a specific axis."""
    other_ind = np.ix_(*[np.arange(s) for s in idx.shape])
    sl = other_ind[:axis] + (idx,) + other_ind[axis:]
    return values[sl]


def last(
    array: xr.DataArray, dim: str, index_name: str | None = None, drop: bool = True
) -> xr.DataArray:
    """Find the last occurring non-null value along a dimension.

    Parameters
    ----------
    array : xr.DataArray
         The array to search.
    dim : str
        The name of the dimension to reduce by finding the last non-null
        value.
    index_name : str, optional
        If given, the name of a coordinate to be added containing the
        index of where on the dimension the nearest value was found.
    drop: bool, optional
        Whether to drop the original dimension after reduction.

    Returns:
    -------
    reduced : xr.DataArray
        An array of the last non-null values.
        The `dim` dimension will be removed, and replaced with a coord
        of the same name, containing the value of that dimension where
        the last value was found.
    """
    if dim not in array.dims:
        msg = f"Dimension {dim!r} not found in DataArray dims {array.dims!r}."
        raise ValueError(msg)

    axis = array.get_axis_num(dim)
    is_valid = ~array.isnull()
    has_valid = is_valid.any(dim=dim)
    rev = (slice(None),) * axis + (slice(None, None, -1),)
    idx_last = -1 - np.argmax(is_valid[rev].data, axis=axis)
    reduced = array.reduce(_select_along_axis, idx=idx_last, axis=axis)

    idx_da = xr.DataArray(idx_last, dims=reduced.dims)
    reduced = reduced.where(has_valid)
    reduced[dim] = array[dim].isel({dim: idx_da}).where(has_valid)

    if index_name is not None:
        reduced[index_name] = idx_da.where(has_valid)
    if drop:
        reduced = reduced.drop_vars(dim)
    return reduced
