"""Adapted from dea_tools."""

import numpy as np
import xarray as xr


def _last_valid(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Select last valid values and their zero-based indices on the final axis."""
    index = np.where(valid, np.arange(values.shape[-1]), -1).max(axis=-1)
    value = np.take_along_axis(values, np.maximum(index, 0)[..., np.newaxis], axis=-1)[
        ..., 0
    ]
    return value, index


def _coordinate_at_index(index: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Select one dimension coordinate for each index."""
    return values[index]


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

    if array.chunks is not None:
        array = array.chunk({dim: -1})
    is_valid = ~array.isnull()
    has_valid = is_valid.any(dim=dim)
    reduced, index = xr.apply_ufunc(
        _last_valid,
        array,
        is_valid,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[], []],
        dask="parallelized",
        output_dtypes=[array.dtype, np.intp],
    )
    reduced = reduced.where(has_valid)

    if not drop:
        coordinate = xr.apply_ufunc(
            _coordinate_at_index,
            index,
            array[dim],
            input_core_dims=[[], [dim]],
            output_core_dims=[[]],
            dask="parallelized",
            output_dtypes=[array[dim].dtype],
        ).where(has_valid)
        reduced = reduced.assign_coords({dim: coordinate})
    if index_name is not None:
        reduced = reduced.assign_coords(
            {index_name: (index - array.sizes[dim]).where(has_valid)}
        )
    return reduced
