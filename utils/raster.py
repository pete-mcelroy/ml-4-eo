import xarray as xr
import numpy as np
from typing import List, Optional

def concat_band(
    data_array: xr.DataArray,
    band: np.ndarray,
    band_names: List[str],
    dims=["band", "y", "x"],
    band_dim="band",
    positions: Optional[List[List[int]]] = None,
):
    """
    Concatenate a numpy band to a DataArray
    """
    if len(band.shape) == 2:
        band = np.expand_dims(band, axis=0)

    coords = {band_dim: band_names}
    for dim in dims:
        if dim != band_dim:
            coords.update({dim: data_array.coords[dim]})

    bands_da = xr.DataArray(
        band,
        dims=dims,
        coords=coords,
    )
    return xr.concat([data_array, bands_da], dim=band_dim, positions=positions)