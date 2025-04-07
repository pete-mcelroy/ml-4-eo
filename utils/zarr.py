from copy import deepcopy
import logging as log
from typing import Union

import numpy as np
import xarray as xr
import zarr
from affine import Affine
from rasterio.crs import CRS
from stackstac.raster_spec import RasterSpec
from upath import UPath


def save_to_zarr(ds: Union[xr.DataArray, xr.Dataset], filename: UPath):
    path = UPath(filename)
    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    ds = serialize_riox_attrs(ds)
    ds.to_zarr(path)
    zarr.consolidate_metadata(path)


def load_from_zarr(filename: UPath):
    path = UPath(filename)
    ds = xr.open_zarr(path)
    ds = deserialize_riox_attrs(ds)

    return ds


def serialize_riox_attrs(ds: Union[xr.DataArray, xr.Dataset]):
    """
    rioxarray introduces object types that are not serializable in the NetCDF protocol.
    """
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset(name="msi")

    if "band_data" in ds.variables.keys():
        attr_key = "band_data"
    else:
        attr_key = None
        for key in ds.variables.keys():
            if "stackstac-" in str(key) or "msi" in str(key) or "xarray" in str(key):
                attr_key = str(key)
                break
        if attr_key is None:
            raise ValueError('Expected variable of form "stackstac-*" in DataArray')

    try:
        ds.variables.mapping[attr_key].attrs["spec"] = str(
            ds.variables.mapping[attr_key].attrs["spec"]
        )
    except KeyError:
        log.warning(f"Could not find spec in {attr_key} attribute")
    try:
        if ds.variables.mapping.get("spatial_ref"):
            ds.variables.mapping[attr_key].attrs["spatial_ref"] = str(
                ds.variables.mapping["spatial_ref"]
            )
        else:
            log.warning(f"Could not find spatial_ref in dataset variables.")
    except KeyError:
        log.warning(f"Could not find spec in {attr_key} attribute")
    try:
        ds.variables.mapping[attr_key].attrs["transform"] = ds.variables.mapping[attr_key].attrs["transform"]
    except KeyError:
        try:
            ds.variables.mapping[attr_key].attrs["transform"] = list(ds.rio.transform())      
        except KeyError:
            log.warning(f"Could not find spec in {attr_key} attribute")
    try:
        ds.variables.mapping[attr_key].attrs["crs"] = str(
            ds.variables.mapping[attr_key].attrs["crs"]
        )
    except KeyError:
        try:
            ds.variables.mapping[attr_key].attrs["crs"] = str(
                str(ds.rio.crs)
            )       
        except KeyError:
            log.warning(f"Could not find spec in {attr_key} attribute")

    if attr_key != "band_data":
        ds = ds.rename({attr_key: "band_data"})

    keep_vars = ["time", "id", "band", "x", "y", "band_data"]
    ds = ds.drop_vars([v for v in ds.variables.keys() if v not in keep_vars])

    for k, v in ds.variables.mapping.items():
        if v.dtype == object:
            v = ds.variables.mapping[k].values.tolist()
            if isinstance(v, set):
                v = np.array(list(v))

            for ii, value in enumerate(v):
                if value is None:
                    v[ii] = np.nan

            nv = np.array(v)
            ds.variables.mapping[k] = xr.Variable(nv.shape, nv)  # type: ignore

    try:
        ds.band_data.attrs.pop("grid_mapping")
    except KeyError:
        pass

    return ds


def deserialize_riox_attrs(ds: xr.DataArray):
    """
    De-serialize rioxarray attribute information for a dataset read
    from a zarr file.
    """
    old_attrs = ds.band_data.attrs
    attrs = deepcopy(old_attrs)

    def parse_spec(spec: str):
        numbers = list()
        current_number = ""

        for char in spec:
            if char == ".":
                current_number += char
            else:
                try:
                    float(char)
                    current_number += char
                except ValueError:
                    try:
                        value = float(current_number)
                        numbers.append(value)
                        current_number = ""
                    except ValueError:
                        pass

        epsg = numbers[0]
        bounds = (numbers[1], numbers[2], numbers[3], numbers[4])
        resolutions_xy = (numbers[5], numbers[6])

        return epsg, bounds, resolutions_xy

    if old_attrs.get("spec") is not None:
        epsg, bounds, resolutions_xy = parse_spec(old_attrs["spec"])
        attrs["spec"] = RasterSpec(epsg, bounds, resolutions_xy)
        epsg = CRS.from_epsg(int(epsg))
        attrs["crs"] = epsg

    if old_attrs.get("transform") is not None:
        attrs["transform"] = Affine(*old_attrs["transform"])

    ds.variables["band_data"].attrs = attrs

    return ds.band_data