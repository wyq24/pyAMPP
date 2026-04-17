from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from sunpy.coordinates import Heliocentric, HeliographicCarrington, Helioprojective

from pyampp.gx_chromo.decompose import decompose
from pyampp.gxbox.boxutils import (
    load_sunpy_map_compat,
    normalize_observer_metadata,
    observer_ephemeris_from_map,
    write_b3d_h5,
)
from pyampp.gxbox.observer_restore import build_pb0r_metadata_from_ephemeris
from pyampp.gxbox.selector_api import DisplayFovBoxSelection, DisplayFovSelection
from pyampp.util.config import IDL_HMI_RSUN_M

from .drms import SHARP_CEA_SERIES


@dataclass(frozen=True)
class SharpEntryBoxBuild:
    output_path: Path
    model_id: str
    base_id: str
    harpnum: int
    t_rec: str
    dx_km: float
    box_dims: tuple[int, int, int]


def _require_segment(segment_paths: Mapping[str, Path | str], key: str) -> Path:
    raw = segment_paths.get(key)
    if raw is None:
        raise ValueError(f"Missing required SHARP segment path: {key}")
    return Path(raw)


def _load_sharp_segment_map(path: Path | str, *, fallback_header: fits.Header | None = None):
    try:
        return load_sunpy_map_compat(path)
    except Exception:
        if fallback_header is None:
            raise
        with fits.open(path) as hdul:
            image_hdu = next((hdu for hdu in hdul if getattr(hdu, "data", None) is not None), None)
            if image_hdu is None:
                raise ValueError(f"No image HDU with data found in FITS file: {path}")
            data = image_hdu.data
            source_header = image_hdu.header.copy()
        merged = fallback_header.copy()
        for key, value in source_header.items():
            if key in ("SIMPLE", "BITPIX", "NAXIS", "EXTEND"):
                continue
            if key.startswith("NAXIS"):
                continue
            if key.startswith("CTYPE") or key.startswith("CUNIT") or key.startswith("CRPIX") or key.startswith("CRVAL") or key.startswith("CDELT") or key in ("PC1_1", "PC1_2", "PC2_1", "PC2_2", "CROTA2", "WCSNAME"):
                continue
            merged[key] = value
        return load_sunpy_map_compat(data, merged)


def infer_native_dx_km(header: fits.Header) -> float:
    if header is None:
        raise ValueError("A FITS header is required to infer SHARP pixel size")
    cdelt1 = header.get("CDELT1")
    if cdelt1 is None:
        raise ValueError("SHARP header is missing CDELT1")
    cunit1 = str(header.get("CUNIT1", "deg")).strip().lower()
    if cunit1 in {"deg", "degree", "degrees"}:
        scale_angle = abs(float(cdelt1)) * u.deg
    elif cunit1 in {"rad", "radian", "radians"}:
        scale_angle = abs(float(cdelt1)) * u.rad
    else:
        raise ValueError(f"Unsupported SHARP CUNIT1 for native scale inference: {cunit1!r}")
    rsun_km = (IDL_HMI_RSUN_M * u.m).to_value(u.km)
    return float(np.sin(scale_angle.to_value(u.rad)) * rsun_km)


def _map_center_hgc(smap) -> SkyCoord:
    observer = getattr(smap, "observer_coordinate", None)
    center = smap.center
    return center.transform_to(HeliographicCarrington(observer=observer, obstime=smap.date))


def _map_hpc_bbox(smap) -> dict[str, float]:
    ny, nx = np.asarray(smap.data).shape
    x_pix = u.Quantity([0, nx - 1, nx - 1, 0], u.pix)
    y_pix = u.Quantity([0, 0, ny - 1, ny - 1], u.pix)
    world = smap.pixel_to_world(x_pix, y_pix)
    observer = getattr(smap, "observer_coordinate", None)
    hpc_frame = Helioprojective(observer=observer, obstime=smap.date, rsun=smap.rsun_meters)
    corners = world.transform_to(hpc_frame)
    tx = np.asarray(corners.Tx.to_value(u.arcsec), dtype=float)
    ty = np.asarray(corners.Ty.to_value(u.arcsec), dtype=float)
    xmin = float(np.nanmin(tx))
    xmax = float(np.nanmax(tx))
    ymin = float(np.nanmin(ty))
    ymax = float(np.nanmax(ty))
    return {
        "xc_arcsec": 0.5 * (xmin + xmax),
        "yc_arcsec": 0.5 * (ymin + ymax),
        "xsize_arcsec": xmax - xmin,
        "ysize_arcsec": ymax - ymin,
    }


def _surface_z_bounds_mm(smap, *, height_mm: float) -> tuple[float, float]:
    ny, nx = np.asarray(smap.data).shape
    x_pix = u.Quantity([0, nx - 1, nx - 1, 0], u.pix)
    y_pix = u.Quantity([0, 0, ny - 1, ny - 1], u.pix)
    world = smap.pixel_to_world(x_pix, y_pix)
    observer = getattr(smap, "observer_coordinate", None)
    hcc = world.transform_to(Heliocentric(observer=observer, obstime=smap.date))
    z_vals = np.asarray(hcc.z.to_value(u.Mm), dtype=float)
    z_min = float(np.nanmin(z_vals))
    z_max = float(np.nanmax(z_vals))
    return z_min, z_max + float(height_mm)


def _stage_base_id(t_rec: Time, harpnum: int) -> str:
    return f"{SHARP_CEA_SERIES}.{t_rec.utc.strftime('%Y%m%d_%H%M%S')}.HARP{int(harpnum)}.CEA"


def build_sharp_entry_box(
    segment_paths: Mapping[str, Path | str],
    output_path: Path | str,
    *,
    harpnum: int,
    t_rec: str,
    requested_time: str | Time,
    nz: int,
    source_product: str = SHARP_CEA_SERIES,
) -> SharpEntryBoxBuild:
    if int(nz) <= 0:
        raise ValueError("--nz must be a positive integer")
    from pyampp.gxbox.gx_fov2box import _build_index_header, _refmap_wcs_header

    br_path = _require_segment(segment_paths, "Br")
    bt_path = _require_segment(segment_paths, "Bt")
    bp_path = _require_segment(segment_paths, "Bp")
    continuum_path = _require_segment(segment_paths, "continuum")
    magnetogram_path = _require_segment(segment_paths, "magnetogram")
    bitmap_path = Path(segment_paths["bitmap"]) if segment_paths.get("bitmap") is not None else None

    br_map = load_sunpy_map_compat(br_path)
    bt_map = load_sunpy_map_compat(bt_path)
    bp_map = load_sunpy_map_compat(bp_path)
    fallback_header = fits.getheader(br_path)
    continuum_map = _load_sharp_segment_map(continuum_path, fallback_header=fallback_header)
    magnetogram_map = _load_sharp_segment_map(magnetogram_path, fallback_header=fallback_header)
    bitmap_map = _load_sharp_segment_map(bitmap_path, fallback_header=fallback_header) if bitmap_path is not None else None

    ny, nx = np.asarray(br_map.data).shape
    source_header = fits.getheader(br_path)
    dx_km = infer_native_dx_km(source_header)
    rsun_km = (IDL_HMI_RSUN_M * u.m).to_value(u.km)
    dr = float(dx_km / rsun_km)
    dr3 = np.array([dr, dr, dr], dtype=np.float64)

    base_bx = np.asarray(bp_map.data, dtype=np.float64)
    base_by = -np.asarray(bt_map.data, dtype=np.float64)
    base_bz = np.asarray(br_map.data, dtype=np.float64)
    base_ic = np.asarray(continuum_map.data, dtype=np.float64)
    base_los = np.asarray(magnetogram_map.data, dtype=np.float64)
    chromo_mask = decompose(base_los.T, base_ic.T).T

    t_rec_time = Time(t_rec)
    center_hgc = _map_center_hgc(br_map)
    fov_meta = _map_hpc_bbox(br_map)
    fov = DisplayFovSelection(
        center_x_arcsec=float(fov_meta["xc_arcsec"]),
        center_y_arcsec=float(fov_meta["yc_arcsec"]),
        width_arcsec=float(fov_meta["xsize_arcsec"]),
        height_arcsec=float(fov_meta["ysize_arcsec"]),
    )
    z_min_mm, z_max_mm = _surface_z_bounds_mm(br_map, height_mm=(int(nz) * dx_km) / 1000.0)
    fov_box = DisplayFovBoxSelection.from_display_fov(
        fov,
        z_min_mm,
        z_max_mm,
        observer_key="sdo",
    )

    index_header = _build_index_header(
        source_header,
        br_map,
        observer_override=getattr(br_map, "observer_coordinate", None),
        obs_time_override=br_map.date,
        rsun_override=br_map.rsun_meters,
    )

    corona_bx = np.zeros((int(nz), ny, nx), dtype=np.float64)
    corona_by = np.zeros((int(nz), ny, nx), dtype=np.float64)
    corona_bz = np.zeros((int(nz), ny, nx), dtype=np.float64)
    corona_bx[0, :, :] = base_bx
    corona_by[0, :, :] = base_by
    corona_bz[0, :, :] = base_bz

    ephemeris, _missing = observer_ephemeris_from_map(br_map)
    observer = {
        "name": "sdo",
        "label": "SDO",
        "fov": {
            "frame": "helioprojective",
            "xc_arcsec": float(fov.center_x_arcsec),
            "yc_arcsec": float(fov.center_y_arcsec),
            "xsize_arcsec": float(fov.width_arcsec),
            "ysize_arcsec": float(fov.height_arcsec),
            "square": False,
        },
        "fov_box": fov_box.as_observer_metadata(square=False),
        "ephemeris": ephemeris,
    }
    pb0r = build_pb0r_metadata_from_ephemeris(
        ephemeris,
        observer_key=observer["name"],
        obs_time=ephemeris.get("obs_date"),
    )
    if pb0r:
        observer["pb0r"] = pb0r

    requested = requested_time if isinstance(requested_time, Time) else Time(requested_time)
    base_id = _stage_base_id(t_rec_time, int(harpnum))
    model_id = f"{base_id}.NONE"
    execute_cmd = shlex.join(
        [
            "gx-fov2box",
            "--time",
            t_rec_time.utc.isot,
            "--coords",
            f"{center_hgc.lon.to_value(u.deg):.12f}",
            f"{center_hgc.lat.to_value(u.deg):.12f}",
            "--hgc",
            "--cea",
            "--box-dims",
            str(int(nx)),
            str(int(ny)),
            str(int(nz)),
            "--dx-km",
            f"{dx_km:.6f}",
            "--observer-name",
            "sdo",
            "--fov-xc",
            f"{fov.center_x_arcsec:.6f}",
            "--fov-yc",
            f"{fov.center_y_arcsec:.6f}",
            "--fov-xsize",
            f"{fov.width_arcsec:.6f}",
            "--fov-ysize",
            f"{fov.height_arcsec:.6f}",
        ]
    )

    box_b3d = {
        "base": {
            "bx": base_bx,
            "by": base_by,
            "bz": base_bz,
            "ic": base_ic,
            "chromo_mask": np.asarray(chromo_mask, dtype=np.int32),
            "index": index_header,
        },
        "corona": {
            "bx": corona_bx,
            "by": corona_by,
            "bz": corona_bz,
            "dr": dr3,
            "attrs": {
                "model_type": "none",
                "source_product": str(source_product),
                "harpnum": int(harpnum),
                "t_rec": str(t_rec_time.utc.isot),
                "obs_time": str(t_rec_time.utc.isot),
            },
        },
        "refmaps": {
            "Bz_reference": {
                "data": np.asarray(magnetogram_map.data),
                "wcs_header": _refmap_wcs_header(magnetogram_map),
            },
            "Ic_reference": {
                "data": np.asarray(continuum_map.data),
                "wcs_header": _refmap_wcs_header(continuum_map),
            },
        },
        "observer": observer,
        "metadata": {
            "id": model_id,
            "execute": execute_cmd,
            "lineage": "ENTRY.NONE.H5->NONE.h5",
            "projection": "CEA",
            "disambiguation": "SHARP",
            "axis_order_2d": "yx",
            "axis_order_3d": "zyx",
            "vector_layout": "split_components",
            "source_product": str(source_product),
            "harpnum": int(harpnum),
            "t_rec": str(t_rec_time.utc.isot),
            "requested_time": str(requested.utc.isot),
            "native_dx_km": float(dx_km),
        },
    }
    if bitmap_map is not None:
        box_b3d["refmaps"]["Bitmap_reference"] = {
            "data": np.asarray(bitmap_map.data),
            "wcs_header": _refmap_wcs_header(bitmap_map),
        }

    box_b3d = normalize_observer_metadata(box_b3d)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_b3d_h5(str(output), box_b3d)
    return SharpEntryBoxBuild(
        output_path=output,
        model_id=model_id,
        base_id=base_id,
        harpnum=int(harpnum),
        t_rec=str(t_rec_time.utc.isot),
        dx_km=float(dx_km),
        box_dims=(int(nx), int(ny), int(nz)),
    )
