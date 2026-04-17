from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import astropy.units as u
import h5py
import numpy as np
import yaml
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from sunpy.coordinates import Helioprojective
from sunpy.map import make_fitswcs_header

from pyampp.gxbox.boxutils import read_b3d_h5
from pyampp.gxbox.observer_restore import resolve_observer_with_info
from pyampp.util.radio import GXRadioImageComputing


@dataclass(frozen=True)
class RadioConfig:
    tbase: float
    nbase: float
    q0: float
    a: float
    b: float
    force_isothermal: int


@dataclass(frozen=True)
class RadioGeometry:
    box_nx: int
    box_ny: int
    box_xc: float
    box_yc: float
    box_dx: float
    box_dy: float


def _decode_text(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "ignore")
    if isinstance(value, np.ndarray) and value.shape == ():
        item = value.item()
        if isinstance(item, (bytes, bytearray)):
            return item.decode("utf-8", "ignore")
        return str(item)
    return str(value)


def _header_from_index_text(index_text: Any) -> fits.Header:
    text = _decode_text(index_text)
    if not text:
        raise ValueError("Model is missing base/index header metadata")
    return fits.Header.fromstring(text, sep="\n")


def _zyx_to_xyz(arr: Any) -> np.ndarray:
    return np.asarray(arr).transpose((2, 1, 0))


def _line_cube_to_xyz(arr: Any, *, reference_xyz_shape: tuple[int, int, int]) -> np.ndarray:
    data = np.asarray(arr)
    if data.ndim == 1:
        expected = int(np.prod(reference_xyz_shape))
        if data.size != expected:
            raise ValueError(
                f"Flattened line payload has {data.size} values; expected {expected} "
                f"for cube shape {reference_xyz_shape}"
            )
        return data.reshape(reference_xyz_shape, order="F")
    if data.ndim == 3:
        return _zyx_to_xyz(data)
    raise ValueError(f"Unsupported line payload rank {data.ndim}; expected 1D or 3D")


def _split_vector_cube(group: dict[str, Any]) -> np.ndarray:
    bx = _zyx_to_xyz(group["bx"])
    by = _zyx_to_xyz(group["by"])
    bz = _zyx_to_xyz(group["bz"])
    return np.stack((bx, by, bz), axis=3).astype(np.float32, copy=False)


def load_radio_config(path: Path | str) -> RadioConfig:
    config_path = Path(path)
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    required = ("tbase", "nbase", "q0", "a", "b", "force_isothermal")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Radio config is missing required keys: {', '.join(missing)}")
    return RadioConfig(
        tbase=float(payload["tbase"]),
        nbase=float(payload["nbase"]),
        q0=float(payload["q0"]),
        a=float(payload["a"]),
        b=float(payload["b"]),
        force_isothermal=int(payload["force_isothermal"]),
    )


def build_radio_model_dict(model_path: Path | str) -> tuple[dict[str, Any], fits.Header, dict[str, Any]]:
    box = read_b3d_h5(str(model_path))
    base = box.get("base", {})
    corona = box.get("corona", {})
    chromo = box.get("chromo", {})
    lines = box.get("lines", {})
    if not base or not corona or not chromo:
        raise ValueError(f"Model is missing required base/corona/chromo groups: {model_path}")
    required_line_keys = ("av_field", "phys_length", "voxel_status")
    missing_line_keys = [key for key in required_line_keys if key not in lines]
    if missing_line_keys:
        raise ValueError(f"Model is missing required line fields: {', '.join(missing_line_keys)}")

    header = _header_from_index_text(base.get("index"))
    dr = np.asarray(corona.get("dr"), dtype=np.float64).reshape(-1)
    if dr.size < 3:
        raise ValueError("Model corona/dr must have at least 3 values")
    full_bcube = _split_vector_cube(corona)
    chromo_dz = _zyx_to_xyz(chromo["dz"]).astype(np.float32, copy=False)
    reference_xyz_shape = tuple(int(v) for v in full_bcube.shape[:3])
    chromo_layers = int(np.asarray(chromo.get("chromo_layers")).reshape(-1)[0])
    corona_base = int(corona.get("corona_base", 0))
    corona_uniform_layers = int(chromo_dz.shape[2] - chromo_layers)
    corona_limit = min(full_bcube.shape[2], corona_base + max(corona_uniform_layers, 0))
    bcube = full_bcube[:, :, :corona_limit, :]
    avfield = _line_cube_to_xyz(lines["av_field"], reference_xyz_shape=reference_xyz_shape).astype(np.float32, copy=False)
    physlength = _line_cube_to_xyz(lines["phys_length"], reference_xyz_shape=reference_xyz_shape).astype(np.float32, copy=False)
    status = _line_cube_to_xyz(lines["voxel_status"], reference_xyz_shape=reference_xyz_shape).astype(np.int32, copy=False)

    model_dict = {
        "dr": dr[:3],
        "dz": chromo_dz,
        "bcube": bcube,
        "chromo_bcube": _split_vector_cube(chromo),
        "chromo_idx": np.asarray(chromo["chromo_idx"], dtype=np.int64).reshape(-1),
        "chromo_n": np.asarray(chromo["chromo_n"], dtype=np.float32).reshape(-1),
        "n_p": np.asarray(chromo["n_p"], dtype=np.float32).reshape(-1),
        "n_hi": np.asarray(chromo["n_hi"], dtype=np.float32).reshape(-1),
        "chromo_t": np.asarray(chromo["chromo_t"], dtype=np.float32).reshape(-1),
        "avfield": avfield[:, :, :corona_limit],
        "physlength": physlength[:, :, :corona_limit],
        "status": status[:, :, :corona_limit],
        "corona_base": corona_base,
        "chromo_layers": chromo_layers,
    }
    context = {
        "box": box,
        "header": header,
        "nx": int(model_dict["bcube"].shape[0]),
        "ny": int(model_dict["bcube"].shape[1]),
        "nz": int(model_dict["bcube"].shape[2]),
    }
    return model_dict, header, context


def derive_radio_geometry(box: dict[str, Any], *, nx: int, ny: int) -> RadioGeometry:
    observer = box.get("observer", {}) if isinstance(box, dict) else {}
    if not isinstance(observer, dict):
        observer = {}
    fov = observer.get("fov")
    if not isinstance(fov, dict):
        fov = observer.get("fov_box")
    if not isinstance(fov, dict):
        raise ValueError("Model observer metadata is missing fov/fov_box")
    return RadioGeometry(
        box_nx=int(nx),
        box_ny=int(ny),
        box_xc=float(fov["xc_arcsec"]),
        box_yc=float(fov["yc_arcsec"]),
        box_dx=float(fov["xsize_arcsec"]) / float(nx),
        box_dy=float(fov["ysize_arcsec"]) / float(ny),
    )


def compute_radio_emission(
    model_path: Path | str,
    *,
    mw_lib: Path | str,
    ebtel_file: Path | str,
    freqs_ghz: list[float] | np.ndarray,
    radio_config: RadioConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    model_dict, header, context = build_radio_model_dict(model_path)
    geometry = derive_radio_geometry(context["box"], nx=context["nx"], ny=context["ny"])
    gx_radio = GXRadioImageComputing(str(mw_lib))
    ebtel, ebtel_dt = gx_radio.load_ebtel(str(ebtel_file))
    model, model_dt = gx_radio.load_model_dict(model_dict, header)
    emission = gx_radio.synth_model(
        model,
        model_dt,
        ebtel,
        ebtel_dt,
        np.asarray(freqs_ghz, dtype=np.float64),
        geometry.box_nx,
        geometry.box_ny,
        geometry.box_xc,
        geometry.box_yc,
        geometry.box_dx,
        geometry.box_dy,
        radio_config.tbase,
        radio_config.nbase,
        radio_config.q0,
        radio_config.a,
        radio_config.b,
        force_isothermal=radio_config.force_isothermal,
    )
    metadata = {
        "geometry": geometry,
        "header": header,
        "box": context["box"],
    }
    return emission, metadata


def _radio_hpc_header(
    box: dict[str, Any],
    geometry: RadioGeometry,
    *,
    date_obs: str,
    nx: int,
    ny: int,
) -> fits.Header:
    obs_time = Time(date_obs)
    observer = box.get("observer", {}) if isinstance(box, dict) else {}
    observer_key = None
    if isinstance(observer, dict):
        fov_box = observer.get("fov_box")
        if isinstance(fov_box, dict):
            observer_key = fov_box.get("observer_key")
        if observer_key is None:
            observer_key = observer.get("name")
    observer_coord, _warning, _used_key = resolve_observer_with_info(box, observer_key, obs_time)
    if observer_coord is None:
        raise ValueError("Could not resolve observer metadata for radio FITS output")
    center = SkyCoord(
        geometry.box_xc * u.arcsec,
        geometry.box_yc * u.arcsec,
        frame=Helioprojective(observer=observer_coord, obstime=obs_time),
    )
    header = make_fitswcs_header(
        (int(ny), int(nx)),
        center,
        scale=u.Quantity([geometry.box_dx, geometry.box_dy], u.arcsec / u.pix),
        projection_code="TAN",
    )
    header["DATE-OBS"] = obs_time.isot
    header["DATE_OBS"] = obs_time.isot
    return header


def write_radio_bundle(
    output_path: Path | str,
    emission: dict[str, np.ndarray],
    *,
    freqs_ghz: list[float] | np.ndarray,
    metadata: Mapping[str, Any],
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ti = np.asarray(emission["TI"], dtype=np.float64).transpose((2, 1, 0))
    tv = np.asarray(emission["TV"], dtype=np.float64).transpose((2, 1, 0))
    with h5py.File(output, "w") as handle:
        handle.create_dataset("TI", data=ti)
        handle.create_dataset("TV", data=tv)
        handle.create_dataset("freq_ghz", data=np.asarray(freqs_ghz, dtype=np.float64))
        meta_group = handle.create_group("metadata")
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, bytes, bytearray, int, float, np.integer, np.floating)):
                meta_group.attrs[str(key)] = value
            else:
                meta_group.attrs[str(key)] = str(value)
        handle["TI"].attrs["axis_order"] = "f,y,x"
        handle["TV"].attrs["axis_order"] = "f,y,x"
    return output


def write_radio_fits(
    output_dir: Path | str,
    emission: dict[str, np.ndarray],
    *,
    freqs_ghz: list[float] | np.ndarray,
    box: dict[str, Any],
    header: fits.Header,
    geometry: RadioGeometry,
    file_stem: str,
    extra_header: dict[str, Any] | None = None,
) -> list[Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    freqs = np.asarray(freqs_ghz, dtype=np.float64).reshape(-1)
    wcs_header = _radio_hpc_header(
        box,
        geometry,
        date_obs=str(header.get("DATE-OBS", header.get("DATE_OBS"))),
        nx=geometry.box_nx,
        ny=geometry.box_ny,
    )
    written: list[Path] = []
    for idx, freq in enumerate(freqs):
        for stokes in ("TI", "TV"):
            image_xy = np.asarray(emission[stokes][:, :, idx], dtype=np.float64)
            image_yx = image_xy.T
            out_header = fits.Header(wcs_header)
            out_header["FREQGHZ"] = float(freq)
            out_header["STOKES"] = stokes
            if extra_header:
                for key, value in extra_header.items():
                    if value is not None:
                        out_header[str(key).upper()[:8]] = value
            out_path = target_dir / f"{file_stem}.{stokes}.{freq:.3f}GHz.fits"
            fits.PrimaryHDU(data=image_yx, header=out_header).writeto(out_path, overwrite=True)
            written.append(out_path)
    return written
