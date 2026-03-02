#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path
from typing import Any, Optional

import astropy.units as u
import numpy as np
from astropy.time import Time
from PyQt5.QtWidgets import QApplication, QFileDialog, QDialog, QMessageBox

from pyampp.data.downloader import SDOImageDownloader
from pyampp.gxbox.boxutils import read_b3d_h5, write_b3d_h5
from pyampp.gxbox.fov_selector_gui import FovBoxSelectorDialog
from pyampp.gxbox.gx_fov2box import (
    _decode_id_text,
    _extract_execute_geometry,
    _extract_execute_paths,
    _infer_time_from_entry_loaded,
    _load_entry_box_any,
)
from pyampp.gxbox.selector_api import (
    BoxGeometrySelection,
    CoordMode,
    DisplayFovBoxSelection,
    DisplayFovSelection,
    SelectorDialogResult,
    SelectorSessionInput,
)

_DEFAULT_MAP_IDS = ("Bz", "Ic", "Br", "Bp", "Bt", "Vert_current", "chromo_mask", "94", "131", "1600", "1700", "171", "193", "211", "304", "335")


def _pick_entry_file(initial: Optional[str], start_dir: Optional[str]) -> Optional[str]:
    dialog = QFileDialog(None, "Open Model Box (.h5 or .sav)")
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)
    dialog.setFileMode(QFileDialog.ExistingFile)
    dialog.setNameFilter("Model Box Files (*.h5 *.sav);;HDF5 Files (*.h5);;SAV Files (*.sav);;All Files (*)")
    if initial:
        candidate = Path(initial).expanduser()
        dialog.setDirectory(str(candidate.parent if candidate.parent.exists() else Path.cwd()))
        dialog.selectFile(candidate.name)
    elif start_dir:
        base = Path(start_dir).expanduser()
        dialog.setDirectory(str(base if base.exists() else Path.cwd()))
    else:
        dialog.setDirectory(str(Path.cwd()))
    if not dialog.exec_():
        return None
    selected = dialog.selectedFiles()
    return selected[0] if selected else None


def _parse_execute_box_dims_and_dx(execute_text: str) -> tuple[Optional[tuple[int, int, int]], Optional[float]]:
    dims = None
    dx_km = None
    if not execute_text:
        return dims, dx_km
    try:
        parts = shlex.split(execute_text)
    except Exception:
        parts = []
    for i, token in enumerate(parts):
        if token == "--box-dims" and i + 3 < len(parts):
            try:
                dims = (int(float(parts[i + 1])), int(float(parts[i + 2])), int(float(parts[i + 3])))
            except Exception:
                pass
        elif token.startswith("--box-dims="):
            try:
                vals = token.split("=", 1)[1].split(",")
                if len(vals) == 3:
                    dims = tuple(int(float(v)) for v in vals)
            except Exception:
                pass
        elif token == "--dx-km" and i + 1 < len(parts):
            try:
                dx_km = float(parts[i + 1])
            except Exception:
                pass
        elif token.startswith("--dx-km="):
            try:
                dx_km = float(token.split("=", 1)[1])
            except Exception:
                pass
    if dims is None:
        m = re.search(
            r"--box-dims\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)",
            execute_text,
        )
        if m:
            dims = tuple(int(round(float(m.group(i)))) for i in (1, 2, 3))
    if dx_km is None:
        m = re.search(r"--dx-km\s+([+-]?\d+(?:\.\d+)?)", execute_text)
        if m:
            dx_km = float(m.group(1))
    return dims, dx_km


def _infer_dims_from_entry(entry_loaded: dict[str, Any]) -> tuple[int, int, int]:
    for group_name in ("corona", "chromo"):
        grp = entry_loaded.get(group_name)
        if not isinstance(grp, dict):
            continue
        for key in ("bx", "by", "bz"):
            if key in grp:
                arr = np.asarray(grp[key])
                if arr.ndim >= 3:
                    return tuple(int(v) for v in arr.shape[:3])
        for key in ("bcube", "chromo_bcube"):
            if key in grp:
                arr = np.asarray(grp[key])
                if arr.ndim == 4 and arr.shape[-1] == 3:
                    return tuple(int(v) for v in arr.shape[:3])
                if arr.ndim == 4 and arr.shape[0] == 3:
                    return tuple(int(v) for v in arr.shape[1:4])
    return (150, 75, 150)


def _infer_dx_km_from_entry(entry_loaded: dict[str, Any]) -> float:
    rsun_km = 695700.0
    for group_name in ("corona", "chromo"):
        grp = entry_loaded.get(group_name)
        if not isinstance(grp, dict):
            continue
        dr = grp.get("dr")
        if dr is None:
            continue
        try:
            arr = np.asarray(dr, dtype=float).ravel()
            if arr.size > 0:
                return float(arr[0] * rsun_km)
        except Exception:
            continue
    return 1400.0


def _observer_fov_from_entry(
    entry_loaded: dict[str, Any],
) -> tuple[Optional[DisplayFovSelection], bool, Optional[DisplayFovBoxSelection]]:
    observer = entry_loaded.get("observer")
    if not isinstance(observer, dict):
        return None, False, None
    fov = observer.get("fov")
    if not isinstance(fov, dict):
        return None, False, None
    try:
        result = DisplayFovSelection(
            center_x_arcsec=float(fov["xc_arcsec"]),
            center_y_arcsec=float(fov["yc_arcsec"]),
            width_arcsec=float(fov["xsize_arcsec"]),
            height_arcsec=float(fov["ysize_arcsec"]),
        )
        fov_box_meta = observer.get("fov_box")
        fov_box = None
        if isinstance(fov_box_meta, dict):
            try:
                fov_box = DisplayFovBoxSelection(
                    center_x_arcsec=float(fov_box_meta.get("xc_arcsec", result.center_x_arcsec)),
                    center_y_arcsec=float(fov_box_meta.get("yc_arcsec", result.center_y_arcsec)),
                    width_arcsec=float(fov_box_meta.get("xsize_arcsec", result.width_arcsec)),
                    height_arcsec=float(fov_box_meta.get("ysize_arcsec", result.height_arcsec)),
                    z_min_mm=float(fov_box_meta["zmin_mm"]),
                    z_max_mm=float(fov_box_meta["zmax_mm"]),
                )
            except Exception:
                fov_box = None
        return result, bool(fov.get("square", False)), fov_box
    except Exception:
        return None, False, None


def _geometry_from_entry(entry_loaded: dict[str, Any], entry_path: Path) -> tuple[str, BoxGeometrySelection]:
    meta = entry_loaded.get("metadata", {}) if isinstance(entry_loaded, dict) else {}
    execute_text = _decode_id_text(meta.get("execute", "")) if isinstance(meta, dict) else ""
    time_iso = _infer_time_from_entry_loaded(entry_loaded, entry_path) or Time.now().to_datetime().strftime("%Y-%m-%dT%H:%M:%S")
    coords, frame, _projection = _extract_execute_geometry(execute_text)
    dims, dx_km = _parse_execute_box_dims_and_dx(execute_text)
    if coords is None:
        coords = (0.0, 0.0)
    if dims is None:
        dims = _infer_dims_from_entry(entry_loaded)
    if dx_km is None:
        dx_km = _infer_dx_km_from_entry(entry_loaded)
    mode = {
        "hpc": CoordMode.HPC,
        "hgc": CoordMode.HGC,
        "hgs": CoordMode.HGS,
    }.get((frame or "hpc").lower(), CoordMode.HPC)
    geometry = BoxGeometrySelection(
        coord_mode=mode,
        coord_x=float(coords[0]),
        coord_y=float(coords[1]),
        grid_x=int(dims[0]),
        grid_y=int(dims[1]),
        grid_z=int(dims[2]),
        dx_km=float(dx_km),
    )
    return time_iso, geometry


def _discover_filesystem_maps(time_iso: str, data_dir: Optional[str]) -> dict[str, str]:
    if not data_dir:
        return {}
    base = Path(data_dir).expanduser()
    if not base.exists() or not base.is_dir():
        return {}
    try:
        downloader = SDOImageDownloader(Time(time_iso), data_dir=str(base), euv=True, uv=True, hmi=True)
        return {k: v for k, v in downloader._check_files_exist(downloader.path, returnfilelist=True).items() if v}
    except Exception:
        return {}


def _available_map_ids_from_sources(map_files: dict[str, str], refmaps: dict[str, dict], base_maps: dict[str, Any]) -> list[str]:
    out: list[str] = []
    fs_availability = {
        "Bz": "magnetogram" in map_files,
        "Ic": "continuum" in map_files,
        "Br": all(k in map_files for k in ("field", "inclination", "azimuth", "disambig")),
        "Bp": all(k in map_files for k in ("field", "inclination", "azimuth", "disambig")),
        "Bt": all(k in map_files for k in ("field", "inclination", "azimuth", "disambig")),
    }
    ref_availability = {
        "Bz": "Bz_reference" in refmaps,
        "Ic": "Ic_reference" in refmaps,
        "Vert_current": "Vert_current" in refmaps,
        "94": "AIA_94" in refmaps,
        "131": "AIA_131" in refmaps,
        "1600": "AIA_1600" in refmaps,
        "1700": "AIA_1700" in refmaps,
        "171": "AIA_171" in refmaps,
        "193": "AIA_193" in refmaps,
        "211": "AIA_211" in refmaps,
        "304": "AIA_304" in refmaps,
        "335": "AIA_335" in refmaps,
    }
    base_availability = {
        "Bz": "bz" in base_maps,
        "Ic": "ic" in base_maps,
        "Br": "bz" in base_maps,
        "Bp": "bx" in base_maps,
        "Bt": "by" in base_maps,
        "chromo_mask": "chromo_mask" in base_maps,
    }
    for map_id in _DEFAULT_MAP_IDS:
        if map_id in fs_availability:
            if fs_availability[map_id] or ref_availability.get(map_id, False) or base_availability.get(map_id, False):
                out.append(map_id)
        elif map_id in ref_availability:
            if ref_availability[map_id]:
                out.append(map_id)
        elif map_id in base_availability:
            if base_availability[map_id]:
                out.append(map_id)
        elif map_id in map_files:
            out.append(map_id)
    return out or list(_DEFAULT_MAP_IDS)


def _build_session_input(entry_path: Path) -> SelectorSessionInput:
    entry_loaded = _load_entry_box_any(entry_path)
    time_iso, geometry = _geometry_from_entry(entry_loaded, entry_path)
    meta = entry_loaded.get("metadata", {}) if isinstance(entry_loaded, dict) else {}
    execute_text = _decode_id_text(meta.get("execute", "")) if isinstance(meta, dict) else ""
    data_dir, _gxmodel_dir = _extract_execute_paths(execute_text)
    explicit_fov, square_fov, explicit_fov_box = _observer_fov_from_entry(entry_loaded)

    map_files = _discover_filesystem_maps(time_iso, data_dir)
    refmaps = {}
    raw_refmaps = entry_loaded.get("refmaps")
    if isinstance(raw_refmaps, dict):
        refmaps = raw_refmaps
    base_maps = {}
    raw_base = entry_loaded.get("base")
    if isinstance(raw_base, dict):
        for key in ("bx", "by", "bz", "ic", "chromo_mask"):
            if key in raw_base:
                base_maps[key] = raw_base[key]
    map_ids = _available_map_ids_from_sources(map_files, refmaps, base_maps)
    initial_map = "171" if "171" in map_ids else ("Bz" if "Bz" in map_ids else (map_ids[0] if map_ids else None))
    map_source_mode = "filesystem" if map_files else ("embedded" if refmaps else "auto")

    return SelectorSessionInput(
        time_iso=time_iso,
        data_dir=data_dir or "",
        geometry=geometry,
        fov=explicit_fov,
        fov_box=explicit_fov_box,
        square_fov=square_fov,
        allow_geometry_edit=False,
        map_ids=tuple(map_ids),
        map_files=map_files or None,
        refmaps=refmaps or None,
        base_maps=base_maps or None,
        base_geometry=geometry,
        map_source_mode=map_source_mode,
        initial_map_id=initial_map,
        pad_frac=0.10,
    )


def _persist_selector_result_to_entry(
    entry_path: Path,
    result: SelectorDialogResult,
    line_seeds=None,
    fov_box: DisplayFovBoxSelection | None = None,
) -> bool:
    if entry_path.suffix.lower() != ".h5":
        return False

    box_data = read_b3d_h5(str(entry_path))
    observer = box_data.get("observer")
    if not isinstance(observer, dict):
        observer = {}
    else:
        observer = dict(observer)

    observer_name = observer.get("name", "earth")
    ephemeris = observer.get("ephemeris")
    fov = {
        "frame": "helioprojective",
        "xc_arcsec": float(result.fov.center_x_arcsec),
        "yc_arcsec": float(result.fov.center_y_arcsec),
        "xsize_arcsec": float(result.fov.width_arcsec),
        "ysize_arcsec": float(result.fov.height_arcsec),
        "square": bool(result.square_fov),
    }
    observer["name"] = observer_name
    observer["fov"] = fov
    if isinstance(fov_box, DisplayFovBoxSelection):
        observer["fov_box"] = fov_box.as_observer_metadata(square=bool(result.square_fov))
    else:
        observer.pop("fov_box", None)
    if isinstance(ephemeris, dict):
        observer["ephemeris"] = ephemeris
    box_data["observer"] = observer
    if isinstance(line_seeds, dict):
        box_data["line_seeds"] = line_seeds
    else:
        box_data.pop("line_seeds", None)
    write_b3d_h5(str(entry_path), box_data)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Open the FOV / Box selector from a saved .h5/.sav box file.")
    parser.add_argument("entry_box", nargs="?", help="Path to the saved .h5 or .sav box file.")
    parser.add_argument("--pick", action="store_true", help="Open a file picker even if a path is provided.")
    parser.add_argument("--dir", dest="start_dir", help="Initial directory for the file picker.")
    args = parser.parse_args()

    app = QApplication.instance()
    owns_app = False
    if app is None:
        app = QApplication([])
        owns_app = True

    entry_arg = args.entry_box
    if args.pick or not entry_arg:
        picked = _pick_entry_file(entry_arg, args.start_dir)
        if not picked:
            return 0
        entry_arg = picked

    entry_path = Path(entry_arg).expanduser().resolve()
    session_input = _build_session_input(entry_path)
    dialog = FovBoxSelectorDialog(session_input=session_input, entry_box_path=entry_path)
    dialog.setWindowTitle(f"FOV / Box Selector - {entry_path.name}")

    def _persist_result_if_needed() -> None:
        if dialog.result() != QDialog.Accepted:
            return
        result = dialog.accepted_selection()
        if result is None:
            return
        line_seeds = dialog.committed_line_seeds()
        fov_box = dialog.current_fov_box_selection()
        try:
            persisted = _persist_selector_result_to_entry(entry_path, result, line_seeds=line_seeds, fov_box=fov_box)
            if not persisted and entry_path.suffix.lower() == ".sav":
                QMessageBox.information(
                    dialog,
                    "FOV Not Saved",
                    "This viewer can persist updated FOV metadata only to .h5 boxes. "
                    "The current .sav file was left unchanged.",
                )
        except Exception as exc:
            QMessageBox.warning(
                dialog,
                "Save Failed",
                f"Failed to write updated observer FOV metadata:\n{exc}",
            )

    if owns_app:
        dialog.finished.connect(lambda _code: (_persist_result_if_needed(), app.quit()))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        app.exec_()
    else:
        accepted = dialog.exec_() == QDialog.Accepted
        if accepted:
            _persist_result_if_needed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
