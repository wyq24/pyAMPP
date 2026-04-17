from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy.io import fits
from astropy.time import Time

from .drms import (
    SharpRecord,
    download_sharp_record_segments,
    find_nearest_sharp_records,
)
from .entry_box import build_sharp_entry_box
from .radio_adapter import (
    compute_radio_emission,
    load_radio_config,
    write_radio_bundle,
    write_radio_fits,
)

DEFAULT_FREQ_SOURCE_FITS = Path(
    "/Users/walterwei/Library/CloudStorage/Dropbox/ff2gr_starter/example_data/tot_flux_2022_04_25.fits"
)


def _time_token(value: str) -> str:
    return (
        value.replace(":", "")
        .replace("-", "")
        .replace("T", "_")
        .replace(".", "")
        .replace("Z", "")
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyampp-sharp-sim",
        description="Run SHARP CEA patch selection, model synthesis, and microwave emission synthesis.",
    )
    parser.add_argument("--time", required=True, help="Requested observation time, e.g. 2025-11-26T15:47:52")
    parser.add_argument(
        "--freq-ghz",
        action="append",
        nargs="+",
        default=None,
        dest="freq_ghz",
        help=(
            "Microwave frequencies in GHz. Accepts space-separated values, comma-separated values, "
            "or a quoted Python-style list. If omitted, defaults to the frequencies stored in "
            f"{DEFAULT_FREQ_SOURCE_FITS}."
        ),
    )
    parser.add_argument("--nz", required=True, type=int, help="Vertical depth of the model box in voxels")
    parser.add_argument(
        "--line-nproc",
        type=int,
        default=0,
        help="Line-tracing process count. Use 0 for the native default/auto behavior.",
    )
    parser.add_argument("--mw-lib", required=True, help="Path to the microwave synthesis shared library")
    parser.add_argument("--ebtel", required=True, help="Path to the EBTEL .sav file")
    parser.add_argument("--radio-config", required=True, help="Path to the radio YAML config file")
    parser.add_argument("--out-root", required=True, help="Root output directory")
    parser.add_argument(
        "--use-potential",
        action="store_true",
        help="Skip NLFFF and reuse the potential field for GEN/CHR/radio synthesis",
    )
    parser.add_argument("--harpnum", action="append", type=int, default=None, help="Optional HARPNUM filter")
    parser.add_argument(
        "--max-time-delta-hours",
        type=float,
        default=12.0,
        help="Maximum absolute offset for nearest SHARP timestamp selection",
    )
    return parser


def _validate_required_path(path_text: str, label: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _freq_unit_scale_to_ghz(unit_text: str | None) -> float:
    unit = (unit_text or "Hz").strip().lower()
    if unit == "ghz":
        return 1.0
    if unit == "mhz":
        return 1.0e-3
    if unit == "khz":
        return 1.0e-6
    if unit == "hz":
        return 1.0e-9
    raise ValueError(f"Unsupported frequency unit in FITS metadata: {unit_text}")


@lru_cache(maxsize=1)
def _default_freqs_ghz() -> list[float]:
    path = DEFAULT_FREQ_SOURCE_FITS.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Default frequency source FITS does not exist: {path}")

    with fits.open(path) as hdul:
        if len(hdul) > 1 and getattr(hdul[1].data, "names", None):
            names = {name.lower(): name for name in hdul[1].data.names}
            if "cfreqs" in names:
                cfreqs = np.asarray(hdul[1].data[names["cfreqs"]], dtype=np.float64)
                return [float(value * 1.0e-9) for value in cfreqs.tolist()]

        header = hdul[0].header
        naxis3 = int(header["NAXIS3"])
        crval3 = float(header["CRVAL3"])
        cdelt3 = float(header["CDELT3"])
        scale = _freq_unit_scale_to_ghz(header.get("CUNIT3"))
        return [float((crval3 + idx * cdelt3) * scale) for idx in range(naxis3)]


def _parse_freq_token(token: str) -> list[float]:
    stripped = token.strip()
    if not stripped:
        return []

    if stripped.startswith(("[", "(")) and stripped.endswith(("]", ")")):
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            parsed = None
        else:
            if isinstance(parsed, (list, tuple)):
                return [float(value) for value in parsed]

    values: list[float] = []
    cleaned = stripped.strip("[]()")
    for item in cleaned.split(","):
        piece = item.strip()
        if piece:
            values.append(float(piece))
    return values


def _resolve_freqs_ghz(raw_values: Sequence[object] | None) -> list[float]:
    if not raw_values:
        return _default_freqs_ghz()

    freqs: list[float] = []
    for item in raw_values:
        if isinstance(item, (list, tuple)):
            for token in item:
                freqs.extend(_parse_freq_token(str(token)))
            continue
        freqs.extend(_parse_freq_token(str(item)))
    if not freqs:
        raise ValueError("No valid frequencies were parsed from --freq-ghz")
    return freqs


def _run_gx_pipeline(
    entry_box: Path,
    *,
    t_rec: str,
    gxmodel_dir: Path,
    observer_name: str = "sdo",
    use_potential: bool = False,
    line_nproc: int = 0,
) -> None:
    from pyampp.gxbox import gx_fov2box

    gx_fov2box.main(
        time=Time(t_rec).utc.isot,
        coords=None,
        hpc=False,
        hgc=False,
        hgs=False,
        cea=False,
        top=False,
        box_dims=None,
        dx_km=0.0,
        pad_frac=0.10,
        data_dir=str(gxmodel_dir / "unused_data_dir"),
        gxmodel_dir=str(gxmodel_dir),
        download_backend="drms",
        use_fido=False,
        force_download=False,
        entry_box=str(entry_box),
        save_empty_box=True,
        save_potential=True,
        save_bounds=True,
        save_nas=not use_potential,
        save_gen=True,
        save_chr=True,
        stop_after=None,
        empty_box_only=False,
        potential_only=False,
        nlfff_only=False,
        generic_only=False,
        use_potential=use_potential,
        skip_lines=False,
        center_vox=False,
        reduce_passed=None,
        line_nproc=int(line_nproc),
        euv=False,
        uv=False,
        sfq=False,
        observer_name=observer_name,
        fov_xc=None,
        fov_yc=None,
        fov_xsize=None,
        fov_ysize=None,
        square_fov=False,
        jump2potential=False,
        jump2bounds=False,
        jump2nlfff=False,
        jump2lines=False,
        jump2chromo=False,
        rebuild=False,
        rebuild_from_none=False,
        clone_only=False,
        info=False,
    )


def _final_chr_path(model_dir: Path, base_id: str, t_rec: str) -> Path:
    date_dir = model_dir / Time(t_rec).utc.to_datetime().strftime("%Y-%m-%d")
    candidates = [
        date_dir / f"{base_id}.NAS.GEN.CHR.h5",
        date_dir / f"{base_id}.NAS.CHR.h5",
        date_dir / f"{base_id}.POT.GEN.CHR.h5",
        date_dir / f"{base_id}.POT.CHR.h5",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate a CHR-stage HDF5 file for {base_id} under {date_dir}")


def _patch_root(out_root: Path, *, requested_time: str, selected_time: str, harpnum: int) -> Path:
    return out_root / _time_token(requested_time) / _time_token(selected_time) / f"HARP{int(harpnum)}"


def _process_record(
    record: SharpRecord,
    *,
    requested_time: str,
    selected_time: str,
    args: argparse.Namespace,
    radio_config,
    out_root: Path,
) -> dict[str, object]:
    patch_root = _patch_root(out_root, requested_time=requested_time, selected_time=selected_time, harpnum=record.harpnum)
    raw_dir = patch_root / "raw_sharp"
    entry_dir = patch_root / "entry_box"
    model_dir = patch_root / "model_stages"
    radio_dir = patch_root / "radio"

    print(f"\nProcessing HARP{record.harpnum} at {record.sample_time}")
    sharp_files = download_sharp_record_segments(
        record,
        raw_dir,
        force_download=False,
    )

    entry_result = build_sharp_entry_box(
        sharp_files,
        entry_dir / f"HARP{record.harpnum}.NONE.h5",
        harpnum=record.harpnum,
        t_rec=record.sample_time,
        requested_time=requested_time,
        nz=args.nz,
    )

    _run_gx_pipeline(
        entry_result.output_path,
        t_rec=entry_result.t_rec,
        gxmodel_dir=model_dir,
        use_potential=bool(args.use_potential),
        line_nproc=int(args.line_nproc),
    )
    chr_path = _final_chr_path(model_dir, entry_result.base_id, entry_result.t_rec)

    emission, synth_meta = compute_radio_emission(
        chr_path,
        mw_lib=args.mw_lib,
        ebtel_file=args.ebtel,
        freqs_ghz=args.freq_ghz,
        radio_config=radio_config,
    )
    geometry = synth_meta["geometry"]
    header = synth_meta["header"]
    box = synth_meta["box"]

    bundle_meta = {
        "requested_time": requested_time,
        "selected_time": selected_time,
        "selected_t_rec": record.t_rec,
        "harpnum": int(record.harpnum),
        "source_product": "hmi.sharp_cea_720s",
        "entry_box": str(entry_result.output_path),
        "model_hdf5": str(chr_path),
        "mw_lib": str(args.mw_lib),
        "ebtel": str(args.ebtel),
        "radio_config": str(args.radio_config),
    }
    bundle_path = write_radio_bundle(
        radio_dir / f"{entry_result.base_id}.radio.h5",
        emission,
        freqs_ghz=args.freq_ghz,
        metadata=bundle_meta,
    )
    fits_paths = write_radio_fits(
        radio_dir,
        emission,
        freqs_ghz=args.freq_ghz,
        box=box,
        header=header,
        geometry=geometry,
        file_stem=entry_result.base_id,
        extra_header={"harpnum": int(record.harpnum)},
    )
    return {
        "harpnum": int(record.harpnum),
        "patch_root": str(patch_root),
        "entry_box": str(entry_result.output_path),
        "model_hdf5": str(chr_path),
        "radio_bundle": str(bundle_path),
        "radio_fits": [str(path) for path in fits_paths],
        "geometry": asdict(geometry),
    }


def run(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    requested_time = Time(args.time).utc
    args.freq_ghz = _resolve_freqs_ghz(args.freq_ghz)
    args.mw_lib = _validate_required_path(args.mw_lib, "Microwave library")
    args.ebtel = _validate_required_path(args.ebtel, "EBTEL file")
    args.radio_config = _validate_required_path(args.radio_config, "Radio config")
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    radio_config = load_radio_config(args.radio_config)
    if argv is None and "--freq-ghz" not in sys.argv[1:]:
        print(f"Using {len(args.freq_ghz)} default frequencies from {DEFAULT_FREQ_SOURCE_FITS}")
    elif argv is not None and "--freq-ghz" not in list(argv):
        print(f"Using {len(args.freq_ghz)} default frequencies from {DEFAULT_FREQ_SOURCE_FITS}")
    selection = find_nearest_sharp_records(
        requested_time,
        max_time_delta_hours=float(args.max_time_delta_hours),
        harpnums=args.harpnum,
    )
    print(
        "Selected SHARP timestamp "
        f"{selection.selected_time} (requested {selection.requested_time}, "
        f"delta {selection.delta_seconds:.1f}s)"
    )
    print(f"Found {len(selection.records)} patch(es) at the selected timestamp.")

    failures: list[tuple[int, str]] = []
    successes: list[dict[str, object]] = []
    for record in selection.records:
        try:
            result = _process_record(
                record,
                requested_time=selection.requested_time,
                selected_time=selection.selected_time,
                args=args,
                radio_config=radio_config,
                out_root=out_root,
            )
            successes.append(result)
        except Exception as exc:
            failures.append((int(record.harpnum), str(exc)))
            print(f"HARP{record.harpnum} failed: {exc}", file=sys.stderr)

    print("\nSummary")
    for result in successes:
        print(
            f"HARP{result['harpnum']}: "
            f"entry={result['entry_box']} "
            f"model={result['model_hdf5']} "
            f"radio={result['radio_bundle']}"
        )
    if failures:
        print("Failures:", file=sys.stderr)
        for harpnum, message in failures:
            print(f"- HARP{harpnum}: {message}", file=sys.stderr)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(argv=argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
