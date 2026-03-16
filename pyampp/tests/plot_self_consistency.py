#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.io import readsav

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)


def _as_ndarray(value) -> np.ndarray:
    out = np.asarray(value)
    if out.dtype.byteorder not in ("=", "|"):
        out = out.byteswap().view(out.dtype.newbyteorder("="))
    return out


def _metrics(base: np.ndarray, z0: np.ndarray) -> dict:
    a = _as_ndarray(base).astype(np.float64, copy=False)
    b = _as_ndarray(z0).astype(np.float64, copy=False)
    d = b - a
    finite = np.isfinite(a) & np.isfinite(b)
    aa = a[finite]
    bb = b[finite]
    dd = d[finite]
    if aa.size == 0:
        return {"mae": np.nan, "rmse": np.nan, "corr": np.nan, "min": np.nan, "max": np.nan}
    corr = np.nan
    if aa.size > 1 and np.nanstd(aa) > 0 and np.nanstd(bb) > 0:
        corr = float(np.corrcoef(aa, bb)[0, 1])
    return {
        "mae": float(np.nanmean(np.abs(dd))),
        "rmse": float(np.sqrt(np.nanmean(dd * dd))),
        "corr": corr,
        "min": float(np.nanmin(dd)),
        "max": float(np.nanmax(dd)),
    }


def _load_h5(path: Path) -> tuple[dict, dict]:
    with h5py.File(path, "r") as f:
        base = {k: _as_ndarray(f["base"][k][:]) for k in ("bx", "by", "bz")}  # [ny, nx]
        coro = {k: _as_ndarray(f["corona"][k][:]) for k in ("bx", "by", "bz")}  # [nz, ny, nx]
    z0 = {k: coro[k][0, :, :] for k in ("bx", "by", "bz")}
    return base, z0


def _load_sav(path: Path) -> tuple[dict, dict]:
    data = readsav(str(path), python_dict=True, verbose=False)
    box = data["box"][0]
    base_raw = box["BASE"][0]
    bcube = _as_ndarray(box["BCUBE"])  # scipy restores as [comp, z, y, x]
    base = {
        "bx": _as_ndarray(base_raw["BX"]),
        "by": _as_ndarray(base_raw["BY"]),
        "bz": _as_ndarray(base_raw["BZ"]),
    }
    z0 = {
        "bx": bcube[0, 0, :, :],
        "by": bcube[1, 0, :, :],
        "bz": bcube[2, 0, :, :],
    }
    return base, z0


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot self-consistency: base vs z=0 layer.")
    parser.add_argument("--input-type", choices=("h5", "sav"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-png", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--title", type=str, required=True)
    args = parser.parse_args()

    if args.input_type == "h5":
        base, z0 = _load_h5(args.input)
        z0_label = "corona z=0"
    else:
        base, z0 = _load_sav(args.input)
        z0_label = "BCUBE z=0"

    report = {k: _metrics(base[k], z0[k]) for k in ("bx", "by", "bz")}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "input_type": args.input_type,
                "input": str(args.input),
                "title": args.title,
                "metrics": report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fig, axs = plt.subplots(3, 4, figsize=(18, 13), constrained_layout=False)
    # Keep enough room for a readable panel title and top colorbars.
    fig.subplots_adjust(top=0.86, hspace=0.70, wspace=0.35)
    fig.suptitle(args.title, fontsize=16, y=0.975)

    for r, (key, label) in enumerate((("bx", "Bx"), ("by", "By"), ("bz", "Bz"))):
        a = _as_ndarray(base[key]).astype(np.float64, copy=False)
        b = _as_ndarray(z0[key]).astype(np.float64, copy=False)
        d = b - a
        denom = np.abs(a) + np.abs(b)
        rel = np.full_like(d, np.nan, dtype=np.float64)
        nz = np.isfinite(d) & np.isfinite(denom) & (denom > 0.0)
        rel[nz] = d[nz] / denom[nz]

        v = np.nanpercentile(np.abs(np.concatenate([a.ravel(), b.ravel()])), 99.5)
        vd = np.nanpercentile(np.abs(d.ravel()), 99.5)
        if not np.isfinite(v) or v <= 0.0:
            v = 1e-12
        if not np.isfinite(vd) or vd <= 0.0:
            vd = 1e-12
        norm_row = TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)
        norm_diff = TwoSlopeNorm(vmin=-vd, vcenter=0.0, vmax=vd)
        norm_rel = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

        im0 = axs[r, 0].imshow(a, origin="lower", cmap="RdBu_r", norm=norm_row)
        axs[r, 0].set_title(f"{label} | BASE\nmin={np.nanmin(a):.3e}, max={np.nanmax(a):.3e}", fontsize=10)
        axs[r, 0].set_xlabel("X [pix]")
        axs[r, 0].set_ylabel("Y [pix]")
        cb0 = fig.colorbar(
            im0, ax=axs[r, 0], orientation="horizontal", location="top", fraction=0.042, pad=0.10
        )
        cb0.set_label("Magnetic field [G]")

        im1 = axs[r, 1].imshow(b, origin="lower", cmap="RdBu_r", norm=norm_row)
        axs[r, 1].set_title(
            f"{label} | {z0_label.upper()}\nmin={np.nanmin(b):.3e}, max={np.nanmax(b):.3e}",
            fontsize=10,
        )
        axs[r, 1].set_xlabel("X [pix]")
        axs[r, 1].set_ylabel("Y [pix]")
        cb1 = fig.colorbar(
            im1, ax=axs[r, 1], orientation="horizontal", location="top", fraction=0.042, pad=0.10
        )
        cb1.set_label("Magnetic field [G]")

        m = report[key]
        im2 = axs[r, 2].imshow(d, origin="lower", cmap="RdBu_r", norm=norm_diff)
        axs[r, 2].set_title(
            f"{label} | ABS DIFF ({z0_label.upper()}-BASE)\n"
            f"MAE={m['mae']:.3e}, RMSE={m['rmse']:.3e}, corr={m['corr']:.4f}",
            fontsize=10,
        )
        axs[r, 2].set_xlabel("X [pix]")
        axs[r, 2].set_ylabel("Y [pix]")
        cb2 = fig.colorbar(
            im2, ax=axs[r, 2], orientation="horizontal", location="top", fraction=0.042, pad=0.10
        )
        cb2.set_label("Difference [G]")

        im3 = axs[r, 3].imshow(rel, origin="lower", cmap="RdBu_r", norm=norm_rel)
        axs[r, 3].set_title(
            f"{label} | REL DIFF\nmin={np.nanmin(rel):.3e}, max={np.nanmax(rel):.3e}",
            fontsize=10,
        )
        axs[r, 3].set_xlabel("X [pix]")
        axs[r, 3].set_ylabel("Y [pix]")
        cb3 = fig.colorbar(
            im3, ax=axs[r, 3], orientation="horizontal", location="top", fraction=0.042, pad=0.10
        )
        cb3.set_label("Relative difference")

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=150)
    plt.close(fig)
    print(args.out_json)
    print(args.out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
