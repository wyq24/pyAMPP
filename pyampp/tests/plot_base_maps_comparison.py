#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.io import readsav


def _as_ndarray(value) -> np.ndarray:
    out = np.asarray(value)
    if out.dtype.byteorder not in ("=", "|"):
        out = out.byteswap().view(out.dtype.newbyteorder("="))
    return out


def _load_h5_base(path: Path):
    with h5py.File(path, "r") as f:
        return {k: _as_ndarray(f["base"][k][:]) for k in ("bx", "by", "bz")}


def _load_sav_base(path: Path):
    data = readsav(str(path), verbose=False)
    if "box" in data:
        box = data["box"].flat[0]
    elif "pbox" in data:
        box = data["pbox"].flat[0]
    else:
        raise KeyError(f"No 'box' or 'pbox' structure found in {path}")
    base = box["base"].flat[0]
    return {
        "bx": _as_ndarray(base["BX"]),
        "by": _as_ndarray(base["BY"]),
        "bz": _as_ndarray(base["BZ"]),
    }


def _relative_diff(a: np.ndarray, b: np.ndarray, mode: str = "bounded") -> np.ndarray:
    da = _as_ndarray(a).astype(np.float64, copy=False)
    db = _as_ndarray(b).astype(np.float64, copy=False)
    diff = da - db
    if mode == "signed":
        denom = da + db
    else:
        denom = np.abs(da) + np.abs(db)
    out = np.full(da.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(diff) & np.isfinite(denom)
    nz = finite & (np.abs(denom) > 0.0)
    out[nz] = diff[nz] / denom[nz]
    return out


def _align_shape(sav_map: np.ndarray, h5_map: np.ndarray) -> np.ndarray:
    if sav_map.shape == h5_map.shape:
        return sav_map
    if sav_map.T.shape == h5_map.shape:
        return sav_map.T
    raise ValueError(f"Shape mismatch: SAV {sav_map.shape}, H5 {h5_map.shape}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot base-map comparison panel: SAV, H5, absolute diff, relative diff."
    )
    parser.add_argument("--h5", required=True, type=Path, help="Path to HDF5 model.")
    parser.add_argument("--sav", required=True, type=Path, help="Path to SAV model.")
    parser.add_argument("--out", required=True, type=Path, help="Output PNG path.")
    parser.add_argument("--title", type=str, default="Base Map Comparison")
    parser.add_argument(
        "--relative-mode",
        choices=("bounded", "signed"),
        default="bounded",
        help=(
            "Relative mode: bounded=(H5-SAV)/(abs(H5)+abs(SAV)), "
            "signed=(H5-SAV)/(H5+SAV)."
        ),
    )
    args = parser.parse_args()

    h5 = _load_h5_base(args.h5)
    sav = _load_sav_base(args.sav)

    components = (("bx", "Bx"), ("by", "By"), ("bz", "Bz"))

    fig, axs = plt.subplots(3, 4, figsize=(18, 13), constrained_layout=False)
    # Reserve top space so title stays clear above colorbars and row-1 panels.
    fig.subplots_adjust(top=0.86, hspace=0.70, wspace=0.35)
    fig.suptitle(args.title, fontsize=16, y=0.975)

    for r, (key, label) in enumerate(components):
        a = _align_shape(sav[key], h5[key])
        b = h5[key]
        d = b - a
        rel = _relative_diff(b, a, mode=args.relative_mode)

        v = np.nanpercentile(np.abs(np.concatenate([a.ravel(), b.ravel()])), 99.5)
        vd = np.nanpercentile(np.abs(d.ravel()), 99.5)
        v = float(v) if np.isfinite(v) else 0.0
        vd = float(vd) if np.isfinite(vd) else 0.0
        if v <= 0.0:
            v = 1e-12
        if vd <= 0.0:
            vd = 1e-12
        norm_row = TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)
        norm_diff = TwoSlopeNorm(vmin=-vd, vcenter=0.0, vmax=vd)
        if args.relative_mode == "bounded":
            norm_rel = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
        else:
            vr = np.nanpercentile(np.abs(rel[np.isfinite(rel)]), 99.5) if np.any(np.isfinite(rel)) else 1.0
            vr = max(vr, 1e-12)
            norm_rel = TwoSlopeNorm(vmin=-vr, vcenter=0.0, vmax=vr)

        im0 = axs[r, 0].imshow(a, origin="lower", cmap="RdBu_r", norm=norm_row)
        axs[r, 0].set_title(f"{label} | SAV\nmin={np.nanmin(a):.3e}, max={np.nanmax(a):.3e}", fontsize=10)
        axs[r, 0].set_xlabel("X [pix]")
        axs[r, 0].set_ylabel("Y [pix]")
        cb0 = fig.colorbar(
            im0, ax=axs[r, 0], orientation="horizontal", location="top", fraction=0.042, pad=0.10
        )
        cb0.set_label("Magnetic field [G]")

        im1 = axs[r, 1].imshow(b, origin="lower", cmap="RdBu_r", norm=norm_row)
        axs[r, 1].set_title(f"{label} | H5\nmin={np.nanmin(b):.3e}, max={np.nanmax(b):.3e}", fontsize=10)
        axs[r, 1].set_xlabel("X [pix]")
        axs[r, 1].set_ylabel("Y [pix]")
        cb1 = fig.colorbar(
            im1, ax=axs[r, 1], orientation="horizontal", location="top", fraction=0.042, pad=0.10
        )
        cb1.set_label("Magnetic field [G]")

        im2 = axs[r, 2].imshow(d, origin="lower", cmap="RdBu_r", norm=norm_diff)
        axs[r, 2].set_title(
            f"{label} | ABS DIFF (H5-SAV)\nmin={np.nanmin(d):.3e}, max={np.nanmax(d):.3e}",
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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
