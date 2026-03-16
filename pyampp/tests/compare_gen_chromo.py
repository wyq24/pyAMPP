#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import numpy as np
from scipy.io import readsav


GEN_MAP: Dict[str, str] = {
    "STARTIDX": "lines/start_idx",
    "ENDIDX": "lines/end_idx",
    "AVFIELD": "lines/av_field",
    "PHYSLENGTH": "lines/phys_length",
    "STATUS": "lines/voxel_status",
    "DR": "lines/dr",
}

CHR_MAP: Dict[str, Any] = {
    "CHROMO_IDX": "chromo/chromo_idx",
    "CHROMO_BCUBE": ["chromo/bx", "chromo/by", "chromo/bz"],
    "N_HTOT": "chromo/n_htot",
    "N_HI": "chromo/n_hi",
    "N_P": "chromo/n_p",
    "DZ": "chromo/dz",
    "CHROMO_N": "chromo/chromo_n",
    "CHROMO_T": "chromo/chromo_t",
    "CHROMO_LAYERS": "chromo/chromo_layers",
    "TR": "chromo/tr",
    "TR_H": "chromo/tr_h",
}


def _load_sav_box(path: str) -> Any:
    data = readsav(path, verbose=False)
    if "box" in data:
        return data["box"].flat[0]
    if "pbox" in data:
        return data["pbox"].flat[0]
    raise KeyError(f"No 'box' or 'pbox' in SAV: {path}")


def _shape_tuple(arr: np.ndarray) -> Tuple[int, ...]:
    return tuple(int(x) for x in arr.shape)


def _to_numpy(value: Any) -> np.ndarray:
    return np.asarray(value)


def _numeric_metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    if a.shape != b.shape:
        if a.size == b.size:
            a_flat = np.ravel(a)
            b_flat = np.ravel(b)
            shape_mode = "flattened"
        else:
            return {
                "comparable": False,
                "a_shape": _shape_tuple(a),
                "b_shape": _shape_tuple(b),
                "a_size": int(a.size),
                "b_size": int(b.size),
            }
    else:
        a_flat = np.ravel(a)
        b_flat = np.ravel(b)
        shape_mode = "direct"

    a_num = a_flat.astype(np.float64)
    b_num = b_flat.astype(np.float64)
    finite = np.isfinite(a_num) & np.isfinite(b_num)
    if not np.any(finite):
        return {
            "comparable": False,
            "reason": "no_finite_overlap",
            "shape_mode": shape_mode,
            "a_shape": _shape_tuple(a),
            "b_shape": _shape_tuple(b),
        }

    a_f = a_num[finite]
    b_f = b_num[finite]
    diff = a_f - b_f
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    maxabs = float(np.max(np.abs(diff)))
    mean_abs_ref = float(np.mean(np.abs(a_f)))
    rel_mae = float(mae / mean_abs_ref) if mean_abs_ref > 0 else None
    if a_f.size and np.std(a_f) > 0 and np.std(b_f) > 0:
        corr = float(np.corrcoef(a_f, b_f)[0, 1])
    else:
        corr = None

    out: Dict[str, Any] = {
        "comparable": True,
        "shape_mode": shape_mode,
        "a_shape": _shape_tuple(a),
        "b_shape": _shape_tuple(b),
        "mae": mae,
        "rmse": rmse,
        "maxabs": maxabs,
        "mean_abs_ref": mean_abs_ref,
        "rel_mae": rel_mae,
        "corr": corr,
    }

    if np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.integer):
        out["exact_match_fraction"] = float(np.mean(a_flat == b_flat))
    return out


def _read_h5_dataset(h5_path: str, dataset_path: str) -> np.ndarray | None:
    with h5py.File(h5_path, "r") as f:
        if dataset_path in f:
            return np.array(f[dataset_path])
    return None


def _compare_map(h5_path: str, sav_path: str, mapping: Dict[str, Any]) -> Dict[str, Any]:
    box = _load_sav_box(sav_path)
    fields = set(box.dtype.names or [])
    out: Dict[str, Any] = {"fields": {}, "missing": {"sav": [], "h5": []}}

    for sav_field, h5_target in mapping.items():
        if sav_field not in fields:
            out["missing"]["sav"].append(sav_field)
            continue

        sav_val = _to_numpy(box[sav_field])
        if isinstance(h5_target, list):
            comps = []
            for i, h5_ds in enumerate(h5_target):
                h5_val = _read_h5_dataset(h5_path, h5_ds)
                if h5_val is None:
                    out["missing"]["h5"].append(h5_ds)
                    continue
                if sav_field == "CHROMO_BCUBE":
                    if sav_val.ndim == 4 and sav_val.shape[-1] == 3:
                        sav_comp = sav_val[..., i]
                    elif sav_val.ndim == 4 and sav_val.shape[0] == 3:
                        sav_comp = sav_val[i, ...]
                    else:
                        sav_comp = sav_val
                    comps.append((h5_ds, _numeric_metrics(sav_comp, h5_val)))
                else:
                    comps.append((h5_ds, _numeric_metrics(sav_val, h5_val)))
            if comps:
                out["fields"][sav_field] = {name: m for name, m in comps}
            continue

        h5_val = _read_h5_dataset(h5_path, h5_target)
        if h5_val is None:
            out["missing"]["h5"].append(h5_target)
            continue
        out["fields"][sav_field] = _numeric_metrics(sav_val, h5_val)

    out["missing"]["sav"] = sorted(set(out["missing"]["sav"]))
    out["missing"]["h5"] = sorted(set(out["missing"]["h5"]))
    return out


def _line_counts(h5_gen: str, sav_gen: str) -> Dict[str, Any]:
    box = _load_sav_box(sav_gen)
    sav_start = _to_numpy(box["STARTIDX"])
    sav_end = _to_numpy(box["ENDIDX"])
    sav_status = _to_numpy(box["STATUS"])

    with h5py.File(h5_gen, "r") as f:
        h5_start = np.array(f["lines/start_idx"])
        h5_end = np.array(f["lines/end_idx"])
        h5_status = np.array(f["lines/voxel_status"])

    return {
        "sav_nonzero_startidx": int(np.count_nonzero(sav_start)),
        "h5_nonzero_startidx": int(np.count_nonzero(h5_start)),
        "sav_nonzero_endidx": int(np.count_nonzero(sav_end)),
        "h5_nonzero_endidx": int(np.count_nonzero(h5_end)),
        "sav_nonzero_status": int(np.count_nonzero(sav_status)),
        "h5_nonzero_status": int(np.count_nonzero(h5_status)),
    }


def _chromo_counts(h5_chr: str, sav_chr: str) -> Dict[str, Any]:
    box = _load_sav_box(sav_chr)
    sav_idx = _to_numpy(box["CHROMO_IDX"])
    with h5py.File(h5_chr, "r") as f:
        h5_idx = np.array(f["chromo/chromo_idx"])
    return {
        "sav_chromo_idx_len": int(sav_idx.size),
        "h5_chromo_idx_len": int(h5_idx.size),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare GEN line-parameters and CHR arrays between H5 and SAV models."
    )
    parser.add_argument(
        "--stage-report",
        default="reports/parity_review/data/stage_compare_resume_none_vs_sav_full.json",
        help="JSON report containing paired H5/SAV files under ['pairs'].",
    )
    parser.add_argument(
        "--out",
        default="reports/parity_review/data/gen_chromo_compare_resume_none_vs_sav.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    src = json.loads(Path(args.stage_report).read_text(encoding="utf-8"))
    pairs = src.get("pairs", {})

    report: Dict[str, Any] = {
        "source_report": str(args.stage_report),
        "pairs": {},
    }

    for branch, branch_pairs in pairs.items():
        if "NAS.GEN" not in branch_pairs or "NAS.CHR" not in branch_pairs:
            continue
        gen_h5 = branch_pairs["NAS.GEN"]["h5"]
        gen_sav = branch_pairs["NAS.GEN"]["sav"]
        chr_h5 = branch_pairs["NAS.CHR"]["h5"]
        chr_sav = branch_pairs["NAS.CHR"]["sav"]

        report["pairs"][branch] = {
            "gen": {
                "paths": {"h5": gen_h5, "sav": gen_sav},
                "line_counts": _line_counts(gen_h5, gen_sav),
                "comparison": _compare_map(gen_h5, gen_sav, GEN_MAP),
            },
            "chr": {
                "paths": {"h5": chr_h5, "sav": chr_sav},
                "chromo_counts": _chromo_counts(chr_h5, chr_sav),
                "comparison": _compare_map(chr_h5, chr_sav, CHR_MAP),
            },
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
