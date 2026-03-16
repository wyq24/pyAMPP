from __future__ import annotations

import argparse
from pathlib import Path

import astropy.units as u
import numpy as np

from pyampp.gxbox.view_h5 import prepare_model_for_viewer


def _format_xyz_rows(rows: np.ndarray) -> str:
    arr = np.asarray(rows, dtype=float).reshape((-1, 3))
    lines = []
    for idx, row in enumerate(arr):
        lines.append(f"  {idx}: [{row[0]: .6f}, {row[1]: .6f}, {row[2]: .6f}]")
    return "\n".join(lines)


def _skycoord_to_xyz_rows(coords) -> np.ndarray | None:
    if coords is None:
        return None
    try:
        return np.column_stack(
            [
                coords.x.to_value(u.Mm),
                coords.y.to_value(u.Mm),
                coords.z.to_value(u.Mm),
            ]
        ).astype(float)
    except Exception:
        return None


def _print_block(label: str, local_rows: np.ndarray | None, world_rows: np.ndarray | None) -> None:
    print(f"{label}:")
    if local_rows is None:
        print("  local_mm: <unavailable>")
    else:
        print("  local_mm:")
        print(_format_xyz_rows(local_rows))
    if world_rows is None:
        print("  world_box_frame_mm: <unavailable>")
    else:
        print("  world_box_frame_mm:")
        print(_format_xyz_rows(world_rows))


def main() -> int:
    parser = argparse.ArgumentParser(description="Print BOX/FOV corner coordinates used by the gxbox viewers.")
    parser.add_argument("model_path", help="Path to a saved model (.h5 or .sav).")
    parser.add_argument(
        "--what",
        choices=("box", "fov", "both"),
        default="both",
        help="Which corners to print.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    box, _obs_time, _b3dtype, _temp_h5 = prepare_model_for_viewer(model_path)

    if args.what in ("box", "both"):
        _print_block(
            "BOX",
            box.model_box_corners_local_mm(),
            _skycoord_to_xyz_rows(box.model_box_corners_world()),
        )
    if args.what in ("fov", "both"):
        _print_block(
            "FOV",
            box.fov_box_corners_local_mm(),
            _skycoord_to_xyz_rows(box.fov_box_corners_world()),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
