from __future__ import annotations

"""Reusable API contracts for focused map/box selection GUIs.

This module intentionally contains no Qt code. It defines the data contracts
that a purpose-built visualization GUI (for example, a post-download
FOV/box-selector) can use to communicate geometry changes back to ``pyAMPP``.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol


class CoordMode(StrEnum):
    """Coordinate representation used by the GUI center fields."""

    HPC = "hpc"
    HGC = "hgc"
    HGS = "hgs"


@dataclass(slots=True)
class BoxGeometrySelection:
    """Workflow-relevant box geometry values in the same shape as pyAMPP GUI fields.

    These map directly to the existing ``PyAmppGUI`` input widgets:
    - ``coord_x`` / ``coord_y`` -> ``coord_x_edit`` / ``coord_y_edit``
    - ``grid_x`` / ``grid_y`` / ``grid_z`` -> ``grid_*_edit``
    - ``dx_km`` -> ``res_edit`` (CLI ``--dx-km``)
    """

    coord_mode: CoordMode
    coord_x: float
    coord_y: float
    grid_x: int
    grid_y: int
    grid_z: int
    dx_km: float

    def as_gui_text_fields(self) -> dict[str, str]:
        """Return values formatted for direct assignment to QLineEdit widgets."""
        return {
            "coord_x_edit": f"{self.coord_x:.2f}",
            "coord_y_edit": f"{self.coord_y:.2f}",
            "grid_x_edit": str(int(self.grid_x)),
            "grid_y_edit": str(int(self.grid_y)),
            "grid_z_edit": str(int(self.grid_z)),
            "res_edit": f"{float(self.dx_km):.2f}",
        }


@dataclass(slots=True)
class SelectorSessionInput:
    """Inputs needed to launch a future standalone map/box selector GUI.

    ``map_ids`` and ``initial_map_id`` are UI-only state for visualization and
    are not part of the geometry return contract.
    """

    time_iso: str
    data_dir: str
    geometry: BoxGeometrySelection
    fov: "DisplayFovSelection | None" = None
    square_fov: bool = False
    allow_geometry_edit: bool = True
    map_ids: tuple[str, ...] = ()
    map_files: dict[str, str] | None = None
    refmaps: dict[str, dict[str, Any]] | None = None
    base_maps: dict[str, Any] | None = None
    base_geometry: "BoxGeometrySelection | None" = None
    map_source_mode: str = "auto"
    initial_map_id: str | None = None
    pad_frac: float | None = None


@dataclass(slots=True)
class DisplayFovSelection:
    """Observer-plane image FOV used for synthetic image framing and canvas centering.

    This is intentionally separate from the model box geometry. Values are expressed
    in helioprojective (observer-plane) arcsec, regardless of the red-box input
    coordinate mode.
    """

    center_x_arcsec: float
    center_y_arcsec: float
    width_arcsec: float
    height_arcsec: float

    def as_gui_text_fields(self) -> dict[str, str]:
        return {
            "fov_x_edit": f"{self.center_x_arcsec:.2f}",
            "fov_y_edit": f"{self.center_y_arcsec:.2f}",
            "fov_w_edit": f"{self.width_arcsec:.2f}",
            "fov_h_edit": f"{self.height_arcsec:.2f}",
        }


@dataclass(slots=True)
class SelectorDialogResult:
    """Accepted result returned by the standalone selector dialog."""

    geometry: BoxGeometrySelection
    fov: "DisplayFovSelection | None" = None
    square_fov: bool = False


class GeometrySelectionConsumer(Protocol):
    """Callback protocol for receiving accepted geometry selections."""

    def apply_geometry_selection(self, selection: BoxGeometrySelection) -> None:
        """Apply an accepted geometry selection to the host GUI/application."""


__all__ = [
    "CoordMode",
    "BoxGeometrySelection",
    "DisplayFovSelection",
    "SelectorDialogResult",
    "SelectorSessionInput",
    "GeometrySelectionConsumer",
]
