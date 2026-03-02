from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Iterable, Optional

import numpy as np
import astropy.units as u
import matplotlib.colors as mcolors
from astropy.io import fits
from astropy.coordinates import SkyCoord
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QStyle,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from sunpy.map import Map
from sunpy.coordinates import Heliocentric, HeliographicCarrington, HeliographicStonyhurst, Helioprojective
from sunpy.visualization import colormaps as sunpy_colormaps

from .box import Box
from .boxutils import hmi_b2ptr, hmi_disambig, load_sunpy_map_compat, read_b3d_h5
from .magfield_viewer import MagFieldViewer, generate_streamlines_from_line_seeds
from .selector_api import BoxGeometrySelection, CoordMode, DisplayFovBoxSelection, DisplayFovSelection, SelectorSessionInput
from .view_h5 import _viewer_camera_vectors, can_prepare_model_for_viewer, prepare_model_for_viewer

_DISPLAY_MAP_ALIASES = {
    "Bz": "magnetogram",
    "Ic": "continuum",
    "Br": "br",
    "Bp": "bp",
    "Bt": "bt",
    "Bx": "bx",
    "By": "by",
}

_HMI_VECTOR_SEGMENTS = ("field", "inclination", "azimuth", "disambig")
_HMI_DISPLAY_KEYS = {"magnetogram", "continuum", "br", "bp", "bt"}
_SIGNED_MAGNETIC_KEYS = {"magnetogram", "br", "bp", "bt", "bx", "by", "bz"}
_TRANSVERSE_MAGNETIC_KEYS = {"bp", "bt"}
_VERT_CURRENT_KEYS = {"Vert_current"}
_CHROMO_MASK_KEYS = {"chromo_mask"}
_AIA_REFERENCE_IDS = ("171", "193", "211", "304", "335", "1600", "1700", "131", "94")
_HMI_VECTOR_DISPLAY_KEYS = {"br", "bp", "bt"}
_AIA_COLOR_KEYS = {"94", "131", "1600", "1700", "171", "193", "211", "304", "335"}
_EMBEDDED_REFMAP_FLAG = "PYEMBED"


@dataclass
class MapBoxViewState:
    """
    Reusable state container for focused map+box visualization tools.

    This is intentionally lightweight and plotting-backend agnostic so it can be
    reused by multiple future GUIs (FOV selector, box inspector, model preview, ...).
    """

    session_input: SelectorSessionInput
    selected_context_id: Optional[str] = None
    selected_bottom_id: Optional[str] = None
    geometry: Optional[BoxGeometrySelection] = None
    fov: Optional[DisplayFovSelection] = None
    fov_box: Optional[DisplayFovBoxSelection] = None
    map_files: dict[str, str] | None = None
    refmaps: dict | None = None
    base_maps: dict | None = None
    base_geometry: Optional[BoxGeometrySelection] = None
    map_source_mode: str = "auto"
    square_fov: bool = False


class _SquareCanvasHost(QWidget):
    """Keep the embedded plot canvas square and centered."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._canvas = None
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_canvas(self, canvas: QWidget) -> None:
        self._canvas = canvas
        self._canvas.setParent(self)
        self._canvas.show()
        self._reposition_canvas()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_canvas()

    def _reposition_canvas(self) -> None:
        if self._canvas is None:
            return
        w, h = self.width(), self.height()
        side = max(1, min(w, h))
        x = (w - side) // 2
        y = (h - side) // 2
        self._canvas.setGeometry(x, y, side, side)


class MapBoxDisplayWidget(QWidget):
    """
    Reusable widget shell for map display + interactive box overlays.

    Current implementation provides:
    - SunPy map plotting using the map's native WCS projection
    - a static box-outline overlay derived from the current geometry state
    - a stable API for future drag/resize interaction layers
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._svg_dir = Path(__file__).resolve().parents[2] / "docs" / "svg"
        self._state: Optional[MapBoxViewState] = None
        self._geometry_change_callback = None
        self._map_summary_cache: dict[str, str] = {}
        self._loaded_map_cache = {}
        self._raw_map_cache = {}
        self._cache_lock = threading.RLock()
        self._background_cache_generation = 0
        self._background_cache_thread = None
        self._current_map = None
        self._current_axes = None
        self._overlay_rect = None
        self._overlay_bbox_rect = None
        self._projected_box_bbox_rect = None
        self._projected_box_fov = None
        self._overlay_center_artist = None
        self._overlay_corner_artists = []
        self._drag_state = None
        self._entry_box_path: Optional[Path] = None
        self._viewer3d = None
        self._viewer3d_temp_h5_path: Optional[Path] = None
        self._viewer3d_watchdog = QTimer(self)
        self._viewer3d_watchdog.setInterval(400)
        self._viewer3d_watchdog.timeout.connect(self._check_viewer3d_state)
        self._hidden_for_live_3d = False
        self._viewer3d_close_handled = False
        self._committed_line_seeds = None
        self._fieldline_frame_hcc = None
        self._fieldline_frame_obs = None
        self._fieldline_streamlines = []
        self._fieldline_z_base = 0.0
        self._fieldline_artists = []
        self._map_info_callback = None
        self._status_callback = None
        self._fov_change_callback = None
        self._last_map_info_text = "Map info: <uninitialized>"
        self._last_status_text = "Map/box display initialized"
        self._last_context_summary_text = "Context map: <uninitialized>"
        self._last_bottom_summary_text = "Base map: <uninitialized>"
        self._view_mode = "box_fov"
        self._full_view_limits = None
        self._interaction_mode = "auto"
        self._mouse_actions_enabled = False
        self._geometry_edit_enabled = True
        self._action_state_callback = None
        self._fig = Figure(figsize=(7.5, 5.0))
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._cid_press = self._canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self._cid_move = self._canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self._cid_release = self._canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self._cid_scroll = self._canvas.mpl_connect("scroll_event", self._on_scroll)
        self._canvas_host = _SquareCanvasHost()
        self._canvas_host.set_canvas(self._canvas)
        self._nav_toolbar = NavigationToolbar(self._canvas, self)

        self._full_view_btn = self._make_svg_button("expand.svg", "Full Sun View", self.show_full_sun_view)
        self._box_view_btn = self._make_svg_button("shrink.svg", "Zoom canvas to image FOV", self.show_box_fov_view)
        self._recompute_fov_btn = self._make_svg_button("rectangle-horizontal.svg", "Recompute image FOV from projected 3D box", self.recompute_fov_from_box)
        self._control_mode_label = QLabel("BOX Controls")
        self._left_btn = self._make_svg_button("arrow-left.svg", "Move box center left", lambda: self._nudge_primary_center("x", -1))
        self._right_btn = self._make_svg_button("arrow-right.svg", "Move box center right", lambda: self._nudge_primary_center("x", +1))
        self._down_btn = self._make_svg_button("arrow-down.svg", "Move box center down", lambda: self._nudge_primary_center("y", -1))
        self._up_btn = self._make_svg_button("arrow-up.svg", "Move box center up", lambda: self._nudge_primary_center("y", +1))
        self._x_minus_btn = self._make_svg_button("shrink-horizontal.svg", "Decrease X box size", lambda: self._nudge_primary_size("x", -1))
        self._x_plus_btn = self._make_svg_button("expand-horizontal.svg", "Increase X box size", lambda: self._nudge_primary_size("x", +1))
        self._y_minus_btn = self._make_svg_button("shrink-vertical.svg", "Decrease Y box size", lambda: self._nudge_primary_size("y", -1))
        self._y_plus_btn = self._make_svg_button("expand-vertical.svg", "Increase Y box size", lambda: self._nudge_primary_size("y", +1))
        self._xy_minus_btn = self._make_svg_button("shrink.svg", "Decrease X and Y box size together", lambda: self._nudge_primary_size_xy(-1))
        self._xy_plus_btn = self._make_svg_button("expand.svg", "Increase X and Y box size together", lambda: self._nudge_primary_size_xy(+1))
        self._zoom_in_btn = self._make_svg_button("zoom-in.svg", "Zoom In (centered on image FOV)", lambda: self._scale_view(1 / 1.25))
        self._zoom_out_btn = self._make_svg_button("zoom-out.svg", "Zoom Out (centered on image FOV)", lambda: self._scale_view(1.25))
        self._can_open_3d = False
        self._can_clear_lines = False

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(4)
        toolbar.addWidget(self._full_view_btn)
        toolbar.addWidget(self._box_view_btn)
        toolbar.addWidget(self._recompute_fov_btn)
        toolbar.addWidget(self._zoom_in_btn)
        toolbar.addWidget(self._zoom_out_btn)
        toolbar.addSpacing(8)
        toolbar.addWidget(self._control_mode_label)
        toolbar.addSpacing(8)
        toolbar.addWidget(self._left_btn)
        toolbar.addWidget(self._right_btn)
        toolbar.addWidget(self._down_btn)
        toolbar.addWidget(self._up_btn)
        toolbar.addSpacing(8)
        toolbar.addWidget(self._x_minus_btn)
        toolbar.addWidget(self._x_plus_btn)
        toolbar.addWidget(self._y_minus_btn)
        toolbar.addWidget(self._y_plus_btn)
        toolbar.addWidget(self._xy_minus_btn)
        toolbar.addWidget(self._xy_plus_btn)
        toolbar.addStretch()

        layout = QVBoxLayout(self)
        layout.addLayout(toolbar)
        layout.addWidget(self._canvas_host, stretch=1)
        layout.addWidget(self._nav_toolbar)
        self._refresh_control_mode_ui()

    def _make_mode_button(self, text: str, mode: str, checked: bool = False) -> QToolButton:
        btn = QToolButton(self)
        btn.setCheckable(True)
        btn.setAutoRaise(False)
        btn.setIconSize(QSize(32, 32))
        btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
        icon, fallback = self._mode_button_icon(mode)
        btn.setIcon(icon)
        if icon.isNull():
            btn.setText(fallback)
            btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        btn.setToolTip(text)
        btn.setCheckable(True)
        btn.setChecked(checked)
        btn.clicked.connect(lambda _=False, m=mode: self._set_interaction_mode(m))
        self._mode_group.addButton(btn)
        return btn

    def _make_glyph_button(self, glyph: str, tooltip: str, callback) -> QToolButton:
        btn = QToolButton(self)
        btn.setText(glyph)
        btn.setToolTip(tooltip)
        btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        btn.setAutoRaise(False)
        btn.setFixedSize(24, 24)
        f = QFont(btn.font())
        f.setPointSize(10)
        btn.setFont(f)
        btn.clicked.connect(callback)
        return btn

    def _make_text_button(self, text: str, tooltip: str, callback) -> QToolButton:
        btn = QToolButton(self)
        btn.setText(text)
        btn.setToolTip(tooltip)
        btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        btn.setAutoRaise(False)
        btn.setMinimumHeight(24)
        btn.clicked.connect(callback)
        return btn

    def _make_svg_button(self, svg_name: str, tooltip: str, callback) -> QToolButton:
        btn = QToolButton(self)
        btn.setToolTip(tooltip)
        btn.setAutoRaise(False)
        btn.setFixedSize(32, 32)
        btn.setIconSize(QSize(20, 20))
        icon_path = self._svg_dir / svg_name
        if icon_path.exists():
            btn.setIcon(QIcon(str(icon_path)))
        else:
            btn.setText("?")
            btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        btn.clicked.connect(callback)
        return btn

    def _make_icon_button(self, icon_names, tooltip, callback, fallback_text="") -> QToolButton:
        btn = QToolButton(self)
        btn.setAutoRaise(False)
        btn.setIconSize(QSize(32, 32))
        btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
        icon = self._theme_icon(icon_names)
        if icon.isNull():
            icon = self._fallback_std_icon()
        if not icon.isNull():
            btn.setIcon(icon)
        else:
            btn.setText(fallback_text)
            btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        btn.setToolTip(tooltip)
        btn.clicked.connect(callback)
        return btn

    def _mode_button_icon(self, mode: str):
        mapping = {
            "auto": (["transform-move"], "◎"),
            "move": (["transform-move"], "✛"),
            "resize_xy": (["transform-scale"], "⤡"),
            "resize_x": (["object-flip-horizontal"], "↔"),
            "resize_y": (["object-flip-vertical"], "↕"),
        }
        names, fallback = mapping.get(mode, ([], mode))
        icon = self._theme_icon(names)
        if icon.isNull():
            # Use standard cursor-like fallback icons where possible.
            if mode == "move":
                icon = self.style().standardIcon(QStyle.SP_ArrowUp)
            elif mode in {"resize_x", "resize_y", "resize_xy"}:
                icon = self.style().standardIcon(QStyle.SP_TitleBarShadeButton)
        return icon, fallback

    @staticmethod
    def _theme_icon(names) -> QIcon:
        for name in names:
            icon = QIcon.fromTheme(name)
            if not icon.isNull():
                return icon
        return QIcon()

    def _fallback_std_icon(self) -> QIcon:
        try:
            return self.style().standardIcon(QStyle.SP_ArrowRight)
        except Exception:
            return QIcon()

    def _set_interaction_mode(self, mode: str) -> None:
        self._interaction_mode = mode
        self._refresh_status_text()
        self._update_cursor_for_mode()

    def initialize(self, session_input: SelectorSessionInput) -> None:
        selected_context_id = self._default_context_id(session_input)
        selected_bottom_id = self._default_bottom_id(session_input)
        self._map_summary_cache.clear()
        self._invalidate_map_caches()
        self._state = MapBoxViewState(
            session_input=session_input,
            selected_context_id=selected_context_id,
            selected_bottom_id=selected_bottom_id,
            geometry=session_input.geometry,
            fov=session_input.fov,
            fov_box=session_input.fov_box,
            map_files=dict(session_input.map_files or {}),
            refmaps=dict(session_input.refmaps or {}),
            base_maps=dict(session_input.base_maps or {}),
            base_geometry=session_input.base_geometry,
            map_source_mode=str(session_input.map_source_mode or "auto"),
            square_fov=bool(session_input.square_fov),
        )
        self._refresh_status_text()
        self._refresh_map_info()
        self._refresh_plot()
        self._refresh_fieldlines_from_committed_seeds()
        self._update_fov_control_enabled_state()
        self._start_background_cache_build()

    def set_available_maps(self, map_ids: Iterable[str]) -> None:
        if self._state is None:
            return
        map_ids = list(map_ids)
        self._state.session_input.map_ids = map_ids
        if self._state.selected_context_id not in map_ids:
            self._state.selected_context_id = self._default_context_id(self._state.session_input)
        if self._state.selected_bottom_id not in map_ids:
            self._state.selected_bottom_id = self._default_bottom_id(self._state.session_input)
        self._refresh_status_text()
        self._refresh_map_info()
        self._refresh_plot()

    def set_context_map_id(self, map_id: Optional[str]) -> None:
        if self._state is None:
            return
        self._state.selected_context_id = map_id
        self._refresh_status_text()
        self._refresh_map_info()
        self._refresh_plot(preserve_current_view=self._should_preserve_pixel_view())

    def set_bottom_map_id(self, map_id: Optional[str]) -> None:
        if self._state is None:
            return
        self._state.selected_bottom_id = map_id
        self._refresh_status_text()
        self._refresh_map_info()
        self._refresh_plot(preserve_current_view=self._should_preserve_pixel_view())

    def set_map_file_paths(self, map_files: dict[str, str]) -> None:
        if self._state is None:
            return
        self._state.map_files = dict(map_files or {})
        self._map_summary_cache.clear()
        self._invalidate_map_caches()
        self._refresh_map_info()
        self._refresh_plot()
        self._start_background_cache_build()

    def set_map_source_mode(self, mode: str) -> None:
        if self._state is None:
            return
        mode = str(mode or "auto").lower()
        if mode not in {"auto", "filesystem", "embedded"}:
            mode = "auto"
        self._state.map_source_mode = mode
        self._map_summary_cache.clear()
        self._invalidate_map_caches()
        self._refresh_status_text()
        self._refresh_map_info()
        self._refresh_plot(preserve_current_view=self._should_preserve_pixel_view())
        self._start_background_cache_build()

    def set_geometry_edit_enabled(self, enabled: bool) -> None:
        self._geometry_edit_enabled = bool(enabled)
        self._refresh_control_mode_ui()
        self._refresh_status_text()

    def set_entry_box_path(self, entry_box_path: Optional[str | Path]) -> None:
        self._entry_box_path = Path(entry_box_path).expanduser().resolve() if entry_box_path else None
        self._committed_line_seeds = None
        if self._entry_box_path is not None and self._entry_box_path.suffix.lower() == ".h5":
            try:
                box_data = read_b3d_h5(str(self._entry_box_path))
                line_seeds = box_data.get("line_seeds")
                if isinstance(line_seeds, dict):
                    self._committed_line_seeds = copy.deepcopy(line_seeds)
            except Exception:
                self._committed_line_seeds = None
        self._refresh_open_3d_state()
        self._emit_action_state()
        self._refresh_fieldlines_from_committed_seeds()

    def set_geometry_selection(self, selection: BoxGeometrySelection) -> None:
        if self._state is None:
            return
        self._state.geometry = selection
        self._map_summary_cache.clear()
        self._invalidate_map_caches()
        self._refresh_status_text()
        self._refresh_map_info()
        self._refresh_plot(preserve_current_view=self._should_preserve_pixel_view())
        self._start_background_cache_build()
        if self._geometry_change_callback is not None:
            self._geometry_change_callback(selection)

    def set_fov_selection(self, selection: DisplayFovSelection) -> None:
        if self._state is None:
            return
        if self._state.square_fov:
            selection = DisplayFovSelection(
                center_x_arcsec=selection.center_x_arcsec,
                center_y_arcsec=selection.center_y_arcsec,
                width_arcsec=selection.width_arcsec,
                height_arcsec=selection.width_arcsec,
            )
        self._state.fov = selection
        self._sync_fov_box_to_selection()
        self._refresh_status_text()
        self._refresh_plot(preserve_current_view=self._should_preserve_pixel_view())
        if self._fov_change_callback is not None:
            self._fov_change_callback(selection)

    def set_square_fov(self, enabled: bool) -> None:
        if self._state is None:
            return
        self._state.square_fov = bool(enabled)
        self._update_fov_control_enabled_state()
        if enabled and self._state.fov is not None:
            self.set_fov_selection(
                DisplayFovSelection(
                    center_x_arcsec=self._state.fov.center_x_arcsec,
                    center_y_arcsec=self._state.fov.center_y_arcsec,
                    width_arcsec=self._state.fov.width_arcsec,
                    height_arcsec=self._state.fov.width_arcsec,
                )
            )

    def _update_fov_control_enabled_state(self) -> None:
        self._refresh_control_mode_ui()

    def set_action_state_callback(self, callback) -> None:
        self._action_state_callback = callback
        self._emit_action_state()

    def _emit_action_state(self) -> None:
        if self._action_state_callback is not None:
            self._action_state_callback(self._can_open_3d, self._can_clear_lines)

    def _refresh_open_3d_state(self) -> None:
        self._can_open_3d = (
            self._viewer3d is None
            and self._entry_box_path is not None
            and can_prepare_model_for_viewer(self._entry_box_path)
        )

    def _on_viewer3d_closed(self, *_args) -> None:
        close_was_handled = self._viewer3d_close_handled
        if not close_was_handled and self._viewer3d is not None:
            try:
                self.commit_live_3d_edits(
                    self._viewer3d._collect_line_seeds_snapshot(),
                    self._viewer3d._collect_streamlines(),
                    z_base=self._viewer3d.grid_zbase,
                )
                close_was_handled = True
            except Exception:
                pass
        self._viewer3d = None
        self._viewer3d_temp_h5_path = None
        self._viewer3d_watchdog.stop()
        if self._hidden_for_live_3d and not self._viewer3d_close_handled:
            host = self.window()
            host.show()
            host.raise_()
            host.activateWindow()
            self._hidden_for_live_3d = False
        self._viewer3d_close_handled = False
        self._refresh_open_3d_state()
        self._emit_action_state()
        if not close_was_handled:
            self._set_runtime_status("Live 3D viewer closed.")

    def committed_line_seeds(self):
        return copy.deepcopy(self._committed_line_seeds) if isinstance(self._committed_line_seeds, dict) else None

    def _refresh_fieldlines_from_committed_seeds(self) -> None:
        if self._current_map is None or self._current_axes is None:
            return
        if self._entry_box_path is None:
            self.clear_fieldlines()
            return
        if not isinstance(self._committed_line_seeds, dict):
            self.clear_fieldlines()
            return
        try:
            box, _obs_time, b3dtype, _temp_h5_path = prepare_model_for_viewer(self._entry_box_path)
            box.b3d["line_seeds"] = copy.deepcopy(self._committed_line_seeds)
            self._fieldline_frame_hcc = getattr(getattr(box, "_center", None), "frame", None)
            self._fieldline_frame_obs = getattr(box, "_frame_obs", None)
            streamlines, z_base = generate_streamlines_from_line_seeds(box, b3dtype, self._committed_line_seeds)
            if streamlines:
                self.plot_fieldlines(streamlines, z_base=z_base)
            else:
                self.clear_fieldlines()
        except Exception as exc:
            self._set_runtime_status(f"Failed to restore saved field lines: {exc}")

    def commit_live_3d_edits(self, line_seeds, streamlines, z_base=0.0) -> None:
        self._committed_line_seeds = copy.deepcopy(line_seeds) if isinstance(line_seeds, dict) else None
        self._viewer3d_close_handled = True
        if self._hidden_for_live_3d:
            host = self.window()
            host.show()
            host.raise_()
            host.activateWindow()
            self._hidden_for_live_3d = False
        self.plot_fieldlines(streamlines, z_base=z_base)
        self._set_runtime_status("Accepted 3D seed edits into the 2D session model.")

    def cancel_live_3d_edits(self) -> None:
        self._viewer3d_close_handled = True
        if self._hidden_for_live_3d:
            host = self.window()
            host.show()
            host.raise_()
            host.activateWindow()
            self._hidden_for_live_3d = False
        self._set_runtime_status("Canceled 3D seed edits; kept the 2D session model unchanged.")

    def _check_viewer3d_state(self) -> None:
        if self._viewer3d is None:
            self._viewer3d_watchdog.stop()
            return
        try:
            window = self._viewer3d.app_window if hasattr(self._viewer3d, "app_window") else self._viewer3d
            if not window.isVisible():
                self._on_viewer3d_closed()
        except Exception:
            self._on_viewer3d_closed()

    def _control_target_mode(self) -> str:
        return "box" if self._geometry_edit_enabled else "fov"

    def _refresh_control_mode_ui(self) -> None:
        mode = self._control_target_mode()
        if mode == "box":
            self._control_mode_label.setText("BOX Controls")
            self._left_btn.setToolTip("Move box center left")
            self._right_btn.setToolTip("Move box center right")
            self._down_btn.setToolTip("Move box center down")
            self._up_btn.setToolTip("Move box center up")
            self._x_minus_btn.setToolTip("Decrease X box size")
            self._x_plus_btn.setToolTip("Increase X box size")
            self._y_minus_btn.setToolTip("Decrease Y box size")
            self._y_plus_btn.setToolTip("Increase Y box size")
            self._xy_minus_btn.setToolTip("Decrease X and Y box size together")
            self._xy_plus_btn.setToolTip("Increase X and Y box size together")
            self._recompute_fov_btn.setEnabled(False)
            self._y_minus_btn.setEnabled(True)
            self._y_plus_btn.setEnabled(True)
            self._xy_minus_btn.setEnabled(True)
            self._xy_plus_btn.setEnabled(True)
        else:
            square = bool(self._state.square_fov) if self._state is not None else False
            self._control_mode_label.setText("FOV Controls")
            self._left_btn.setToolTip("Move FOV center left")
            self._right_btn.setToolTip("Move FOV center right")
            self._down_btn.setToolTip("Move FOV center down")
            self._up_btn.setToolTip("Move FOV center up")
            self._x_minus_btn.setToolTip("Decrease FOV X size")
            self._x_plus_btn.setToolTip("Increase FOV X size")
            self._y_minus_btn.setToolTip("Decrease FOV Y size")
            self._y_plus_btn.setToolTip("Increase FOV Y size")
            self._xy_minus_btn.setToolTip("Decrease FOV X and Y size together")
            self._xy_plus_btn.setToolTip("Increase FOV X and Y size together")
            self._recompute_fov_btn.setEnabled(self._projected_box_fov is not None)
            self._y_minus_btn.setEnabled(not square)
            self._y_plus_btn.setEnabled(not square)
            self._xy_minus_btn.setEnabled(True)
            self._xy_plus_btn.setEnabled(True)

    def _nudge_primary_center(self, axis: str, sign: int) -> None:
        if self._control_target_mode() == "box":
            self._nudge_box_center(axis, sign)
        else:
            self._nudge_fov_center(axis, sign)

    def _nudge_primary_size(self, axis: str, sign: int) -> None:
        if self._control_target_mode() == "box":
            self._nudge_box_size(axis, sign)
        else:
            self._nudge_fov_size(axis, sign)

    def _nudge_primary_size_xy(self, sign: int) -> None:
        if self._control_target_mode() == "box":
            self._nudge_box_size_xy(sign)
        else:
            self._nudge_fov_size_xy(sign)

    def current_geometry_selection(self) -> Optional[BoxGeometrySelection]:
        if self._state is None:
            return None
        return self._state.geometry

    def current_fov_selection(self) -> Optional[DisplayFovSelection]:
        if self._state is None:
            return None
        return self._state.fov

    def current_fov_box_selection(self) -> Optional[DisplayFovBoxSelection]:
        if self._state is None:
            return None
        return self._state.fov_box

    def projected_box_fov(self) -> Optional[DisplayFovSelection]:
        return self._projected_box_fov

    def set_geometry_change_callback(self, callback) -> None:
        self._geometry_change_callback = callback

    def set_map_info_callback(self, callback) -> None:
        self._map_info_callback = callback
        if callback is not None:
            callback(self._last_map_info_text)

    def set_status_callback(self, callback) -> None:
        self._status_callback = callback
        if callback is not None:
            callback(self._last_status_text)

    def set_fov_change_callback(self, callback) -> None:
        self._fov_change_callback = callback
        if callback is not None and self._state is not None and self._state.fov is not None:
            callback(self._state.fov)

    def state(self) -> Optional[MapBoxViewState]:
        return self._state

    def _sync_fov_box_to_selection(self) -> None:
        if self._state is None or self._state.fov is None or self._state.fov_box is None:
            return
        self._state.fov_box = DisplayFovBoxSelection(
            center_x_arcsec=float(self._state.fov.center_x_arcsec),
            center_y_arcsec=float(self._state.fov.center_y_arcsec),
            width_arcsec=float(self._state.fov.width_arcsec),
            height_arcsec=float(self._state.fov.height_arcsec),
            z_min_mm=float(self._state.fov_box.z_min_mm),
            z_max_mm=float(self._state.fov_box.z_max_mm),
        )

    def _should_preserve_pixel_view(self) -> bool:
        return self._view_mode == "full_sun"

    def _refresh_status_text(self) -> None:
        if self._state is None:
            self._last_status_text = "Map/box display placeholder (uninitialized)"
            if self._status_callback is not None:
                self._status_callback(self._last_status_text)
            return
        geom = self._state.geometry
        if geom is None:
            geom_text = "geometry: <none>"
        else:
            geom_text = (
                f"geometry: {geom.coord_mode.value} "
                f"({geom.coord_x:.3f}, {geom.coord_y:.3f}), "
                f"dims={geom.grid_x}x{geom.grid_y}x{geom.grid_z}, dx={geom.dx_km:.3f} km"
            )
        if self._state.fov is None:
            fov_text = "fov: <auto>"
        else:
            fov = self._state.fov
            fov_text = (
                f"fov: center=({fov.center_x_arcsec:.2f}, {fov.center_y_arcsec:.2f}) arcsec, "
                f"size={fov.width_arcsec:.2f}x{fov.height_arcsec:.2f} arcsec"
            )
        self._last_status_text = (
            "Map/box selector interaction\n"
            f"mouse_actions={'on' if self._mouse_actions_enabled else 'off'}\n"
            f"geometry_edit={'on' if self._geometry_edit_enabled else 'off'}\n"
            f"context={self._display_map_label(self._state.selected_context_id, bottom=False)!r}, "
            f"base={self._display_map_label(self._state.selected_bottom_id, bottom=True)!r}\n"
            f"map_source={self._state.map_source_mode}\n"
            f"square_fov={'on' if self._state.square_fov else 'off'}\n"
            f"{geom_text}\n{fov_text}"
        )
        if self._status_callback is not None:
            self._status_callback(self._last_status_text)

    def _refresh_map_info(self) -> None:
        if self._state is None:
            self._set_map_info_text("Map info: <uninitialized>")
            return
        context_summary = self._single_map_summary(self._state.selected_context_id, role="Context", bottom=False)
        bottom_summary = self._single_map_summary(self._state.selected_bottom_id, role="Base", bottom=True)
        self._set_map_info_text(f"{context_summary}\n\n{bottom_summary}")

    def _set_map_info_text(self, text: str) -> None:
        self._last_map_info_text = text
        if self._map_info_callback is not None:
            self._map_info_callback(text)

    def _single_map_summary(self, map_id: Optional[str], role: str, bottom: bool) -> str:
        if not map_id:
            return f"{role} map: <none>"
        cache_key = f"{role}:{map_id}"
        if cache_key in self._map_summary_cache:
            return self._map_summary_cache[cache_key]
        try:
            smap = self._selected_bottom_map() if bottom else self._selected_context_map()
            if smap is None:
                txt = f"{role} map ({self._display_map_label(map_id, bottom)}): unavailable"
                self._map_summary_cache[cache_key] = txt
                return txt
            data = np.asarray(smap.data)
            finite = np.isfinite(data)
            n_finite = int(finite.sum())
            stats = "all non-finite"
            if n_finite > 0:
                vals = data[finite]
                stats = (
                    f"min={float(np.nanmin(vals)):.3g}, "
                    f"max={float(np.nanmax(vals)):.3g}, "
                    f"mean={float(np.nanmean(vals)):.3g}"
                )
            obs_time = getattr(smap, "date", None)
            txt = (
                f"{role} map ({self._display_map_label(map_id, bottom)})\n"
                f"source={self._map_source_label(map_id)}\n"
                f"shape={tuple(data.shape)}, finite={n_finite}/{data.size}\n"
                f"{stats}\n"
                f"obs_time={obs_time}"
            )
        except Exception as exc:
            txt = f"{role} map ({map_id}) load failed:\n{self._map_source_label(map_id)}\n{exc}"
        self._map_summary_cache[cache_key] = txt
        return txt

    @staticmethod
    def _display_map_label(map_id: Optional[str], bottom: bool) -> Optional[str]:
        if map_id is None:
            return None
        if not bottom and map_id == "Bz":
            return "Blos"
        return map_id

    def _selected_context_map(self):
        if self._state is None:
            return None
        map_id = self._state.selected_context_id
        if not map_id:
            return None
        return self._map_for_id(map_id, purpose="context")

    def _selected_bottom_map(self):
        if self._state is None:
            return None
        map_id = self._state.selected_bottom_id
        if not map_id:
            return None
        return self._map_for_id(map_id, purpose="bottom")

    def _map_for_id(self, map_id: str, purpose: str):
        canonical_key = self._canonical_map_key(map_id)
        display_key = f"__{purpose}__:{canonical_key}"
        alias_key = f"__{purpose}__:{map_id}"
        with self._cache_lock:
            if alias_key in self._loaded_map_cache:
                return self._loaded_map_cache[alias_key]
            if display_key in self._loaded_map_cache:
                smap = self._loaded_map_cache[display_key]
                self._loaded_map_cache[alias_key] = smap
                return smap

        if canonical_key in {"br", "bp", "bt"}:
            smap = self._load_hmi_vector_product(canonical_key)
        else:
            smap = self._load_raw_map(canonical_key)
        if smap is None:
            return None
        if purpose == "context":
            smap = self._prepare_context_map(canonical_key, smap)
        else:
            smap = self._prepare_bottom_map(canonical_key, smap)
        with self._cache_lock:
            self._loaded_map_cache[display_key] = smap
            self._loaded_map_cache[alias_key] = smap
        return smap

    def _invalidate_map_caches(self) -> None:
        with self._cache_lock:
            self._loaded_map_cache.clear()
            self._raw_map_cache.clear()
            self._background_cache_generation += 1

    def _start_background_cache_build(self) -> None:
        if self._state is None:
            return
        map_ids = tuple(m for m in (self._state.session_input.map_ids or ()) if m in {"Bz", "Ic", "Br", "Bp", "Bt"})
        if not map_ids:
            return
        with self._cache_lock:
            generation = self._background_cache_generation
            thread = self._background_cache_thread
            if thread is not None and thread.is_alive():
                return
            self._background_cache_thread = threading.Thread(
                target=self._background_cache_worker,
                args=(generation, map_ids),
                daemon=True,
            )
            self._background_cache_thread.start()

    def _background_cache_worker(self, generation: int, map_ids: tuple[str, ...]) -> None:
        for map_id in map_ids:
            with self._cache_lock:
                if generation != self._background_cache_generation:
                    return
            try:
                self._map_for_id(map_id, purpose="context")
                self._map_for_id(map_id, purpose="bottom")
            except Exception:
                continue

    @staticmethod
    def _canonical_map_key(map_id: str) -> str:
        return _DISPLAY_MAP_ALIASES.get(map_id, map_id)

    def _map_source_label(self, map_id: str) -> str:
        canonical_key = self._canonical_map_key(map_id)
        if canonical_key in {"br", "bp", "bt"}:
            if self._embedded_base_key_for_map(canonical_key) and self._embedded_base_array(canonical_key) is not None:
                return f"embedded:base.{self._embedded_base_key_for_map(canonical_key)}"
            return f"derived from {', '.join(_HMI_VECTOR_SEGMENTS[:-1])} + disambig"
        path = self._filesystem_path_for_key(canonical_key)
        if path:
            return Path(path).name
        if self._embedded_base_key_for_map(canonical_key) and self._embedded_base_array(canonical_key) is not None:
            return f"embedded:base.{self._embedded_base_key_for_map(canonical_key)}"
        ref_key = self._embedded_refmap_key(canonical_key)
        if ref_key and self._embedded_payload_for_key(ref_key):
            return f"embedded:{ref_key}"
        return canonical_key

    def _filesystem_enabled(self) -> bool:
        return self._state is not None and self._state.map_source_mode in {"auto", "filesystem"}

    def _embedded_enabled(self) -> bool:
        return self._state is not None and self._state.map_source_mode in {"auto", "embedded"}

    def _filesystem_path_for_key(self, map_key: str) -> str | None:
        if not self._filesystem_enabled():
            return None
        return (self._state.map_files or {}).get(map_key) if self._state is not None else None

    def _embedded_payload_for_key(self, ref_key: str):
        if not self._embedded_enabled() or self._state is None:
            return None
        return (self._state.refmaps or {}).get(ref_key)

    @staticmethod
    def _embedded_base_key_for_map(map_key: str) -> str | None:
        if map_key in {"bx", "by", "bz"}:
            return map_key
        if map_key in {"magnetogram", "br"}:
            return "bz"
        if map_key == "bp":
            return "bx"
        if map_key == "bt":
            return "by"
        if map_key == "continuum":
            return "ic"
        if map_key == "chromo_mask":
            return "chromo_mask"
        return None

    def _embedded_base_array(self, map_key: str):
        if not self._embedded_enabled() or self._state is None:
            return None
        base_key = self._embedded_base_key_for_map(map_key)
        if not base_key:
            return None
        base_maps = self._state.base_maps or {}
        if base_key not in base_maps:
            return None
        arr = np.asarray(base_maps[base_key])
        if arr.ndim != 2:
            return None
        if map_key == "bt" and base_key == "by":
            arr = -arr
        return arr

    def _load_embedded_base_map(self, map_key: str):
        if self._state is None:
            return None
        with self._cache_lock:
            cache_key = f"__base__:{map_key}"
            if cache_key in self._raw_map_cache:
                return self._raw_map_cache[cache_key]
        data = self._embedded_base_array(map_key)
        if data is None:
            return None
        ref_map = self._reference_context_map()
        if ref_map is None:
            return None
        base_geom = self._state.base_geometry or self._state.geometry
        box = self._build_legacy_box(ref_map, geom=base_geom)
        if box is None:
            return None
        try:
            smap = Map(np.asarray(data), box.bottom_cea_header)
        except Exception:
            return None
        with self._cache_lock:
            self._raw_map_cache[cache_key] = smap
        return smap

    @staticmethod
    def _header_text_from_value(value) -> str:
        if value is None:
            return ""
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8", "ignore")
        if isinstance(value, np.ndarray) and value.shape == ():
            return MapBoxDisplayWidget._header_text_from_value(value.item())
        return str(value)

    @staticmethod
    def _normalize_embedded_header_text(header_text: str) -> str:
        text = str(header_text or "")
        # Embedded box files may persist FITS headers with literal "\\n"
        # separators instead of real newlines.
        if "\\n" in text and "\n" not in text:
            text = text.replace("\\n", "\n")
        return text

    @staticmethod
    def _embedded_refmap_key(map_key: str) -> str | None:
        if map_key == "magnetogram":
            return "Bz_reference"
        if map_key == "continuum":
            return "Ic_reference"
        if map_key == "Vert_current":
            return "Vert_current"
        if map_key.isdigit():
            return f"AIA_{map_key}"
        return None

    def _load_embedded_refmap(self, ref_key: str):
        if self._state is None:
            return None
        with self._cache_lock:
            cache_key = f"__embedded__:{ref_key}"
            if cache_key in self._raw_map_cache:
                return self._raw_map_cache[cache_key]
        payload = self._embedded_payload_for_key(ref_key)
        if not isinstance(payload, dict):
            return None
        data = payload.get("data")
        header_text = self._normalize_embedded_header_text(
            self._header_text_from_value(payload.get("wcs_header"))
        )
        if data is None or not header_text.strip():
            return None
        try:
            header = fits.Header.fromstring(header_text, sep="\n")
            header[_EMBEDDED_REFMAP_FLAG] = True
            smap = Map(np.asarray(data), header)
        except Exception:
            return None
        with self._cache_lock:
            self._raw_map_cache[cache_key] = smap
        return smap

    def _load_raw_map(self, map_key: str):
        with self._cache_lock:
            if map_key in self._raw_map_cache:
                return self._raw_map_cache[map_key]
        path = self._filesystem_path_for_key(map_key)
        if path:
            smap = load_sunpy_map_compat(path)
            if map_key in _HMI_VECTOR_SEGMENTS:
                smap = self._submap_to_geometry_fov(smap)
        else:
            smap = self._load_embedded_base_map(map_key)
            if smap is not None:
                with self._cache_lock:
                    self._raw_map_cache[map_key] = smap
                return smap
            ref_key = self._embedded_refmap_key(map_key)
            smap = self._load_embedded_refmap(ref_key) if ref_key else None
            if smap is None:
                return None
        with self._cache_lock:
            self._raw_map_cache[map_key] = smap
        return smap

    def _load_hmi_vector_segment(self, map_key: str):
        with self._cache_lock:
            if map_key in self._raw_map_cache:
                return self._raw_map_cache[map_key]
        base_map = self._load_raw_map(map_key)
        if base_map is None:
            return None
        if map_key == "azimuth":
            disambig_map = self._load_raw_map("disambig")
            if disambig_map is None:
                return None
            base_map = hmi_disambig(base_map, disambig_map)
        with self._cache_lock:
            self._raw_map_cache[map_key] = base_map
        return base_map

    def _load_hmi_vector_product(self, map_key: str):
        raw_key = f"__raw__:{map_key}"
        with self._cache_lock:
            if raw_key in self._raw_map_cache:
                return self._raw_map_cache[raw_key]
        field_map = self._load_hmi_vector_segment("field")
        inclination_map = self._load_hmi_vector_segment("inclination")
        azimuth_map = self._load_hmi_vector_segment("azimuth")
        if field_map is None or inclination_map is None or azimuth_map is None:
            return None
        map_bp, map_bt, map_br = hmi_b2ptr(field_map, inclination_map, azimuth_map)
        with self._cache_lock:
            self._raw_map_cache["__raw__:bp"] = map_bp
            self._raw_map_cache["__raw__:bt"] = map_bt
            self._raw_map_cache["__raw__:br"] = map_br
            return self._raw_map_cache.get(raw_key)

    def _prepare_context_map(self, map_key: str, smap):
        display_map = smap
        if map_key in _HMI_DISPLAY_KEYS:
            try:
                display_map = display_map.rotate(order=3)
            except Exception:
                pass
            ref_map = self._reference_context_map()
            if ref_map is not None:
                try:
                    display_map = self._with_matching_rsun(display_map, ref_map)
                    display_map = display_map.reproject_to(ref_map.wcs)
                except Exception:
                    pass
        self._apply_display_scaling(display_map, map_key)
        return display_map

    def _prepare_bottom_map(self, map_key: str, smap):
        display_map = smap
        box = self._build_legacy_box(display_map)
        if box is not None:
            display_map = self._submap_to_box_bounds(display_map, box)
            try:
                display_map = self._with_matching_rsun(display_map, box.bottom_cea_header)
                display_map = display_map.reproject_to(
                    box.bottom_cea_header,
                    algorithm="adaptive",
                    roundtrip_coords=False,
                )
            except Exception:
                pass
        self._apply_display_scaling(display_map, map_key)
        return display_map

    def _reference_context_map(self):
        if self._state is None:
            return None
        for map_id in _AIA_REFERENCE_IDS:
            path = self._filesystem_path_for_key(map_id)
            cache_key = f"__ref__:{map_id}"
            with self._cache_lock:
                if cache_key in self._raw_map_cache:
                    return self._raw_map_cache[cache_key]
            try:
                if path:
                    ref_map = load_sunpy_map_compat(path)
                else:
                    ref_map = self._load_embedded_refmap(f"AIA_{map_id}")
                    if ref_map is None:
                        continue
                with self._cache_lock:
                    self._raw_map_cache[cache_key] = ref_map
                return ref_map
            except Exception:
                continue
        return None

    def _geometry_anchor_coord(self, geom: BoxGeometrySelection, smap):
        obstime = getattr(smap, "date", None)
        observer = getattr(smap, "observer_coordinate", None)
        if observer is None:
            observer = "earth"
        if geom.coord_mode == CoordMode.HPC:
            return SkyCoord(
                Tx=geom.coord_x * u.arcsec,
                Ty=geom.coord_y * u.arcsec,
                obstime=obstime,
                observer=observer,
                frame=Helioprojective,
            )
        if geom.coord_mode == CoordMode.HGC:
            return SkyCoord(
                lon=geom.coord_x * u.deg,
                lat=geom.coord_y * u.deg,
                radius=696 * u.Mm,
                obstime=obstime,
                observer=observer,
                frame=HeliographicCarrington,
            )
        return SkyCoord(
            lon=geom.coord_x * u.deg,
            lat=geom.coord_y * u.deg,
            radius=696 * u.Mm,
            obstime=obstime,
            observer=observer,
            frame=HeliographicStonyhurst,
        )

    def _build_legacy_box(self, smap, geom: BoxGeometrySelection | None = None):
        if self._state is None:
            return None
        geom = geom or self._state.geometry
        if geom is None:
            return None
        box_dims = u.Quantity([geom.grid_x, geom.grid_y, geom.grid_z], u.pix)
        box_res = geom.dx_km * u.km
        box_origin = self._geometry_anchor_coord(geom, smap)
        observer = getattr(smap, "observer_coordinate", None)
        if observer is None:
            observer = "earth"
        obstime = getattr(smap, "date", None)
        frame_obs = Helioprojective(observer=observer, obstime=obstime)
        frame_hcc = Heliocentric(observer=box_origin, obstime=obstime)
        box_dimensions = box_dims / u.pix * box_res
        box_center = box_origin.transform_to(frame_hcc)
        box_center = SkyCoord(
            x=box_center.x,
            y=box_center.y,
            z=box_center.z + box_dimensions[2] / 2,
            frame=box_center.frame,
        )
        return Box(frame_obs, box_origin, box_center, box_dims, box_res)

    def _submap_to_box_bounds(self, smap, box, pad_frac: float | None = None):
        if box is None:
            return smap
        if pad_frac is None:
            pad_frac = float(self._state.session_input.pad_frac or 0.10) if self._state is not None else 0.10
        try:
            fov = box.bounds_coords_bl_tr(pad_frac=pad_frac)
            return smap.submap(fov[0], top_right=fov[1])
        except Exception:
            return smap

    def _submap_to_geometry_fov(self, smap):
        return self._submap_to_box_bounds(smap, self._build_legacy_box(smap))

    def _submap_to_explicit_fov(self, smap, pad_factor: float = 1.10):
        fov = self._state.fov if (self._state and self._state.fov) else None
        if fov is None:
            box = self._build_legacy_box(smap)
            if box is None:
                return smap
            fov = self._box_bounds_to_fov_selection(box, smap)
        half_w = 0.5 * max(float(fov.width_arcsec), 1e-3) * float(pad_factor)
        half_h = 0.5 * max(float(fov.height_arcsec), 1e-3) * float(pad_factor)
        observer = getattr(smap, "observer_coordinate", None) or "earth"
        obstime = getattr(smap, "date", None)
        bottom_left = SkyCoord(
            Tx=(float(fov.center_x_arcsec) - half_w) * u.arcsec,
            Ty=(float(fov.center_y_arcsec) - half_h) * u.arcsec,
            frame=Helioprojective(observer=observer, obstime=obstime),
        )
        top_right = SkyCoord(
            Tx=(float(fov.center_x_arcsec) + half_w) * u.arcsec,
            Ty=(float(fov.center_y_arcsec) + half_h) * u.arcsec,
            frame=Helioprojective(observer=observer, obstime=obstime),
        )
        try:
            return smap.submap(bottom_left, top_right=top_right)
        except Exception:
            return smap

    @staticmethod
    def _target_rsun_meters(target) -> float | None:
        try:
            if hasattr(target, "rsun_meters"):
                return float(target.rsun_meters.to_value(u.m))
        except Exception:
            pass
        try:
            if isinstance(target, dict):
                value = target.get("rsun_ref")
                if value is not None:
                    return float(value)
        except Exception:
            pass
        return None

    @staticmethod
    def _with_matching_rsun(smap, target):
        target_rsun_m = MapBoxDisplayWidget._target_rsun_meters(target)
        if not target_rsun_m or target_rsun_m <= 0:
            return smap
        try:
            current_rsun_m = float(smap.rsun_meters.to_value(u.m))
        except Exception:
            current_rsun_m = None
        if current_rsun_m is not None and abs(current_rsun_m - target_rsun_m) < 1e-3:
            return smap
        try:
            meta = smap.meta.copy()
            meta["rsun_ref"] = target_rsun_m
            return smap._new_instance(smap.data, meta, plot_settings=smap.plot_settings)
        except Exception:
            return smap

    @staticmethod
    def _apply_display_scaling(smap, map_key: str) -> None:
        data = np.asarray(smap.data)
        finite = np.isfinite(data)
        if not finite.any():
            return
        vals = data[finite]
        try:
            if map_key in _AIA_COLOR_KEYS:
                if bool(getattr(smap, "meta", {}).get(_EMBEDDED_REFMAP_FLAG, False)):
                    cmap = sunpy_colormaps.cm.cmlist.get(f"sdoaia{map_key}")
                    if cmap is not None:
                        smap.plot_settings["cmap"] = cmap
                    lo = float(np.nanpercentile(vals, 0.5))
                    hi = float(np.nanpercentile(vals, 99.8))
                    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                        smap.plot_settings["norm"] = mcolors.Normalize(vmin=lo, vmax=hi)
            if map_key in _SIGNED_MAGNETIC_KEYS:
                if map_key in _TRANSVERSE_MAGNETIC_KEYS:
                    pct = 92.5
                elif map_key == "br":
                    pct = 97.5
                else:
                    pct = 99.0
                hi = float(np.nanpercentile(np.abs(vals), pct))
                if hi > 0:
                    smap.plot_settings["cmap"] = "gray"
                    smap.plot_settings["norm"] = mcolors.TwoSlopeNorm(vmin=-hi, vcenter=0.0, vmax=hi)
            elif map_key in _VERT_CURRENT_KEYS:
                hi = float(np.nanpercentile(np.abs(vals), 99.0))
                if hi > 0:
                    smap.plot_settings["cmap"] = "RdBu_r"
                    smap.plot_settings["norm"] = mcolors.TwoSlopeNorm(vmin=-hi, vcenter=0.0, vmax=hi)
            elif map_key in _CHROMO_MASK_KEYS:
                cmap = mcolors.ListedColormap([
                    "#000000", "#1f77b4", "#ff7f0e", "#2ca02c",
                    "#d62728", "#9467bd", "#8c564b", "#e377c2",
                    "#7f7f7f",
                ])
                smap.plot_settings["cmap"] = cmap
                smap.plot_settings["norm"] = mcolors.BoundaryNorm(np.arange(-0.5, 9.5, 1.0), cmap.N)
            elif map_key == "continuum":
                lo = float(np.nanpercentile(vals, 1.0))
                hi = float(np.nanpercentile(vals, 99.5))
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    smap.plot_settings["norm"] = mcolors.Normalize(vmin=lo, vmax=hi)
                    smap.plot_settings["cmap"] = "gray"
        except Exception:
            # Display scaling should not break rendering.
            pass

    def _refresh_plot(self, preserve_current_view: bool = False) -> None:
        prev_xlim = prev_ylim = None
        if preserve_current_view and self._current_axes is not None:
            try:
                prev_xlim = self._current_axes.get_xlim()
                prev_ylim = self._current_axes.get_ylim()
            except Exception:
                prev_xlim = prev_ylim = None
        self._fig.clear()
        self._current_map = None
        self._current_axes = None
        self._overlay_rect = None
        self._overlay_bbox_rect = None
        self._projected_box_bbox_rect = None
        self._projected_box_fov = None
        self._overlay_center_artist = None
        self._overlay_corner_artists = []
        self._full_view_limits = None
        smap = overlay_map = None
        try:
            smap = self._selected_context_map()
            overlay_map = self._selected_bottom_map()
        except Exception as exc:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Map load failed:\n{exc}", ha="center", va="center")
            ax.axis("off")
            self._canvas.draw_idle()
            return

        if smap is None:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, "No local map available for selected map ID", ha="center", va="center")
            ax.axis("off")
            self._fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))
            self._canvas.draw_idle()
            return

        if not preserve_current_view and self._view_mode == "box_fov":
            try:
                smap = self._submap_to_explicit_fov(smap, pad_factor=1.10)
            except Exception:
                pass

        try:
            ax = self._fig.add_subplot(111, projection=smap)
            self._current_map = smap
            self._current_axes = ax
            ax.set_facecolor("black")
            try:
                ax.set_box_aspect(1)
            except Exception:
                pass
            try:
                ax.set_aspect("equal", adjustable="box")
            except Exception:
                pass
            try:
                smap.plot(axes=ax, annotate=False)
            except TypeError:
                smap.plot(axes=ax)
            try:
                ax.set_title("")
            except Exception:
                pass
            if overlay_map is not None:
                try:
                    overlay_map.plot(axes=ax, autoalign=True, alpha=0.90)
                except Exception:
                    pass
            # These often improve readability and mimic the legacy gxbox style.
            try:
                smap.draw_grid(axes=ax, color="w", lw=0.5)
            except Exception:
                pass
            try:
                smap.draw_limb(axes=ax, color="w", lw=0.8)
            except Exception:
                pass
            self._plot_box_outline(ax, smap)
            try:
                ax.set_title("")
            except Exception:
                pass
            title = (
                f"Context={self._display_map_label(self._state.selected_context_id, bottom=False)} | "
                f"Base={self._display_map_label(self._state.selected_bottom_id, bottom=True)} @ {getattr(smap, 'date', '')}"
            )
            self._fig.text(0.5, 0.992, title, ha="center", va="top", fontsize=10)
            self._full_view_limits = (ax.get_xlim(), ax.get_ylim())
            if preserve_current_view and prev_xlim is not None and prev_ylim is not None:
                self._restore_preserved_view(prev_xlim, prev_ylim)
            elif self._view_mode == "box_fov" and self._state is not None and self._state.fov is not None:
                self._set_view_window_hpc(
                    center_x_arcsec=float(self._state.fov.center_x_arcsec),
                    center_y_arcsec=float(self._state.fov.center_y_arcsec),
                    width_arcsec=max(float(self._state.fov.width_arcsec) * 1.10, 1e-3),
                    height_arcsec=max(float(self._state.fov.height_arcsec) * 1.10, 1e-3),
                )
        except Exception as exc:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Plot failed:\n{exc}", ha="center", va="center")
            ax.axis("off")

        self._fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        self._render_fieldlines()
        self._canvas.draw_idle()
        self._update_cursor_for_mode()

    def open_live_3d_viewer(self) -> None:
        self._check_viewer3d_state()
        if self._viewer3d is not None:
            try:
                self._viewer3d.show()
                if hasattr(self._viewer3d, "app_window"):
                    self._viewer3d.app_window.showNormal()
                    self._viewer3d.app_window.raise_()
                    self._viewer3d.app_window.activateWindow()
                return
            except Exception:
                self._viewer3d = None
                self._viewer3d_temp_h5_path = None
                self._refresh_open_3d_state()
                self._emit_action_state()
        if self._entry_box_path is None:
            self._set_runtime_status("3D viewer unavailable: no entry box is attached to this selector.")
            return
        try:
            box, obs_time, b3dtype, temp_h5_path = prepare_model_for_viewer(self._entry_box_path)
            if isinstance(self._committed_line_seeds, dict):
                box.b3d["line_seeds"] = copy.deepcopy(self._committed_line_seeds)
            else:
                box.b3d.pop("line_seeds", None)
            box_norm_direction, box_view_up = _viewer_camera_vectors(box, obs_time)
            self._fieldline_frame_hcc = getattr(getattr(box, "_center", None), "frame", None)
            self._fieldline_frame_obs = getattr(box, "_frame_obs", None)
            self._viewer3d_close_handled = False
            self._viewer3d = MagFieldViewer(
                box,
                time=obs_time,
                b3dtype=b3dtype,
                parent=self,
                box_norm_direction=box_norm_direction,
                box_view_up=box_view_up,
                session_mode="embedded",
            )
            self._viewer3d_temp_h5_path = temp_h5_path
            if hasattr(self._viewer3d, "app_window"):
                self._viewer3d.app_window.setWindowTitle(f"GxBox 3D viewer - {self._entry_box_path.name}")
                self._viewer3d.app_window.destroyed.connect(self._on_viewer3d_closed)
            else:
                self._viewer3d.destroyed.connect(self._on_viewer3d_closed)
            self._refresh_open_3d_state()
            self._emit_action_state()
            self._viewer3d_watchdog.start()
            host = self.window()
            host.hide()
            self._hidden_for_live_3d = True
            self._viewer3d.show()
            if hasattr(self._viewer3d, "app_window"):
                self._viewer3d.app_window.showNormal()
                self._viewer3d.app_window.raise_()
                self._viewer3d.app_window.activateWindow()
            self._set_runtime_status(f"Opened live 3D viewer for: {self._entry_box_path}")
        except Exception as exc:
            self._viewer3d = None
            self._viewer3d_temp_h5_path = None
            self._refresh_open_3d_state()
            self._emit_action_state()
            self._set_runtime_status(f"3D viewer launch failed: {exc}")

    def clear_fieldlines(self) -> None:
        self._fieldline_streamlines = []
        self._fieldline_z_base = 0.0
        while self._fieldline_artists:
            artist = self._fieldline_artists.pop()
            try:
                artist.remove()
            except Exception:
                pass
        self._can_clear_lines = False
        self._emit_action_state()
        self._canvas.draw_idle()
        self._set_runtime_status("Cleared over-plotted field lines.")

    def plot_fieldlines(self, streamlines, z_base=0.0) -> None:
        self._fieldline_streamlines = list(streamlines or [])
        self._fieldline_z_base = float(z_base)
        rendered = self._render_fieldlines()
        self._can_clear_lines = bool(self._fieldline_streamlines)
        self._emit_action_state()
        self._canvas.draw_idle()
        if self._fieldline_streamlines:
            if rendered > 0:
                self._set_runtime_status(
                    f"Received {len(self._fieldline_streamlines)} field-line bundle(s) from 3D viewer; "
                    f"rendered {rendered} line(s)."
                )
            else:
                self._set_runtime_status(
                    f"Received {len(self._fieldline_streamlines)} field-line bundle(s) from 3D viewer, "
                    "but no line segments projected into the current 2D view."
                )

    def _render_fieldlines(self) -> int:
        while self._fieldline_artists:
            artist = self._fieldline_artists.pop()
            try:
                artist.remove()
            except Exception:
                pass
        if not self._fieldline_streamlines or self._current_axes is None or self._current_map is None:
            return 0
        rendered_count = 0
        try:
            frame_hcc = self._fieldline_frame_hcc
            frame_obs = self._fieldline_frame_obs
            if frame_hcc is None or frame_obs is None:
                self._set_runtime_status(
                    "Field-line overlay unavailable: no legacy-equivalent 3D viewer frames are attached."
                )
                return 0
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "selector_fieldlines",
                ["#4c9aff", "#f6c945", "#e85d3f"],
                N=256,
            )
            norm = mcolors.Normalize(vmin=0.0, vmax=1000.0)
            for streamlines_subset in self._fieldline_streamlines:
                for coord, field in self._extract_streamlines(streamlines_subset):
                    # Mirror legacy gxbox_factory.plot_fieldlines():
                    # convert streamline coords from HCC to observer HPC, project to
                    # map pixels, then render pixel-space LineCollection segments.
                    coord_hcc = SkyCoord(
                        x=coord[:, 0] * u.Mm,
                        y=coord[:, 1] * u.Mm,
                        z=(coord[:, 2] + self._fieldline_z_base) * u.Mm,
                        frame=frame_hcc,
                    )
                    coord_hpc = coord_hcc.transform_to(frame_obs)
                    xpix, ypix = self._current_map.world_to_pixel(coord_hpc)
                    x = np.asarray(xpix.value if hasattr(xpix, "value") else xpix, dtype=float)
                    y = np.asarray(ypix.value if hasattr(ypix, "value") else ypix, dtype=float)
                    magnitude = np.asarray(field["magnitude"], dtype=float)
                    if x.size < 2 or y.size < 2 or magnitude.size < 2:
                        continue
                    finite = np.isfinite(x) & np.isfinite(y)
                    if np.count_nonzero(finite) < 2:
                        continue
                    segments = []
                    colors = []
                    for i in range(len(x) - 1):
                        if not (finite[i] and finite[i + 1]):
                            continue
                        segments.append(((x[i], y[i]), (x[i + 1], y[i + 1])))
                        color_idx = min(i, magnitude.size - 1)
                        colors.append(cmap(norm(magnitude[color_idx])))
                    if not segments:
                        continue
                    lc = LineCollection(segments, colors=colors, linewidths=0.7, alpha=0.7)
                    lc.set_zorder(20)
                    self._current_axes.add_collection(lc)
                    self._fieldline_artists.append(lc)
                    rendered_count += 1
        except Exception as exc:
            self._set_runtime_status(f"Field-line overlay failed: {exc}")
            return 0
        return rendered_count

    @staticmethod
    def _extract_streamlines(streamlines) -> list[tuple[np.ndarray, dict[str, np.ndarray]]]:
        out = []
        lines_arr = np.asarray(streamlines.lines)
        points = np.asarray(streamlines.points)
        i = 0
        n_lines = int(lines_arr.shape[0])
        while i < n_lines:
            num_points = int(lines_arr[i])
            start_idx = int(lines_arr[i + 1])
            end_idx = start_idx + num_points
            coord = points[start_idx:end_idx]
            bx = np.asarray(streamlines["bx"][start_idx:end_idx])
            by = np.asarray(streamlines["by"][start_idx:end_idx])
            bz = np.asarray(streamlines["bz"][start_idx:end_idx])
            out.append(
                (
                    coord,
                    {
                        "bx": bx,
                        "by": by,
                        "bz": bz,
                        "magnitude": np.sqrt(bx ** 2 + by ** 2 + bz ** 2),
                    },
                )
            )
            i += num_points + 1
        return out

    def _set_runtime_status(self, message: str) -> None:
        self._last_status_text = message
        if self._status_callback is not None:
            self._status_callback(message)

    def save_current_plot(self, output_path: str) -> None:
        self._fig.savefig(output_path, dpi=150, bbox_inches="tight")

    def _restore_preserved_view(self, prev_xlim, prev_ylim) -> None:
        if self._current_axes is None:
            return
        if self._view_mode == "box_fov":
            return
        prev_width = abs(float(prev_xlim[1] - prev_xlim[0]))
        prev_height = abs(float(prev_ylim[1] - prev_ylim[0]))
        if prev_width <= 0 or prev_height <= 0:
            return
        if self._state is not None and self._state.fov is not None and self._current_map is not None:
            try:
                observer = getattr(self._current_map, "observer_coordinate", None) or "earth"
                obstime = getattr(self._current_map, "date", None)
                fov = self._state.fov
                center_world = SkyCoord(
                    Tx=float(fov.center_x_arcsec) * u.arcsec,
                    Ty=float(fov.center_y_arcsec) * u.arcsec,
                    frame=Helioprojective(observer=observer, obstime=obstime),
                )
                cpx, cpy = self._current_map.wcs.world_to_pixel(center_world)
                if np.isfinite(cpx) and np.isfinite(cpy):
                    self._set_view_window(float(cpx), float(cpy), prev_width, prev_height)
                    return
            except Exception:
                pass
        self._current_axes.set_xlim(prev_xlim)
        self._current_axes.set_ylim(prev_ylim)

    @staticmethod
    def _default_context_id(session_input: SelectorSessionInput) -> Optional[str]:
        map_ids = list(session_input.map_ids or [])
        preferred = ["171", "193", "211", "304", "335", "1600", "Bz", "Ic", "Br", "Bp", "Bt"]
        for key in preferred:
            if key in map_ids:
                return key
        return map_ids[0] if map_ids else None

    @staticmethod
    def _default_bottom_id(session_input: SelectorSessionInput) -> Optional[str]:
        base_maps = dict(session_input.base_maps or {})
        if "bz" in base_maps:
            return "Bz"
        if "bx" in base_maps:
            return "Bx"
        if "by" in base_maps:
            return "By"
        if "vert_current" in base_maps:
            return "Vert_current"
        if "chromo_mask" in base_maps:
            return "chromo_mask"
        return None

    def _plot_box_outline(self, ax, smap) -> None:
        if self._state is None or self._state.geometry is None:
            return
        try:
            box = self._build_legacy_box(smap)
            if box is None:
                return

            for edge in box.bottom_edges:
                ax.plot_coord(edge, color="tab:red", ls="--", marker="", lw=1.0)
            for edge in box.non_bottom_edges:
                ax.plot_coord(edge, color="tab:red", ls="-", marker="", lw=1.0)

            def _edge_pixel_bounds(edges) -> tuple[float, float, float, float] | None:
                xs: list[float] = []
                ys: list[float] = []
                for edge in edges:
                    try:
                        px, py = smap.wcs.world_to_pixel(edge)
                    except Exception:
                        continue
                    px = np.asarray(px, dtype=float).ravel()
                    py = np.asarray(py, dtype=float).ravel()
                    finite = np.isfinite(px) & np.isfinite(py)
                    if np.any(finite):
                        xs.extend(px[finite].tolist())
                        ys.extend(py[finite].tolist())
                if not xs or not ys:
                    return None
                return (
                    float(np.nanmin(xs)),
                    float(np.nanmax(xs)),
                    float(np.nanmin(ys)),
                    float(np.nanmax(ys)),
                )

            full_bounds = _edge_pixel_bounds(list(box.bottom_edges) + list(box.non_bottom_edges))
            bottom_bounds = _edge_pixel_bounds(list(box.bottom_edges))
            if full_bounds is None or bottom_bounds is None:
                return
            x0, x1, y0, y1 = full_bounds
            bx0, bx1, by0, by1 = bottom_bounds
            self._projected_box_fov = self._box_bounds_to_fov_selection(box, smap)
            self._projected_box_bbox_rect = self._fov_selection_to_pixel_rect(smap, self._projected_box_fov)
            if self._state.fov is None:
                self._state.fov = DisplayFovSelection(
                    center_x_arcsec=self._projected_box_fov.center_x_arcsec,
                    center_y_arcsec=self._projected_box_fov.center_y_arcsec,
                    width_arcsec=self._projected_box_fov.width_arcsec,
                    height_arcsec=self._projected_box_fov.height_arcsec,
                )
                if self._fov_change_callback is not None:
                    self._fov_change_callback(self._state.fov)
            if self._state.fov_box is None:
                self._state.fov_box = self._compute_fov_box_from_geometry()
            fov_rect = self._fov_selection_to_pixel_rect(smap, self._state.fov)
            fx0, fy0 = fov_rect.get_x(), fov_rect.get_y()
            fw, fh = fov_rect.get_width(), fov_rect.get_height()
            fov_half_w = 0.5 * max(float(self._state.fov.width_arcsec), 1e-3)
            fov_half_h = 0.5 * max(float(self._state.fov.height_arcsec), 1e-3)
            fov_bl = SkyCoord(
                Tx=(self._state.fov.center_x_arcsec - fov_half_w) * u.arcsec,
                Ty=(self._state.fov.center_y_arcsec - fov_half_h) * u.arcsec,
                frame=Helioprojective(
                    observer=getattr(smap, "observer_coordinate", None) or "earth",
                    obstime=getattr(smap, "date", None),
                ),
            )
            fov_tr = SkyCoord(
                Tx=(self._state.fov.center_x_arcsec + fov_half_w) * u.arcsec,
                Ty=(self._state.fov.center_y_arcsec + fov_half_h) * u.arcsec,
                frame=Helioprojective(
                    observer=getattr(smap, "observer_coordinate", None) or "earth",
                    obstime=getattr(smap, "date", None),
                ),
            )
            try:
                smap.draw_quadrangle(
                    fov_bl,
                    top_right=fov_tr,
                    axes=ax,
                    edgecolor="deepskyblue",
                    linestyle="--",
                    linewidth=0.8,
                )
            except Exception:
                pass

            # Invisible rectangles retained as the internal geometry extents for
            # button-driven manipulations and Box-FOV calculations.
            self._overlay_rect = Rectangle(
                (bx0, by0),
                max(1e-6, bx1 - bx0),
                max(1e-6, by1 - by0),
                visible=False,
            )
            self._overlay_bbox_rect = Rectangle(
                (fx0, fy0),
                max(1e-6, fw),
                max(1e-6, fh),
                visible=False,
            )

            anchor = self._geometry_anchor_coord(self._state.geometry, smap).transform_to(box._frame_obs)
            cpx, cpy = smap.wcs.world_to_pixel(anchor)
            self._overlay_center_artist = ax.plot(
                [float(cpx)], [float(cpy)],
                marker="+", color="yellow", ms=10, mew=1.5,
                transform=ax.get_transform("pixel"),
            )[0]
        except Exception:
            # Overlay failure should not break map display.
            return

    def _geometry_center_hpc_for_map(self, geom: BoxGeometrySelection, smap):
        return self._geometry_anchor_coord(geom, smap).transform_to(
            Helioprojective(
                observer=getattr(smap, "observer_coordinate", None) or "earth",
                obstime=getattr(smap, "date", None),
            )
        )

    @staticmethod
    def _box_half_extent_arcsec(n_pix: int, dx_km: float, smap) -> float:
        # Approximate small-angle conversion using observer distance.
        dsun_km = None
        try:
            dsun_km = smap.observer_coordinate.radius.to_value(u.km)
        except Exception:
            try:
                dsun_km = smap.dsun.to_value(u.km)
            except Exception:
                try:
                    dsun_obs = smap.meta.get("dsun_obs")
                    if dsun_obs is not None:
                        dsun_km = (float(dsun_obs) * u.m).to_value(u.km)
                except Exception:
                    dsun_km = None
        if not dsun_km or dsun_km <= 0:
            dsun_km = 1.496e8  # fallback ~1 AU
        half_size_km = 0.5 * float(n_pix) * float(dx_km)
        return half_size_km / dsun_km * 206265.0

    @staticmethod
    def _geometry_pixel_arcsec(geom: BoxGeometrySelection, smap) -> float:
        dsun_km = MapBoxDisplayWidget._dsun_km_from_map(smap)
        return float(max(geom.dx_km, 1e-6) / max(dsun_km, 1e-6) * 206265.0)

    def _geometry_from_world_center(self, geom: BoxGeometrySelection, world_center) -> BoxGeometrySelection:
        out = BoxGeometrySelection(
            coord_mode=geom.coord_mode,
            coord_x=geom.coord_x,
            coord_y=geom.coord_y,
            grid_x=geom.grid_x,
            grid_y=geom.grid_y,
            grid_z=geom.grid_z,
            dx_km=geom.dx_km,
        )
        try:
            if geom.coord_mode == CoordMode.HPC:
                c = world_center.transform_to(
                    Helioprojective(
                        obstime=getattr(self._current_map, "date", None),
                        observer=getattr(self._current_map, "observer_coordinate", "earth"),
                    )
                )
                out.coord_x = float(c.Tx.to_value(u.arcsec))
                out.coord_y = float(c.Ty.to_value(u.arcsec))
            elif geom.coord_mode == CoordMode.HGC:
                c = world_center.transform_to(
                    HeliographicCarrington(
                        obstime=getattr(self._current_map, "date", None),
                        observer=getattr(self._current_map, "observer_coordinate", "earth"),
                    )
                )
                out.coord_x = float(c.lon.to_value(u.deg))
                out.coord_y = float(c.lat.to_value(u.deg))
            else:
                c = world_center.transform_to(
                    HeliographicStonyhurst(obstime=getattr(self._current_map, "date", None))
                )
                out.coord_x = float(c.lon.to_value(u.deg))
                out.coord_y = float(c.lat.to_value(u.deg))
        except Exception:
            pass
        return out

    def show_full_sun_view(self) -> None:
        self._view_mode = "full_sun"
        self._refresh_plot()

    def show_box_fov_view(self, pad_factor: float | None = None) -> None:
        # `pad_factor` is handled by the display-crop helper; keep the method
        # signature stable for existing button hookups.
        self._view_mode = "box_fov"
        self._refresh_plot()

    def _set_view_window(self, cx: float, cy: float, width: float, height: float) -> None:
        if self._current_axes is None:
            return
        ax = self._current_axes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_dir = 1.0 if xlim[1] >= xlim[0] else -1.0
        y_dir = 1.0 if ylim[1] >= ylim[0] else -1.0
        half_w = 0.5 * max(width, 4.0)
        half_h = 0.5 * max(height, 4.0)
        ax.set_xlim((cx - half_w, cx + half_w) if x_dir > 0 else (cx + half_w, cx - half_w))
        ax.set_ylim((cy - half_h, cy + half_h) if y_dir > 0 else (cy + half_h, cy - half_h))
        self._canvas.draw_idle()

    def _set_view_window_hpc(
        self,
        center_x_arcsec: float,
        center_y_arcsec: float,
        width_arcsec: float,
        height_arcsec: float,
    ) -> None:
        if self._current_axes is None or self._current_map is None:
            return
        half_w = 0.5 * max(float(width_arcsec), 1e-3)
        half_h = 0.5 * max(float(height_arcsec), 1e-3)
        observer = getattr(self._current_map, "observer_coordinate", None) or "earth"
        obstime = getattr(self._current_map, "date", None)
        bottom_left = SkyCoord(
            Tx=(center_x_arcsec - half_w) * u.arcsec,
            Ty=(center_y_arcsec - half_h) * u.arcsec,
            frame=Helioprojective(observer=observer, obstime=obstime),
        )
        top_right = SkyCoord(
            Tx=(center_x_arcsec + half_w) * u.arcsec,
            Ty=(center_y_arcsec + half_h) * u.arcsec,
            frame=Helioprojective(observer=observer, obstime=obstime),
        )
        try:
            corners = SkyCoord([bottom_left, top_right])
            px, py = self._current_map.wcs.world_to_pixel(corners)
            px = np.asarray(px, dtype=float).ravel()
            py = np.asarray(py, dtype=float).ravel()
            finite = np.isfinite(px) & np.isfinite(py)
            if not np.any(finite):
                return
            px = px[finite]
            py = py[finite]
            x0, x1 = float(np.nanmin(px)), float(np.nanmax(px))
            y0, y1 = float(np.nanmin(py)), float(np.nanmax(py))
            self._set_view_window(
                cx=0.5 * (x0 + x1),
                cy=0.5 * (y0 + y1),
                width=max(x1 - x0, 4.0),
                height=max(y1 - y0, 4.0),
            )
        except Exception:
            return

    def _scale_view(self, factor: float, center_px: tuple[float, float] | None = None) -> None:
        if self._current_axes is None:
            return
        ax = self._current_axes
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        if self._state is not None and self._state.fov is not None and self._current_map is not None:
            try:
                observer = getattr(self._current_map, "observer_coordinate", None) or "earth"
                obstime = getattr(self._current_map, "date", None)
                fov = self._state.fov
                center_world = SkyCoord(
                    Tx=float(fov.center_x_arcsec) * u.arcsec,
                    Ty=float(fov.center_y_arcsec) * u.arcsec,
                    frame=Helioprojective(observer=observer, obstime=obstime),
                )
                cpx, cpy = self._current_map.wcs.world_to_pixel(center_world)
                if np.isfinite(cpx) and np.isfinite(cpy):
                    cx = float(cpx)
                    cy = float(cpy)
                else:
                    cx = 0.5 * (x0 + x1)
                    cy = 0.5 * (y0 + y1)
            except Exception:
                cx = 0.5 * (x0 + x1)
                cy = 0.5 * (y0 + y1)
        elif center_px is not None:
            cx, cy = center_px
        else:
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
        width = abs(x1 - x0) * float(factor)
        height = abs(y1 - y0) * float(factor)
        self._set_view_window(cx, cy, max(width, 4.0), max(height, 4.0))

    def _nudge_box_size(self, axis: str, sign: int) -> None:
        if not self._geometry_edit_enabled or self._state is None or self._state.geometry is None:
            return
        geom = self._state.geometry
        step = self._coarse_box_grid_step(axis)
        new_geom = BoxGeometrySelection(
            coord_mode=geom.coord_mode,
            coord_x=geom.coord_x,
            coord_y=geom.coord_y,
            grid_x=geom.grid_x,
            grid_y=geom.grid_y,
            grid_z=geom.grid_z,
            dx_km=geom.dx_km,
        )
        if axis == "x":
            new_geom.grid_x = max(1, geom.grid_x + int(sign) * step)
        elif axis == "y":
            new_geom.grid_y = max(1, geom.grid_y + int(sign) * step)
        else:
            return
        self.set_geometry_selection(new_geom)

    def _nudge_box_size_xy(self, sign: int) -> None:
        if not self._geometry_edit_enabled or self._state is None or self._state.geometry is None:
            return
        geom = self._state.geometry
        step_x = self._coarse_box_grid_step("x")
        step_y = self._coarse_box_grid_step("y")
        new_geom = BoxGeometrySelection(
            coord_mode=geom.coord_mode,
            coord_x=geom.coord_x,
            coord_y=geom.coord_y,
            grid_x=max(1, geom.grid_x + int(sign) * step_x),
            grid_y=max(1, geom.grid_y + int(sign) * step_y),
            grid_z=geom.grid_z,
            dx_km=geom.dx_km,
        )
        self.set_geometry_selection(new_geom)

    def _nudge_fov_size(self, axis: str, sign: int) -> None:
        if self._state is None or self._state.fov is None:
            return
        fov = self._state.fov
        step = self._coarse_fov_step(axis)
        width = float(fov.width_arcsec)
        height = float(fov.height_arcsec)
        if axis == "x":
            width = max(step, width + int(sign) * step)
            if self._state.square_fov:
                height = width
        elif axis == "y":
            if self._state.square_fov:
                return
            height = max(step, height + int(sign) * step)
        else:
            return
        self.set_fov_selection(
            DisplayFovSelection(
                center_x_arcsec=fov.center_x_arcsec,
                center_y_arcsec=fov.center_y_arcsec,
                width_arcsec=width,
                height_arcsec=height,
            )
        )

    def _nudge_fov_size_xy(self, sign: int) -> None:
        if self._state is None or self._state.fov is None:
            return
        fov = self._state.fov
        step_x = self._coarse_fov_step("x")
        step_y = self._coarse_fov_step("y")
        width = max(step_x, float(fov.width_arcsec) + int(sign) * step_x)
        height = max(step_y, float(fov.height_arcsec) + int(sign) * step_y)
        if self._state.square_fov:
            height = width
        self.set_fov_selection(
            DisplayFovSelection(
                center_x_arcsec=fov.center_x_arcsec,
                center_y_arcsec=fov.center_y_arcsec,
                width_arcsec=width,
                height_arcsec=height,
            )
        )

    def _nudge_fov_center(self, axis: str, sign: int) -> None:
        if self._state is None or self._state.fov is None:
            return
        fov = self._state.fov
        step = self._coarse_fov_step(axis)
        cx = float(fov.center_x_arcsec)
        cy = float(fov.center_y_arcsec)
        if axis == "x":
            cx += float(sign) * step
        elif axis == "y":
            cy += float(sign) * step
        else:
            return
        self.set_fov_selection(
            DisplayFovSelection(
                center_x_arcsec=cx,
                center_y_arcsec=cy,
                width_arcsec=fov.width_arcsec,
                height_arcsec=fov.height_arcsec,
            )
        )

    def _nudge_box_center(self, axis: str, sign: int) -> None:
        if not self._geometry_edit_enabled or self._state is None or self._state.geometry is None or self._current_map is None:
            return
        geom = self._state.geometry
        step_arcsec = self._coarse_box_center_step_arcsec(axis)
        if not np.isfinite(step_arcsec) or step_arcsec <= 0:
            return
        center_hpc = self._geometry_center_hpc_for_map(geom, self._current_map)
        tx = float(center_hpc.Tx.to_value(u.arcsec))
        ty = float(center_hpc.Ty.to_value(u.arcsec))
        if axis == "x":
            tx += float(sign) * step_arcsec
        elif axis == "y":
            ty += float(sign) * step_arcsec
        else:
            return
        nudged_center = SkyCoord(
            Tx=tx * u.arcsec,
            Ty=ty * u.arcsec,
            frame=Helioprojective(
                observer=getattr(self._current_map, "observer_coordinate", None) or "earth",
                obstime=getattr(self._current_map, "date", None),
            ),
        )
        new_geom = self._geometry_from_world_center(geom, nudged_center)
        self.set_geometry_selection(new_geom)

    def _coarse_box_grid_step(self, axis: str) -> int:
        if self._state is None or self._state.geometry is None:
            return 1
        geom = self._state.geometry
        dim = geom.grid_x if axis == "x" else geom.grid_y if axis == "y" else geom.grid_z
        return max(1, int(round(float(dim) * 0.10)))

    def _coarse_fov_step(self, axis: str) -> float:
        if self._state is None or self._state.fov is None:
            return 1.0
        fov = self._state.fov
        span = float(fov.width_arcsec) if axis == "x" else float(fov.height_arcsec)
        base_step = 1.0
        if self._state.geometry is not None and self._current_map is not None:
            try:
                base_step = max(1e-3, self._geometry_pixel_arcsec(self._state.geometry, self._current_map))
            except Exception:
                base_step = 1.0
        return max(base_step, abs(span) * 0.10)

    def _coarse_box_center_step_arcsec(self, axis: str) -> float:
        if self._state is None or self._state.geometry is None or self._current_map is None:
            return 0.0
        geom = self._state.geometry
        step_arcsec = self._geometry_pixel_arcsec(geom, self._current_map)
        if not np.isfinite(step_arcsec) or step_arcsec <= 0:
            return 0.0
        grid_step = self._coarse_box_grid_step(axis)
        return float(step_arcsec) * float(grid_step)

    def _compute_fov_box_from_geometry(self) -> Optional[DisplayFovBoxSelection]:
        if self._state is None or self._state.fov is None or self._current_map is None:
            return None
        box = self._build_legacy_box(self._current_map)
        if box is None:
            return None
        observer = getattr(self._current_map, "observer_coordinate", None) or "earth"
        obstime = getattr(self._current_map, "date", None)
        try:
            frame_hcc_obs = Heliocentric(observer=observer, obstime=obstime)
            xx = []
            yy = []
            zz = []
            for edge in box.all_edges:
                edge_hcc = edge.transform_to(frame_hcc_obs)
                xx.extend(np.asarray(edge_hcc.x.to_value(u.Mm), dtype=float).ravel().tolist())
                yy.extend(np.asarray(edge_hcc.y.to_value(u.Mm), dtype=float).ravel().tolist())
                zz.extend(np.asarray(edge_hcc.z.to_value(u.Mm), dtype=float).ravel().tolist())
            z_arr = np.asarray(zz, dtype=float)
            finite = np.isfinite(z_arr)
            if not np.any(finite):
                return None
            z_arr = z_arr[finite]
            z_min = float(np.nanmin(z_arr))
            z_max = float(np.nanmax(z_arr))
            span = max(1e-3, z_max - z_min)
            pad = 0.10 * span
            return DisplayFovBoxSelection.from_display_fov(self._state.fov, z_min - pad, z_max + pad)
        except Exception:
            return None

    def recompute_fov_from_box(self) -> None:
        if self._state is None or self._projected_box_fov is None:
            return
        width = self._projected_box_fov.width_arcsec
        height = self._projected_box_fov.height_arcsec
        if self._state.square_fov:
            height = width
        self.set_fov_selection(
            DisplayFovSelection(
                center_x_arcsec=self._projected_box_fov.center_x_arcsec,
                center_y_arcsec=self._projected_box_fov.center_y_arcsec,
                width_arcsec=width,
                height_arcsec=height,
            )
        )
        self._state.fov_box = self._compute_fov_box_from_geometry()
        self._refresh_status_text()

    def _on_scroll(self, event) -> None:
        if self._current_axes is None or event.inaxes is not self._current_axes:
            return
        if event.xdata is None or event.ydata is None:
            return
        if getattr(event, "button", None) == "up":
            self._scale_view(1 / 1.12, center_px=(event.xdata, event.ydata))
        elif getattr(event, "button", None) == "down":
            self._scale_view(1.12, center_px=(event.xdata, event.ydata))

    def _overlay_rect_bounds(self):
        if self._overlay_rect is None:
            return None
        x0, y0 = self._overlay_rect.get_x(), self._overlay_rect.get_y()
        w, h = self._overlay_rect.get_width(), self._overlay_rect.get_height()
        return x0, y0, x0 + w, y0 + h

    def _fov_selection_to_pixel_rect(self, smap, fov: DisplayFovSelection | None) -> Rectangle:
        if fov is None:
            if self._projected_box_bbox_rect is not None:
                return Rectangle(
                    (self._projected_box_bbox_rect.get_x(), self._projected_box_bbox_rect.get_y()),
                    self._projected_box_bbox_rect.get_width(),
                    self._projected_box_bbox_rect.get_height(),
                    visible=False,
                )
            return Rectangle((0.0, 0.0), 10.0, 10.0, visible=False)
        observer = getattr(smap, "observer_coordinate", None) or "earth"
        obstime = getattr(smap, "date", None)
        half_w = 0.5 * max(float(fov.width_arcsec), 1e-3)
        half_h = 0.5 * max(float(fov.height_arcsec), 1e-3)
        bottom_left = SkyCoord(
            Tx=(fov.center_x_arcsec - half_w) * u.arcsec,
            Ty=(fov.center_y_arcsec - half_h) * u.arcsec,
            frame=Helioprojective(observer=observer, obstime=obstime),
        )
        top_right = SkyCoord(
            Tx=(fov.center_x_arcsec + half_w) * u.arcsec,
            Ty=(fov.center_y_arcsec + half_h) * u.arcsec,
            frame=Helioprojective(observer=observer, obstime=obstime),
        )
        try:
            px, py = smap.wcs.world_to_pixel(SkyCoord([bottom_left, top_right]))
            px = np.asarray(px, dtype=float).ravel()
            py = np.asarray(py, dtype=float).ravel()
            finite = np.isfinite(px) & np.isfinite(py)
            if np.any(finite):
                px = px[finite]
                py = py[finite]
                x0, x1 = float(np.nanmin(px)), float(np.nanmax(px))
                y0, y1 = float(np.nanmin(py)), float(np.nanmax(py))
                return Rectangle((x0, y0), max(1e-6, x1 - x0), max(1e-6, y1 - y0), visible=False)
        except Exception:
            pass
        # Fallback to projected-box bounds if corner projection fails.
        if self._projected_box_bbox_rect is not None:
            return Rectangle(
                (self._projected_box_bbox_rect.get_x(), self._projected_box_bbox_rect.get_y()),
                self._projected_box_bbox_rect.get_width(),
                self._projected_box_bbox_rect.get_height(),
                visible=False,
            )
        return Rectangle((0.0, 0.0), 10.0, 10.0, visible=False)

    def _box_bounds_to_fov_selection(self, box, smap) -> DisplayFovSelection:
        bounds = box.bounds_coords.transform_to(
            Helioprojective(
                observer=getattr(smap, "observer_coordinate", None) or "earth",
                obstime=getattr(smap, "date", None),
            )
        )
        tx = np.asarray(bounds.Tx.to_value(u.arcsec), dtype=float).ravel()
        ty = np.asarray(bounds.Ty.to_value(u.arcsec), dtype=float).ravel()
        finite = np.isfinite(tx) & np.isfinite(ty)
        if not np.any(finite):
            return DisplayFovSelection(0.0, 0.0, 10.0, 10.0)
        tx = tx[finite]
        ty = ty[finite]
        xmin, xmax = float(np.nanmin(tx)), float(np.nanmax(tx))
        ymin, ymax = float(np.nanmin(ty)), float(np.nanmax(ty))
        return DisplayFovSelection(
            center_x_arcsec=0.5 * (xmin + xmax),
            center_y_arcsec=0.5 * (ymin + ymax),
            width_arcsec=max(1e-3, xmax - xmin),
            height_arcsec=max(1e-3, ymax - ymin),
        )

    def _pixel_rect_to_fov_selection(self, smap, rect: Rectangle) -> DisplayFovSelection:
        x0, y0 = rect.get_x(), rect.get_y()
        w, h = rect.get_width(), rect.get_height()
        cx = x0 + 0.5 * w
        cy = y0 + 0.5 * h
        world = smap.wcs.pixel_to_world(cx, cy)
        hpc = world.transform_to(
            Helioprojective(
                observer=getattr(smap, "observer_coordinate", None) or "earth",
                obstime=getattr(smap, "date", None),
            )
        )
        scale_x = self._map_pixel_scale_arcsec(smap, axis=0)
        scale_y = self._map_pixel_scale_arcsec(smap, axis=1)
        return DisplayFovSelection(
            center_x_arcsec=float(hpc.Tx.to_value(u.arcsec)),
            center_y_arcsec=float(hpc.Ty.to_value(u.arcsec)),
            width_arcsec=float(abs(w) * scale_x),
            height_arcsec=float(abs(h) * scale_y),
        )

    @staticmethod
    def _map_pixel_scale_arcsec(smap, axis: int) -> float:
        try:
            if axis == 0:
                return abs(float(smap.scale.axis1.to_value(u.arcsec / u.pix)))
            return abs(float(smap.scale.axis2.to_value(u.arcsec / u.pix)))
        except Exception:
            return 0.6

    def _hit_test_overlay(self, ex: float, ey: float):
        bounds = self._overlay_rect_bounds()
        if bounds is None:
            return None
        x0, y0, x1, y1 = bounds
        w, h = x1 - x0, y1 - y0
        cx, cy = x0 + 0.5 * w, y0 + 0.5 * h
        tol = max(6.0, 0.03 * max(w, h))

        if abs(ex - cx) <= tol and abs(ey - cy) <= tol:
            return {"kind": "move"}

        corners = {"bl": (x0, y0), "br": (x1, y0), "tr": (x1, y1), "tl": (x0, y1)}
        for corner_name, (hx, hy) in corners.items():
            if abs(ex - hx) <= tol and abs(ey - hy) <= tol:
                return {"kind": "resize_corner", "corner": corner_name}

        if y0 - tol <= ey <= y1 + tol and abs(ex - x0) <= tol:
            return {"kind": "resize_x", "side": "left"}
        if y0 - tol <= ey <= y1 + tol and abs(ex - x1) <= tol:
            return {"kind": "resize_x", "side": "right"}
        if x0 - tol <= ex <= x1 + tol and abs(ey - y0) <= tol:
            return {"kind": "resize_y", "side": "bottom"}
        if x0 - tol <= ex <= x1 + tol and abs(ey - y1) <= tol:
            return {"kind": "resize_y", "side": "top"}

        if x0 <= ex <= x1 and y0 <= ey <= y1:
            return {"kind": "inside"}
        return None

    def _build_drag_state_from_click(self, ex: float, ey: float):
        bounds = self._overlay_rect_bounds()
        if bounds is None:
            return None
        x0, y0, x1, y1 = bounds
        w, h = x1 - x0, y1 - y0
        cx, cy = x0 + 0.5 * w, y0 + 0.5 * h
        hit = self._hit_test_overlay(ex, ey)

        if self._interaction_mode == "auto":
            if hit is None:
                return None
            if hit["kind"] in {"move", "inside"}:
                return {"mode": "move", "dx": ex - cx, "dy": ey - cy}
            if hit["kind"] == "resize_corner":
                corner_name = hit["corner"]
                return {
                    "mode": "resize",
                    "corner": corner_name,
                    "anchor_x": x1 if "l" in corner_name else x0,
                    "anchor_y": y1 if "b" in corner_name else y0,
                }
            if hit["kind"] == "resize_x":
                return {"mode": "resize_x", "anchor_x": x1 if hit["side"] == "left" else x0}
            if hit["kind"] == "resize_y":
                return {"mode": "resize_y", "anchor_y": y1 if hit["side"] == "bottom" else y0}
            return None

        if hit is None and self._interaction_mode in {"move", "resize_xy", "resize_x", "resize_y"}:
            return None

        if self._interaction_mode == "move":
            return {"mode": "move", "dx": ex - cx, "dy": ey - cy}
        if self._interaction_mode == "resize_xy":
            # Pick the active corner by click quadrant around current center.
            return {
                "mode": "resize",
                "corner": ("t" if ey >= cy else "b") + ("r" if ex >= cx else "l"),
                "anchor_x": x0 if ex >= cx else x1,
                "anchor_y": y0 if ey >= cy else y1,
            }
        if self._interaction_mode == "resize_x":
            return {"mode": "resize_x", "anchor_x": x0 if ex >= cx else x1}
        if self._interaction_mode == "resize_y":
            return {"mode": "resize_y", "anchor_y": y0 if ey >= cy else y1}
        return None

    def _update_cursor_for_mode(self) -> None:
        if self._drag_state is not None:
            return
        if self._interaction_mode == "move":
            self._canvas.setCursor(Qt.SizeAllCursor)
        elif self._interaction_mode == "resize_x":
            self._canvas.setCursor(Qt.SizeHorCursor)
        elif self._interaction_mode == "resize_y":
            self._canvas.setCursor(Qt.SizeVerCursor)
        elif self._interaction_mode == "resize_xy":
            self._canvas.setCursor(Qt.SizeFDiagCursor)
        else:
            self._canvas.setCursor(Qt.ArrowCursor)

    def _update_hover_cursor(self, event) -> None:
        if self._interaction_mode != "auto":
            self._update_cursor_for_mode()
            return
        if event is None or event.inaxes is not self._current_axes or event.xdata is None or event.ydata is None:
            self._canvas.setCursor(Qt.ArrowCursor)
            return
        hit = self._hit_test_overlay(float(event.xdata), float(event.ydata))
        if hit is None:
            self._canvas.setCursor(Qt.ArrowCursor)
            return
        if hit["kind"] in {"move", "inside"}:
            self._canvas.setCursor(Qt.SizeAllCursor)
        elif hit["kind"] == "resize_x":
            self._canvas.setCursor(Qt.SizeHorCursor)
        elif hit["kind"] == "resize_y":
            self._canvas.setCursor(Qt.SizeVerCursor)
        elif hit["kind"] == "resize_corner":
            self._canvas.setCursor(Qt.SizeFDiagCursor if hit["corner"] in {"bl", "tr"} else Qt.SizeBDiagCursor)
        else:
            self._canvas.setCursor(Qt.ArrowCursor)

    def _on_mouse_press(self, event) -> None:
        if not self._mouse_actions_enabled:
            return
        if event.button != 1 or event.inaxes is None:
            return
        if self._state is None or self._state.geometry is None or self._current_map is None or self._overlay_rect is None:
            return
        if event.inaxes is not self._current_axes:
            return
        ex, ey = event.xdata, event.ydata
        if ex is None or ey is None:
            return

        self._drag_state = self._build_drag_state_from_click(float(ex), float(ey))
        if self._drag_state is not None:
            self._update_cursor_for_mode()

    def _on_mouse_move(self, event) -> None:
        if not self._mouse_actions_enabled:
            return
        if self._drag_state is None:
            self._update_hover_cursor(event)
            return
        if event.inaxes is not self._current_axes or event.xdata is None or event.ydata is None:
            return
        if self._state is None or self._state.geometry is None or self._current_map is None:
            return
        geom = self._state.geometry
        smap = self._current_map
        x = float(event.xdata)
        y = float(event.ydata)

        try:
            if self._drag_state["mode"] == "move":
                new_cx = x - self._drag_state["dx"]
                new_cy = y - self._drag_state["dy"]
                new_geom = self._geometry_from_pixel_edit(
                    geom,
                    center_px=(new_cx, new_cy),
                )
            elif self._drag_state["mode"] == "resize":
                ax_x = float(self._drag_state["anchor_x"])
                ax_y = float(self._drag_state["anchor_y"])
                cx = 0.5 * (ax_x + x)
                cy = 0.5 * (ax_y + y)
                new_geom = self._geometry_from_pixel_edit(
                    geom,
                    center_px=(cx, cy),
                    size_px=(abs(x - ax_x), abs(y - ax_y)),
                )
            elif self._drag_state["mode"] == "resize_x":
                rect = self._overlay_rect
                y0, h = rect.get_y(), rect.get_height()
                ax_x = float(self._drag_state["anchor_x"])
                cx = 0.5 * (ax_x + x)
                cy = y0 + 0.5 * h
                new_geom = self._geometry_from_pixel_edit(
                    geom,
                    center_px=(cx, cy),
                    size_px=(abs(x - ax_x), h),
                )
            elif self._drag_state["mode"] == "resize_y":
                rect = self._overlay_rect
                x0, w = rect.get_x(), rect.get_width()
                ax_y = float(self._drag_state["anchor_y"])
                cx = x0 + 0.5 * w
                cy = 0.5 * (ax_y + y)
                new_geom = self._geometry_from_pixel_edit(
                    geom,
                    center_px=(cx, cy),
                    size_px=(w, abs(y - ax_y)),
                )
            else:
                return
        except Exception:
            return

        self.set_geometry_selection(new_geom)

    def _on_mouse_release(self, event) -> None:
        if not self._mouse_actions_enabled:
            return
        self._drag_state = None
        self._update_hover_cursor(event)

    def _geometry_from_pixel_edit(self, geom: BoxGeometrySelection, center_px=None, size_px=None) -> BoxGeometrySelection:
        smap = self._current_map
        out = BoxGeometrySelection(
            coord_mode=geom.coord_mode,
            coord_x=geom.coord_x,
            coord_y=geom.coord_y,
            grid_x=geom.grid_x,
            grid_y=geom.grid_y,
            grid_z=geom.grid_z,
            dx_km=geom.dx_km,
        )

        if center_px is not None:
            world = smap.wcs.pixel_to_world(float(center_px[0]), float(center_px[1]))
            try:
                if geom.coord_mode == CoordMode.HPC:
                    c = world.transform_to(Helioprojective(
                        obstime=getattr(smap, "date", None),
                        observer=getattr(smap, "observer_coordinate", "earth"),
                    ))
                    out.coord_x = float(c.Tx.to_value(u.arcsec))
                    out.coord_y = float(c.Ty.to_value(u.arcsec))
                elif geom.coord_mode == CoordMode.HGC:
                    c = world.transform_to(HeliographicCarrington(
                        obstime=getattr(smap, "date", None),
                        observer=getattr(smap, "observer_coordinate", "earth"),
                    ))
                    out.coord_x = float(c.lon.to_value(u.deg))
                    out.coord_y = float(c.lat.to_value(u.deg))
                else:
                    c = world.transform_to(HeliographicStonyhurst(obstime=getattr(smap, "date", None)))
                    out.coord_x = float(c.lon.to_value(u.deg))
                    out.coord_y = float(c.lat.to_value(u.deg))
            except Exception:
                pass

        if size_px is not None:
            try:
                center_hpc = self._geometry_center_hpc_for_map(out, smap)
                cpx, cpy = smap.wcs.world_to_pixel(center_hpc)
                wpx = max(1.0, float(size_px[0]))
                hpx = max(1.0, float(size_px[1]))
                x0 = float(cpx) - 0.5 * wpx
                x1 = float(cpx) + 0.5 * wpx
                y0 = float(cpy) - 0.5 * hpx
                y1 = float(cpy) + 0.5 * hpx
                wx = smap.wcs.pixel_to_world([x0, x1], [float(cpy), float(cpy)])
                wy = smap.wcs.pixel_to_world([float(cpx), float(cpx)], [y0, y1])
                dx_arcsec = abs(wx[1].Tx.to_value(u.arcsec) - wx[0].Tx.to_value(u.arcsec))
                dy_arcsec = abs(wy[1].Ty.to_value(u.arcsec) - wy[0].Ty.to_value(u.arcsec))
                dsun_km = self._dsun_km_from_map(smap)
                width_km = dx_arcsec / 206265.0 * dsun_km
                height_km = dy_arcsec / 206265.0 * dsun_km
                out.grid_x = max(1, int(round(width_km / max(out.dx_km, 1e-6))))
                out.grid_y = max(1, int(round(height_km / max(out.dx_km, 1e-6))))
            except Exception:
                pass

        return out

    @staticmethod
    def _dsun_km_from_map(smap) -> float:
        try:
            return float(smap.observer_coordinate.radius.to_value(u.km))
        except Exception:
            try:
                return float(smap.dsun.to_value(u.km))
            except Exception:
                dsun_obs = smap.meta.get("dsun_obs")
                if dsun_obs is not None:
                    return float((float(dsun_obs) * u.m).to_value(u.km))
        return 1.496e8
