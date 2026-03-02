from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from .map_box_view import MapBoxDisplayWidget
from .selector_api import (
    BoxGeometrySelection,
    CoordMode,
    DisplayFovSelection,
    SelectorDialogResult,
    SelectorSessionInput,
)


class FovBoxSelectorDialog(QDialog):
    """
    Standalone post-download FOV/box selection GUI scaffold.

    This dialog is intentionally minimal at this stage:
    - it accepts/returns the finalized geometry contract (`BoxGeometrySelection`)
    - it hosts a reusable `MapBoxDisplayWidget`
    - it does not yet implement interactive plotting/dragging
    """

    def __init__(
        self,
        session_input: SelectorSessionInput,
        parent: Optional[QWidget] = None,
        entry_box_path: Optional[str | Path] = None,
    ):
        super().__init__(parent)
        self._session_input = session_input
        self._accepted_selection: Optional[SelectorDialogResult] = None
        self._entry_box_path = Path(entry_box_path).expanduser().resolve() if entry_box_path else None
        self.setWindowTitle("FOV / Box Selector")
        self.resize(1180, 760)

        self._build_ui()
        self._load_session_input(session_input)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        body = QGridLayout()
        root.addLayout(body, stretch=1)

        self.context_map_combo = QComboBox()
        self.context_map_combo.currentIndexChanged.connect(self._on_context_map_changed)
        self.bottom_map_combo = QComboBox()
        self.bottom_map_combo.currentIndexChanged.connect(self._on_bottom_map_changed)
        self.map_source_combo = QComboBox()
        self.map_source_combo.addItem("Auto", "auto")
        self.map_source_combo.addItem("Filesystem", "filesystem")
        self.map_source_combo.addItem("Embedded", "embedded")
        self.map_source_combo.currentIndexChanged.connect(self._on_map_source_changed)

        left_group = QGroupBox("Visualization")
        left_layout = QVBoxLayout(left_group)
        selector_grid = QGridLayout()
        selector_grid.addWidget(QLabel("Context Maps"), 0, 0)
        selector_grid.addWidget(QLabel("Base Maps"), 0, 1)
        selector_grid.addWidget(QLabel("Map Source"), 0, 2)
        selector_grid.addWidget(self.context_map_combo, 1, 0)
        selector_grid.addWidget(self.bottom_map_combo, 1, 1)
        selector_grid.addWidget(self.map_source_combo, 1, 2)
        left_layout.addLayout(selector_grid)
        self.map_box_widget = MapBoxDisplayWidget()
        self.map_box_widget.set_entry_box_path(self._entry_box_path)
        left_layout.addWidget(self.map_box_widget, stretch=1)
        body.addWidget(left_group, 0, 0)
        body.setColumnStretch(0, 3)
        body.setColumnStretch(1, 2)

        right_group = QGroupBox("Geometry")
        right_layout = QVBoxLayout(right_group)
        self.coord_mode_combo = QComboBox()
        self.coord_mode_combo.addItems([m.value for m in CoordMode])

        self.coord_x_edit = QLineEdit()
        self.coord_y_edit = QLineEdit()
        self.grid_x_edit = QLineEdit()
        self.grid_y_edit = QLineEdit()
        self.grid_z_edit = QLineEdit()
        self.res_edit = QLineEdit()
        self.fov_x_edit = QLineEdit()
        self.fov_y_edit = QLineEdit()
        self.fov_w_edit = QLineEdit()
        self.fov_h_edit = QLineEdit()
        self.square_fov_box = QCheckBox()
        self.coord_mode_combo.setMaximumWidth(72)
        compact_fields = (
            self.coord_x_edit,
            self.coord_y_edit,
            self.grid_x_edit,
            self.grid_y_edit,
            self.grid_z_edit,
            self.res_edit,
            self.fov_x_edit,
            self.fov_y_edit,
            self.fov_w_edit,
            self.fov_h_edit,
        )
        for edit in compact_fields:
            edit.setMaximumWidth(92)

        geometry_grid = QGridLayout()
        geometry_grid.setHorizontalSpacing(6)
        geometry_grid.setVerticalSpacing(4)
        geometry_grid.setColumnMinimumWidth(2, 16)
        geometry_grid.addWidget(QLabel("Box"), 0, 0, 1, 2, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        geometry_grid.addWidget(QLabel("FOV"), 0, 3, 1, 2, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        geometry_grid.addWidget(QLabel("mode"), 1, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        geometry_grid.addWidget(self.coord_mode_combo, 1, 1, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        geometry_grid.addWidget(QLabel("box_xc"), 2, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        geometry_grid.addWidget(self.coord_x_edit, 2, 1, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        geometry_grid.addWidget(QLabel("fov_xc"), 2, 3, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        geometry_grid.addWidget(self.fov_x_edit, 2, 4, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        geometry_grid.addWidget(QLabel("box_yc"), 3, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        geometry_grid.addWidget(self.coord_y_edit, 3, 1, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        geometry_grid.addWidget(QLabel("fov_yc"), 3, 3, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        geometry_grid.addWidget(self.fov_y_edit, 3, 4, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        geometry_grid.addWidget(QLabel("grid_nx"), 4, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        geometry_grid.addWidget(self.grid_x_edit, 4, 1, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        geometry_grid.addWidget(QLabel("fov_xsize"), 4, 3, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        geometry_grid.addWidget(self.fov_w_edit, 4, 4, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        geometry_grid.addWidget(QLabel("grid_ny"), 5, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        geometry_grid.addWidget(self.grid_y_edit, 5, 1, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        geometry_grid.addWidget(QLabel("fov_ysize"), 5, 3, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        geometry_grid.addWidget(self.fov_h_edit, 5, 4, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        geometry_grid.addWidget(QLabel("grid_nz"), 6, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        geometry_grid.addWidget(self.grid_z_edit, 6, 1, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        geometry_grid.addWidget(QLabel("Square FOV"), 6, 3, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        geometry_grid.addWidget(self.square_fov_box, 6, 4, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        geometry_grid.addWidget(QLabel("dx_km"), 7, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        geometry_grid.addWidget(self.res_edit, 7, 1, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        geometry_grid.setColumnStretch(1, 1)
        geometry_grid.setColumnStretch(4, 1)
        right_layout.addLayout(geometry_grid)

        self.selector_status_label = QPlainTextEdit()
        self.selector_status_label.setReadOnly(True)
        self.selector_status_label.setMaximumHeight(110)
        status_group = QGroupBox("Interaction Status")
        status_layout = QVBoxLayout(status_group)
        status_layout.addWidget(self.selector_status_label)
        right_layout.addWidget(status_group)

        self.map_meta_label = QPlainTextEdit()
        self.map_meta_label.setReadOnly(True)
        meta_group = QGroupBox("Selected Map Metadata")
        meta_layout = QVBoxLayout(meta_group)
        meta_layout.addWidget(self.map_meta_label)
        right_layout.addWidget(meta_group, stretch=1)
        body.addWidget(right_group, 0, 1)

        for edit in (
            self.coord_x_edit,
            self.coord_y_edit,
            self.grid_x_edit,
            self.grid_y_edit,
            self.grid_z_edit,
            self.res_edit,
            self.fov_x_edit,
            self.fov_y_edit,
            self.fov_w_edit,
            self.fov_h_edit,
        ):
            edit.editingFinished.connect(self._push_form_to_view_state)
            edit.installEventFilter(self)

        self.coord_mode_combo.currentTextChanged.connect(self._push_form_to_view_state)
        self.square_fov_box.toggled.connect(self._on_square_fov_toggled)
        self.map_box_widget.set_geometry_change_callback(self._apply_selection_to_form)
        self.map_box_widget.set_fov_change_callback(self._apply_fov_to_form)
        self.map_box_widget.set_map_info_callback(self._on_map_info_changed)
        self.map_box_widget.set_status_callback(self._on_selector_status_changed)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        self._ok_button = buttons.button(QDialogButtonBox.Ok)
        self._cancel_button = buttons.button(QDialogButtonBox.Cancel)
        if self._ok_button is not None:
            self._ok_button.setText("Apply && Close")
            self._ok_button.setDefault(False)
            self._ok_button.setAutoDefault(False)
        if self._cancel_button is not None:
            self._cancel_button.setText("Close")
            self._cancel_button.setDefault(False)
            self._cancel_button.setAutoDefault(False)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self._open_3d_button = QPushButton("Open 3D Viewer")
        self._open_3d_button.clicked.connect(self.map_box_widget.open_live_3d_viewer)
        button_row = QHBoxLayout()
        button_row.addWidget(self._open_3d_button)
        button_row.addStretch()
        button_row.addWidget(buttons)
        root.addLayout(button_row)
        self.map_box_widget.set_action_state_callback(self._on_map_action_state_changed)

    def _load_session_input(self, session_input: SelectorSessionInput) -> None:
        self.map_box_widget.initialize(session_input)
        self.map_box_widget.set_map_file_paths(session_input.map_files or {})
        self.map_box_widget.set_geometry_edit_enabled(bool(session_input.allow_geometry_edit))

        state = self.map_box_widget.state()
        context_ids = self._context_map_ids(session_input)
        bottom_ids = self._bottom_map_ids(session_input)

        self.context_map_combo.blockSignals(True)
        self.context_map_combo.clear()
        for map_id in context_ids:
            self.context_map_combo.addItem(self._context_display_text(map_id), map_id)
        if state is not None and state.selected_context_id:
            idx = self.context_map_combo.findData(state.selected_context_id)
            if idx >= 0:
                self.context_map_combo.setCurrentIndex(idx)
            elif context_ids:
                self.context_map_combo.setCurrentIndex(0)
        self.context_map_combo.blockSignals(False)

        self.bottom_map_combo.blockSignals(True)
        self.bottom_map_combo.clear()
        self.bottom_map_combo.addItem("none", None)
        for map_id in bottom_ids:
            self.bottom_map_combo.addItem(map_id, map_id)
        if state is not None and state.selected_bottom_id:
            idx = self.bottom_map_combo.findData(state.selected_bottom_id)
            if idx >= 0:
                self.bottom_map_combo.setCurrentIndex(idx)
            else:
                self.bottom_map_combo.setCurrentIndex(0)
                self.map_box_widget.set_bottom_map_id(None)
        else:
            self.bottom_map_combo.setCurrentIndex(0)
            self.map_box_widget.set_bottom_map_id(None)
        self.bottom_map_combo.blockSignals(False)

        self.map_source_combo.blockSignals(True)
        idx = self.map_source_combo.findData(str(session_input.map_source_mode or "auto"))
        self.map_source_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.map_source_combo.blockSignals(False)

        self._apply_selection_to_form(session_input.geometry)
        if state is not None and state.fov is not None:
            self._apply_fov_to_form(state.fov)
        else:
            inferred_fov = self.map_box_widget.current_fov_selection() or self.map_box_widget.projected_box_fov()
            if inferred_fov is not None:
                self.map_box_widget.set_fov_selection(inferred_fov)
        self.square_fov_box.blockSignals(True)
        self.square_fov_box.setChecked(bool(session_input.square_fov))
        self.square_fov_box.blockSignals(False)
        self.map_box_widget.set_square_fov(bool(session_input.square_fov))
        self._apply_square_fov_state_to_form()
        self._apply_geometry_edit_enabled_state(bool(session_input.allow_geometry_edit))

    def _on_map_action_state_changed(self, can_open_3d: bool, _can_clear_lines: bool) -> None:
        self._open_3d_button.setEnabled(bool(can_open_3d))

    @staticmethod
    def _context_map_ids(session_input: SelectorSessionInput) -> list[str]:
        base_ids = list(session_input.map_ids or [])
        bottom_only = {"Vert_current", "chromo_mask", "Bx", "By"}
        preferred = ["94", "131", "1600", "1700", "171", "193", "211", "304", "335", "Bz", "Ic", "Br", "Bp", "Bt"]
        allowed = [map_id for map_id in base_ids if map_id not in bottom_only]
        ordered = [map_id for map_id in preferred if map_id in allowed]
        ordered.extend(map_id for map_id in allowed if map_id not in ordered)
        return ordered

    @staticmethod
    def _bottom_map_ids(session_input: SelectorSessionInput) -> list[str]:
        base_maps = dict(session_input.base_maps or {})
        out: list[str] = []

        base_group = [("Bx", "bx"), ("By", "by"), ("Bz", "bz"), ("Vert_current", "vert_current"), ("chromo_mask", "chromo_mask")]
        for display_id, base_key in base_group:
            if base_key in base_maps and display_id not in out:
                out.append(display_id)
        if "Vert_current" in (session_input.map_ids or ()) and "Vert_current" not in out:
            out.append("Vert_current")
        return out

    @staticmethod
    def _context_display_text(map_id: str) -> str:
        return "Blos" if map_id == "Bz" else map_id

    def _on_context_map_changed(self, _index: int) -> None:
        map_id = self.context_map_combo.currentData()
        self.map_box_widget.set_context_map_id(str(map_id) if map_id else None)

    def _on_bottom_map_changed(self, _index: int) -> None:
        map_id = self.bottom_map_combo.currentData()
        self.map_box_widget.set_bottom_map_id(str(map_id) if map_id else None)

    def _on_map_source_changed(self, _index: int) -> None:
        mode = self.map_source_combo.currentData()
        self.map_box_widget.set_map_source_mode(str(mode or "auto"))

    def _on_map_info_changed(self, text: str) -> None:
        self.map_meta_label.setPlainText(text)

    def _on_selector_status_changed(self, text: str) -> None:
        self.selector_status_label.setPlainText(text)

    def _apply_selection_to_form(self, selection: BoxGeometrySelection) -> None:
        self.coord_mode_combo.blockSignals(True)
        self.coord_mode_combo.setCurrentText(selection.coord_mode.value)
        self.coord_mode_combo.blockSignals(False)

        fields = selection.as_gui_text_fields()
        self.coord_x_edit.setText(fields["coord_x_edit"])
        self.coord_y_edit.setText(fields["coord_y_edit"])
        self.grid_x_edit.setText(fields["grid_x_edit"])
        self.grid_y_edit.setText(fields["grid_y_edit"])
        self.grid_z_edit.setText(fields["grid_z_edit"])
        self.res_edit.setText(fields["res_edit"])

    def _push_form_to_view_state(self) -> None:
        try:
            if self.square_fov_box.isChecked():
                self.fov_h_edit.setText(self.fov_w_edit.text())
            selection = self._selection_from_form() if self._session_input.allow_geometry_edit else None
            fov = self._fov_from_form()
        except ValueError:
            return
        if selection is not None:
            self.map_box_widget.set_geometry_selection(selection)
        self.map_box_widget.set_fov_selection(fov)

    def _apply_geometry_edit_enabled_state(self, enabled: bool) -> None:
        editable = bool(enabled)
        self.coord_mode_combo.setEnabled(editable)
        for widget in (
            self.coord_x_edit,
            self.coord_y_edit,
            self.grid_x_edit,
            self.grid_y_edit,
            self.grid_z_edit,
            self.res_edit,
        ):
            widget.setEnabled(editable)

    def _apply_square_fov_state_to_form(self) -> None:
        square = self.square_fov_box.isChecked()
        if square:
            self.fov_h_edit.setText(self.fov_w_edit.text())
        self.fov_h_edit.setEnabled(not square)

    def _on_square_fov_toggled(self, checked: bool) -> None:
        self.map_box_widget.set_square_fov(bool(checked))
        self._apply_square_fov_state_to_form()
        self._push_form_to_view_state()

    def _selection_from_form(self) -> BoxGeometrySelection:
        return BoxGeometrySelection(
            coord_mode=CoordMode(self.coord_mode_combo.currentText()),
            coord_x=float(self.coord_x_edit.text()),
            coord_y=float(self.coord_y_edit.text()),
            grid_x=int(float(self.grid_x_edit.text())),
            grid_y=int(float(self.grid_y_edit.text())),
            grid_z=int(float(self.grid_z_edit.text())),
            dx_km=float(self.res_edit.text()),
        )

    def _fov_from_form(self) -> DisplayFovSelection:
        return DisplayFovSelection(
            center_x_arcsec=float(self.fov_x_edit.text()),
            center_y_arcsec=float(self.fov_y_edit.text()),
            width_arcsec=max(1e-3, float(self.fov_w_edit.text())),
            height_arcsec=max(1e-3, float(self.fov_h_edit.text())),
        )

    def _apply_fov_to_form(self, selection: DisplayFovSelection) -> None:
        fields = selection.as_gui_text_fields()
        self.fov_x_edit.setText(fields["fov_x_edit"])
        self.fov_y_edit.setText(fields["fov_y_edit"])
        self.fov_w_edit.setText(fields["fov_w_edit"])
        self.fov_h_edit.setText(fields["fov_w_edit"] if self.square_fov_box.isChecked() else fields["fov_h_edit"])

    def eventFilter(self, obj, event):
        # Prevent Enter/Return in input fields from closing the dialog through the default button.
        # Instead, treat Enter as "apply field edit" only.
        if obj in {
            self.coord_x_edit,
            self.coord_y_edit,
            self.grid_x_edit,
            self.grid_y_edit,
            self.grid_z_edit,
            self.res_edit,
            self.fov_x_edit,
            self.fov_y_edit,
            self.fov_w_edit,
            self.fov_h_edit,
        } and event.type() == QEvent.KeyPress and event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._push_form_to_view_state()
            return True
        return super().eventFilter(obj, event)

    def accept(self) -> None:
        self._accepted_selection = SelectorDialogResult(
            geometry=self._selection_from_form(),
            fov=self._fov_from_form(),
            square_fov=bool(self.square_fov_box.isChecked()),
        )
        super().accept()

    def accepted_selection(self) -> Optional[SelectorDialogResult]:
        return self._accepted_selection

    def committed_line_seeds(self):
        return self.map_box_widget.committed_line_seeds()

    def current_fov_box_selection(self):
        return self.map_box_widget.current_fov_box_selection()


def run_fov_box_selector(
    session_input: SelectorSessionInput,
    parent: Optional[QWidget] = None,
    entry_box_path: Optional[str | Path] = None,
) -> Optional[SelectorDialogResult]:
    dialog = FovBoxSelectorDialog(session_input=session_input, parent=parent, entry_box_path=entry_box_path)
    if dialog.exec_() == QDialog.Accepted:
        return dialog.accepted_selection()
    return None
