from __future__ import annotations

import numpy as np

import pyampp.gx_chromo.combo_model as combo_model_module
from pyampp.gx_chromo.combo_model import combo_model


def test_combo_model_accepts_3d_field_cubes():
    nx, ny, nz = 2, 3, 4
    box = {
        "bx": np.arange(nx * ny * nz, dtype=np.float32).reshape(nx, ny, nz),
        "by": (100.0 + np.arange(nx * ny * nz, dtype=np.float32)).reshape(nx, ny, nz),
        "bz": (200.0 + np.arange(nx * ny * nz, dtype=np.float32)).reshape(nx, ny, nz),
    }
    dr = np.array([0.1, 0.1, 0.1], dtype=np.float64)
    chromo_mask = np.ones((nx, ny), dtype=np.int32)

    chromo_box = combo_model(
        box,
        dr,
        np.zeros((ny, nx), dtype=np.float32),
        np.ones((ny, nx), dtype=np.float32),
        chromo_mask=chromo_mask,
    )

    assert chromo_box["bcube"].shape == (nx, ny, nz, 3)
    np.testing.assert_allclose(chromo_box["bcube"][:, :, :, 0], box["bx"][:, :, ::-1])
    np.testing.assert_allclose(chromo_box["bcube"][:, :, :, 1], box["by"][:, :, ::-1])
    np.testing.assert_allclose(chromo_box["bcube"][:, :, :, 2], box["bz"][:, :, ::-1])


def test_combo_model_uses_supplied_chromo_mask(monkeypatch):
    called = False

    def _fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("decompose() should not be called when chromo_mask is supplied")

    monkeypatch.setattr(combo_model_module, "decompose", _fail_if_called)

    box = {
        "bx": np.zeros((2, 2, 2), dtype=np.float32),
        "by": np.zeros((2, 2, 2), dtype=np.float32),
        "bz": np.zeros((2, 2, 2), dtype=np.float32),
    }
    dr = np.array([0.1, 0.1, 0.1], dtype=np.float64)
    chromo_mask = np.ones((2, 2), dtype=np.int32)

    chromo_box = combo_model(
        box,
        dr,
        np.zeros((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=np.float32),
        chromo_mask=chromo_mask,
    )

    assert called is False
    np.testing.assert_array_equal(chromo_box["chromo_mask"], chromo_mask)
