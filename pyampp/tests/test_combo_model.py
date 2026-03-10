from __future__ import annotations

import numpy as np

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
