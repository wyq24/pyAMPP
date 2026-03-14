from __future__ import annotations

"""Compatibility wrapper for the SAV->HDF5 converter.

The implementation now lives in ``pyampp.util.build_h5_from_sav`` so runtime
code does not depend on the ``pyampp.tests`` package.
"""

from pyampp.util.build_h5_from_sav import build_h5_from_sav, main

__all__ = ["build_h5_from_sav", "main"]


if __name__ == "__main__":
    main()
