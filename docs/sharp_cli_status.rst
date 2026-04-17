SHARP CLI Status
================

This note summarizes the current SHARP-based CLI workflow added to ``pyAMPP``
and the runtime status as of 2026-04-16.

Implemented
-----------

- New CLI-first SHARP workflow via ``pyampp-sharp-sim``.
- New SHARP subpackage under ``pyampp/sharp/``:

  - ``drms.py``: nearest-time ``hmi.sharp_cea_720s`` query and segment export.
  - ``entry_box.py``: SHARP CEA patch to ``NONE`` entry-box HDF5 conversion.
  - ``radio_adapter.py``: CHR-stage HDF5 to radio-synthesis model adapter.
  - ``cli.py``: one-shot pipeline driver.

- New packaging/script support in ``pyproject.toml``:

  - ``PyYAML`` added to dependencies.
  - ``pyampp-sharp-sim`` added to ``[project.scripts]``.

Behavior
--------

- Input source is ``hmi.sharp_cea_720s`` only.
- The CLI selects the globally nearest available ``T_REC`` within the allowed
  time window and processes all matching HARPNUMs, or only the requested
  ``--harpnum`` values.
- SHARP CEA boundary fields are mapped as:

  - ``bx = Bp``
  - ``by = -Bt``
  - ``bz = Br``

- Native SHARP horizontal sampling is preserved.
- Vertical depth is controlled by ``--nz``.
- If ``--freq-ghz`` is omitted, frequencies default to the axis stored in
  ``ff2gr_starter/example_data/tot_flux_2022_04_25.fits``.

Compatibility Fixes Added During Validation
-------------------------------------------

- Deferred ``sunpy.net.Fido`` import in ``pyampp/data/downloader.py`` to avoid
  import-time failures in headless runs.
- Added ``chromo_mask == 0`` fallback handling in:

  - ``pyampp/gx_chromo/decompose.py``
  - ``pyampp/gx_chromo/combo_model.py``

- Extended ``pyampp/gxbox/boxutils.py`` loader fallback to tolerate missing
  WCS unit metadata when building SunPy maps from SHARP-related FITS.
- Added SHARP non-WCS segment loading fallback in ``pyampp/sharp/entry_box.py``
  so ``continuum``, ``magnetogram``, and ``bitmap`` can inherit the ``Br`` WCS
  when they are on the same pixel grid but do not carry their own WCS headers.
- Refined ``pyampp/sharp/radio_adapter.py`` to correctly reconstruct the legacy
  radio model payload from CHR-stage HDF5 line products.

Validated Cases
---------------

Completed end-to-end real runs:

- ``2022-04-25T12:00:00``:

  - ``HARP8175`` with ``--use-potential`` for a small fast smoke test.
  - ``HARP8150`` with ``--use-potential`` for a real NOAA active region.

- ``2021-01-21T20:00:05``:

  - ``HARP7536`` with ``--use-potential`` against
    ``ff2gr_starter/example_data/tot_flux_2021_01_21.fits``.

Partial deeper NLFFF runs:

- ``2021-01-21T20:00:05`` / ``HARP7536`` / ``nz=32``:

  - ``NONE``, ``POT``, ``BND``, and ``NAS`` completed successfully.
  - ``NAS.GEN`` remained the dominant bottleneck and did not finish within the
    interactive session.

- ``2021-01-21T20:00:05`` / ``HARP7536`` / ``nz=16``:

  - Reached the deeper NLFFF path successfully but was stopped while still in
    the magnetic solve/tracing chain.

Current Limitation
------------------

- The main runtime bottleneck is the native line-tracing stage used for
  ``NAS.GEN``.
- The hot path is not Python; it is the compiled ``pyAMaFiL`` tracer
  ``mfoGetLines`` inside ``WWNLFFFReconstruction``.
- A move to a multi-core server should help overall throughput, but the current
  CLI does not yet expose a dedicated line-tracing process-count parameter.

Recommended Server Tests
------------------------

Fast end-to-end validation:

.. code-block:: bash

   pyampp-sharp-sim \
     --time 2021-01-21T20:00:05 \
     --harpnum 7536 \
     --nz 8 \
     --use-potential \
     --mw-lib /path/to/RenderGRFF_arm64.dylib \
     --ebtel /path/to/ebtel.sav \
     --radio-config /path/to/example_radio_config.yaml \
     --out-root /path/to/sharp_runs

First NLFFF validation:

.. code-block:: bash

   pyampp-sharp-sim \
     --time 2021-01-21T20:00:05 \
     --harpnum 7536 \
     --nz 16 \
     --mw-lib /path/to/RenderGRFF_arm64.dylib \
     --ebtel /path/to/ebtel.sav \
     --radio-config /path/to/example_radio_config.yaml \
     --out-root /path/to/sharp_runs

Deeper NLFFF stress test:

.. code-block:: bash

   pyampp-sharp-sim \
     --time 2021-01-21T20:00:05 \
     --harpnum 7536 \
     --nz 32 \
     --mw-lib /path/to/RenderGRFF_arm64.dylib \
     --ebtel /path/to/ebtel.sav \
     --radio-config /path/to/example_radio_config.yaml \
     --out-root /path/to/sharp_runs
