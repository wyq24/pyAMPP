Model Viewers
=============

gxbox-view3d
------------

``gxbox-view3d`` opens an existing model HDF5 file in the 3D viewer without recomputing.

Legacy alias:

- ``gxbox-view`` (kept for backward compatibility)

Expected inputs:

- A model file containing at least one of:
  - ``corona`` (preferred),
  - ``nlfff`` / ``pot`` (accepted and normalized), or
  - ``chromo`` with sufficient magnetic cube fields.

Usage:

.. code-block:: bash

   gxbox-view3d /path/to/model.h5

Optional file picker mode:

.. code-block:: bash

   gxbox-view3d --pick

gxbox-view2d
------------

``gxbox-view2d`` opens the 2D FOV / box selector from an existing ``.h5`` or ``.sav`` box file.

Behavior:

- loads geometry, stage metadata, and available embedded/base/refmap context from the selected box
- persists updated observer/FOV metadata back to ``.h5`` boxes when the dialog is accepted
- is used by the main ``pyampp`` GUI after explicit stop-stage completions (POT, NAS, GEN, CHR)

Legacy alias:

- ``gxbox-select`` (kept for backward compatibility)

Usage:

.. code-block:: bash

   gxbox-view2d /path/to/model.h5

Optional file picker mode:

.. code-block:: bash

   gxbox-view2d --pick

gxrefmap-view
-------------

``gxrefmap-view`` is a 2D map browser for base maps and refmaps stored in model HDF5 files.

Expected layout:

- Base maps:
  - ``base/bx``, ``base/by``, ``base/bz``, ``base/ic``, ``base/chromo_mask``
  - base header in ``base/index`` (fallbacks: ``base/index_header``, ``base/wcs_header``)
- Refmaps:
  - ``refmaps/<map_id>/data``
  - ``refmaps/<map_id>/wcs_header``

Usage:

.. code-block:: bash

   gxrefmap-view /path/to/model.h5

List available maps only:

.. code-block:: bash

   gxrefmap-view /path/to/model.h5 --list

Start from a specific map:

.. code-block:: bash

   gxrefmap-view /path/to/model.h5 --start AIA_94

h5tree
------

``h5tree`` prints a compact tree of HDF5 groups/datasets and their shapes/dtypes.

Metadata behavior:

- Metadata values (``metadata/*``) are shown by default.
- Use ``--no-metadata`` to suppress metadata value lines.
- Use ``--meta`` to print only metadata values (no tree output).

Usage:

.. code-block:: bash

   h5tree /path/to/model.h5

Suppress metadata values:

.. code-block:: bash

   h5tree /path/to/model.h5 --no-metadata

Metadata-only mode:

.. code-block:: bash

   h5tree /path/to/model.h5 --meta
