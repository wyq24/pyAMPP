# GxBox / pyAMPP Implementation Notes (2026-02-28)

This note records the GUI and pipeline fixes implemented during the February 28, 2026 session.

## Scope

The work focused on the `gx-fov2box` resume path and the `pyAMPP` GUI workflow around stop-stage runs, viewer launch behavior, and continue-from-entry command generation.

## Implemented Changes

### 1. Fixed HDF5 resume axis-order handling for POT -> BND/NAS continuation

Files:
- `pyampp/gxbox/gx_fov2box.py`

Problem:
- Resuming from an HDF5 entry box could misread `axis_order_3d` because the stored metadata value was bytes (for example `b'zyx'`) while the code compared it to the string `"zyx"`.
- This caused the resume path to skip the H5 `(z, y, x)` to internal `(x, y, z)` transpose.
- It also caused wrong `box_dims` inference from the entry cube.
- The direct symptom was a broadcast error in `_make_bnd_from_pot()` when a `(150, 100)` base map was written into a `(100, 100)` boundary slice.

Fix:
- Decode `axis_order_3d` via `_decode_id_text(...).lower()` before:
  - deciding whether to transpose H5 corona cubes back to internal layout
  - inferring `box_dims` from an HDF5 entry box

Result:
- Resume from saved POT HDF5 files now preserves the correct internal cube orientation and no longer fails with the previous BND boundary broadcast mismatch.

### 2. Hardened `gxbox-view2d` launch reporting in the GUI

Files:
- `pyampp/gxbox/gxampp.py`

Problem:
- The GUI logged `Launched gxbox-view2d with: ...` immediately after `subprocess.Popen(...)`.
- If the viewer exited during startup, the log still claimed success and gave no useful failure signal.

Fix:
- Added a launch-pending state for the 2D viewer process.
- Changed the log flow to:
  - `Starting gxbox-view2d for: ...` immediately after spawn
  - `Launched gxbox-view2d with: ...` only after the process is still alive on the next timer poll
- If the viewer exits during startup, the GUI now logs:
  - the startup exit code
  - any captured stdout/stderr text

Result:
- The GUI now distinguishes between "spawned" and "actually stayed alive".
- Viewer-launch failures are visible in the console instead of being misreported as successful launches.

### 3. Made final CHR runs explicit in GUI-generated commands

Files:
- `pyampp/gxbox/gxampp.py`

Problem:
- The GUI treated CHR as an implicit final save, but the generated CLI command did not explicitly include:
  - `--save-chr`
  - `--stop-after chr`
- Because of that, the process monitor only recognized explicit stop stages (`dl`, `none`, `pot`, `nas`, `gen`) and did not reliably treat the final CHR pass as a stop-stage completion.
- This prevented the automatic post-run `gxbox-view2d` launch after final CHR.

Fix:
- The GUI command builder now always computes a final target stage.
- For the default full pipeline path, the final stage is now emitted explicitly as:
  - `--save-chr`
  - `--stop-after chr`

Result:
- Final CHR is now an explicit stop-stage from the GUI's perspective.
- The post-run GUI monitor correctly launches `gxbox-view2d` after a successful CHR completion.

### 4. Prevented redundant GEN recomputation when continuing from a GEN entry box to CHR

Files:
- `pyampp/gxbox/gxampp.py`

Problem:
- Continuing from a `NAS.GEN.h5` entry box and proceeding to CHR could recompute `NAS.GEN` even though valid line metadata already existed in the entry model.
- This happened because the GUI did not add `--jump2chromo`, so the CLI default target-stage detection could still route through GEN-stage behavior.

Fix:
- In `Continue` mode, when:
  - an entry box exists
  - the detected entry stage is `GEN` or `CHR`
  - the final target is CHR
- the GUI now appends:
  - `--jump2chromo`

Result:
- Continuing from `NAS.GEN.h5` to final CHR reuses the existing lines metadata.
- The CHR stage now completes directly without recomputing GEN.

### 5. Aligned checkbox state with the real command path for GEN/CHR entry boxes

Files:
- `pyampp/gxbox/gxampp.py`

Problem:
- After adopting a `GEN` entry box in `Continue` mode, the command line at the bottom correctly represented a direct CHR completion path, but some GEN-only controls still appeared active or available by default.
- This created a mismatch between the visible command and the workflow checkboxes.

Fix:
- In `Continue` mode from a `GEN` or `CHR` entry box, when no earlier stop-stage is selected:
  - `Save Lines (GEN)` is cleared and disabled
  - `Skip Line Computation` is cleared and disabled
  - `center-vox` is cleared and disabled
- `Stop after GEN` remains available so the user can explicitly request a same-stage rerun.

Result:
- The default checkbox state now reflects the actual direct-to-CHR command path shown in the command preview.
- The user can still opt into a GEN rerun deliberately, but it is no longer implied by the default UI state.

## Verified Outcomes

Observed during the session:
- POT stop-stage runs launched `gxbox-view2d` and were adopted back into the GUI as the next continue entry.
- NAS stop-stage runs completed and launched `gxbox-view2d`.
- Continuing from `NAS.GEN.h5` to CHR now emitted:
  - `--save-chr --stop-after chr --jump2chromo`
- That final CHR run completed without recomputing GEN.
- The final CHR run launched `gxbox-view2d` and was adopted back into the GUI.

## Practical Effect

The resume-and-continue workflow is now coherent across these stages:
- POT
- NAS
- NAS.GEN
- NAS.GEN.CHR

Specifically:
- HDF5 resume uses the correct cube orientation
- stop-stage detection includes CHR
- final CHR runs are explicit
- GEN line metadata is reused when appropriate
- the GUI state better matches the actual command it will run

## Notes

- Existing in-progress documentation edits elsewhere in the repository were left untouched.
- This note was added as a standalone session record to avoid interfering with other doc work already present in the worktree.
