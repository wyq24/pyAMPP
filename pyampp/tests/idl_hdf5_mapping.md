# pyAMPP HDF5 Stage Contract (Review Draft, Pre-Implementation)

Status
- Review document only.  
- No additional refactor should be done until this contract is approved.
- GX/IDL semantics are the physics reference.
- pyAMPP internal/API conventions are Python-first.

---

## 1) Canonical Axis Contract

This document defines the **pyAMPP canonical in-memory and HDF5 axis order**.

Canonical pyAMPP order:
- 2D maps: `(ny, nx)`
- 3D scalar volumes: `(nz, ny, nx)`
- 3D vectors (component-last): `(nz, ny, nx, 3)` where `[..., 0]=Bx`, `[..., 1]=By`, `[..., 2]=Bz`

Rationale:
- Python/Numpy ecosystem compatibility.
- Fewer user-facing transposes in Python applications.
- Explicit adapter transforms at IDL/C++ boundaries instead of mixed internal conventions.

Important:
- `grid/voxel_id` follows scalar-volume order: `(nz, ny, nx)` in this draft.
- `grid/dz` follows scalar-volume order when 3D: `(nz, ny, nx)`.

---

## 2) Boundary Adapter Rules (Required)

These are adapter responsibilities, not H5 contract exceptions.

IDL SAV (from `readsav`) -> pyAMPP canonical:
- IDL `BCUBE` semantic order is `(nx, ny, nz, 3)`.
- `readsav` commonly exposes it as `(3, nz, ny, nx)`.
- Adapter must convert to canonical `(nz, ny, nx, 3)`.
- Same principle for `CHROMO_BCUBE`.

pyAMPP canonical -> C/C++ renderer:
- Convert explicitly to renderer-expected layout at call boundary only.
- Never leak renderer layout back into pyAMPP H5/API storage.

Rule:
- Internal/H5 stays canonical.
- IDL and renderer are treated as import/export interfaces with explicit transforms.

---

## 3) Stage Namespace Contract

Stage groups:
- `NONE`: `base`, `refmaps`, `metadata`, `grid`
- `BND`: `base`, `refmaps`, `metadata`, `grid`, `corona`
- `POT`: `base`, `refmaps`, `metadata`, `grid`, `corona`
- `NAS`: `base`, `refmaps`, `metadata`, `grid`, `corona`
- `NAS.GEN`: `base`, `refmaps`, `metadata`, `grid`, `corona`, `lines`
- `NAS.CHR`: `base`, `refmaps`, `metadata`, `grid`, `corona`, `lines`, `chromo`

Notes:
- No `bounds/` namespace. BND uses `corona/`.
- `chromo/` appears only at CHR stage.
- `lines/` appears at GEN and is preserved in CHR.

---

## 4) Common Groups

### 4.1 `base/` (all stages)
- `bx`: `(ny, nx)` float
- `by`: `(ny, nx)` float
- `bz`: `(ny, nx)` float
- `ic`: `(ny, nx)` float
- `chromo_mask`: `(ny, nx)` int
- `index`: IDL-compatible index metadata payload

### 4.2 `refmaps/` (all stages)
- per map entry:
  - `data`: `(ny, nx)` map-native 2D shape
  - `wcs_header`: map WCS metadata payload
  - `order_index` (attribute): integer insertion order used to preserve IDL map ordering

### 4.3 `metadata/` (all stages)
- `id`
- `execute`
- `projection`
- `disambiguation`
- recommended: `stage`

### 4.4 `grid/` (all stages)
- `dx`: scalar
- `dy`: scalar
- `dz`: scalar or 3D `(nz, ny, nx)` depending on stage
- `voxel_id`: `(nz, ny, nx)` uint32

---

## 5) Stage-Specific Content

## 5.1 NONE (`*.NONE.h5`)
- No `corona/`, no `lines/`, no `chromo/`.
- `grid/voxel_id` policy: allowed if computed; otherwise optional for NONE.

## 5.2 BND (`*.BND.h5`)
`corona/` required:
- `bx`: `(nz, ny, nx)` float
- `by`: `(nz, ny, nx)` float
- `bz`: `(nz, ny, nx)` float
- `dr`: `(3,)` float
- `corona_base`: scalar int (explicit)

## 5.3 POT (`*.POT.h5`)
Same required `corona/` contract as BND.

## 5.4 NAS (`*.NAS.h5`)
Same required `corona/` contract as BND/POT.

## 5.5 NAS.GEN (`*.NAS.GEN.h5`)
`corona/` preserved, plus `lines/`:
- `start_idx`: 1D int
- `end_idx`: 1D int
- `seed_idx`: 1D int
- `apex_idx`: 1D int
- `codes`: 1D int
- `voxel_status`: 1D uint8
- `av_field`: 1D float
- `phys_length`: 1D float
  Unit: solar radii in saved `NAS.GEN`/`NAS.CHR` products.
  The raw DLL `VPHYSLENGTH` output is in voxel-length units and must be
  multiplied by `dr[0]` before being persisted as `lines/phys_length`.

## 5.6 NAS.CHR (`*.NAS.CHR.h5`)
`corona/` + `lines/` preserved, plus `chromo/`:
- `chromo_idx`: 1D int
- `chromo_n`: 1D float
- `chromo_t`: 1D float
- `n_p`: 1D float
- `n_hi`: 1D float
- `n_htot`: 1D float
- `tr`: `(ny, nx)` int
- `tr_h`: `(ny, nx)` float
- `chromo_layers`: scalar int
- `dz`: `(nz, ny, nx)` float (final CHR geometry)
- `bx`: `(nz, ny, nx)` float (CHR vector field)
- `by`: `(nz, ny, nx)` float
- `bz`: `(nz, ny, nx)` float

Clarification:
- `chromo/bx|by|bz` are CHR fields; they are not aliases of `corona/bx|by|bz`.
- `corona/corona_base` remains under `corona/` at CHR too.
- `lines/phys_length` in saved HDF5 and IDL `GEN/CHR` products is stored in
  solar-radius units, not centimeters. Consumers that need cgs length must
  convert via `phys_length * RSun`.

---

## 6) IDL Semantic Mapping (Field-Level, Not Axis-Level)

IDL semantic source -> H5 target namespace:
- `BOX.BASE.*` -> `base/*`
- `BOX.REFMAPS` -> `refmaps/*`
- `BOX.BCUBE` components -> `corona/bx|by|bz`
- line-tracing arrays (`STARTIDX`, `ENDIDX`, `STATUS`, `AVFIELD`, `PHYSLENGTH`, etc.) -> `lines/*`
- CHR arrays (`CHROMO_IDX`, `CHROMO_N`, `CHROMO_T`, `N_P`, `N_HI`, `N_HTOT`, `TR`, `TR_H`, `CHROMO_LAYERS`, `CHROMO_BCUBE`) -> `chromo/*`
- voxel-id derived outputs -> `grid/voxel_id` and `corona/corona_base`

Axis conversion is handled by adapters, not by changing semantic mapping.

---

## 7) Invariants (Must Pass)

For each stage where `corona/` exists:
1. `base/bz` must match `corona/bz` at `z=0` within numerical tolerance.
2. `base/bx` and `base/by` must match `corona/bx` and `corona/by` at `z=0` after only the documented adapter transform (no hidden heuristic swaps).
3. `grid/voxel_id` and `corona/corona_base` must be consistent with `gx_box2id` logic.

For SAV internal checks:
4. `BOX.BASE.{BX,BY,BZ}` must match `BOX.BCUBE` at `z=0` (within floating precision) after declared adapter mapping.

---

## 8) Open Items For Review (Before Coding)

1. Confirm canonical 3D order choice:
- This draft uses `(nz, ny, nx)` / `(nz, ny, nx, 3)`.
- If team prefers `(nx, ny, nz)` / `(nx, ny, nz, 3)`, all tables remain valid semantically but axis section must be updated globally.

2. Confirm `NONE` stage `grid/voxel_id` policy:
- Optional vs required.

3. Confirm whether `grid/dz` is scalar-only pre-CHR or always volume.

No implementation changes should proceed until these 3 points are approved.
