import shlex
import shutil
import sys
import re
import ast
import io
import contextlib
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import astropy.units as u
import numpy as np
import typer
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.io import fits
from pyAMaFiL.mag_field_lin_fff import MagFieldLinFFF
from pyAMaFiL.mag_field_proc import MagFieldProcessor
from scipy.io import readsav
from sunpy.coordinates import (Heliocentric, HeliographicCarrington, HeliographicStonyhurst,
                               Helioprojective, get_earth)
from sunpy.map import Map
from sunpy.sun import constants as sun_consts

from pyampp.data.downloader import SDOImageDownloader
from pyampp.gx_chromo.combo_model import combo_model
from pyampp.gx_chromo.decompose import decompose
from pyampp.gxbox.box import Box
from pyampp.gxbox.boxutils import (
    hmi_b2ptr,
    hmi_disambig,
    read_b3d_h5,
    write_b3d_h5,
    compute_vertical_current,
    load_sunpy_map_compat,
)
from pyampp.gxbox.gx_box2id import gx_box2id
from pyampp.util.config import DOWNLOAD_DIR, GXMODEL_DIR, AIA_EUV_PASSBANDS, AIA_UV_PASSBANDS


app = typer.Typer(help="Run gx_fov2box pipeline without GUI and save model stages.")


@dataclass
class Fov2BoxConfig:
    time: Optional[str]
    coords: Optional[Tuple[float, float]]
    hpc: bool
    hgc: bool
    hgs: bool
    cea: bool
    top: bool
    box_dims: Optional[Tuple[int, int, int]]
    dx_km: float
    pad_frac: float
    data_dir: str
    gxmodel_dir: str
    entry_box: Optional[str]
    save_empty_box: bool
    save_potential: bool
    save_bounds: bool
    save_nas: bool
    save_gen: bool
    save_chr: bool
    stop_after: Optional[str]
    empty_box_only: bool
    potential_only: bool
    nlfff_only: bool
    generic_only: bool
    use_potential: bool
    skip_lines: bool
    center_vox: bool
    reduce_passed: Optional[int]
    euv: bool
    uv: bool
    sfq: bool
    observer_name: str
    fov_xc: Optional[float]
    fov_yc: Optional[float]
    fov_xsize: Optional[float]
    fov_ysize: Optional[float]
    square_fov: bool
    jump2potential: bool
    jump2bounds: bool
    jump2nlfff: bool
    jump2lines: bool
    jump2chromo: bool
    rebuild: bool
    rebuild_from_none: bool
    clone_only: bool
    info: bool


def _infer_time_from_path(path: Path) -> Optional[str]:
    stem = path.name
    if "hmi.M_720s." in stem:
        try:
            tag = stem.split("hmi.M_720s.", 1)[1].split(".", 1)[0]
            if len(tag) == 15:
                return f"{tag[0:4]}-{tag[4:6]}-{tag[6:8]}T{tag[9:11]}:{tag[11:13]}:{tag[13:15]}"
        except Exception:
            return None
    return None


def _extract_time_tokens(text: str) -> list[str]:
    if not text:
        return []
    s = _decode_id_text(text)
    patterns = [
        r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z)?",
        r"\d{4}\.\d{2}\.\d{2}_\d{2}:\d{2}:\d{2}(?:_TAI)?",
        r"\d{1,2}-[A-Za-z]{3}-\d{2,4}\s+\d{2}:\d{2}:\d{2}",
    ]
    out: list[str] = []
    for pat in patterns:
        for m in re.finditer(pat, s):
            tok = m.group(0).strip()
            if tok and tok not in out:
                out.append(tok)
    return out


def _parse_time_candidate(token: str) -> Optional[Time]:
    t = _decode_id_text(token).strip()
    if not t:
        return None
    # First try astropy's native parser.
    try:
        return Time(t)
    except Exception:
        pass

    # JSOC T_REC-like values, often e.g. 2025.11.26_15:36:00_TAI
    m = re.match(r"^(\d{4})\.(\d{2})\.(\d{2})_(\d{2}):(\d{2}):(\d{2})(?:_(TAI))?$", t, flags=re.IGNORECASE)
    if m:
        yyyy, mm, dd, hh, mi, ss, scale = m.groups()
        dtx = dt.datetime(int(yyyy), int(mm), int(dd), int(hh), int(mi), int(ss))
        try:
            return Time(dtx, scale="tai" if scale else "utc").utc
        except Exception:
            return None

    # IDL-like values, e.g. 26-Nov-25 15:47:52
    for fmt in ("%d-%b-%y %H:%M:%S", "%d-%b-%Y %H:%M:%S"):
        try:
            dtx = dt.datetime.strptime(t, fmt)
            return Time(dtx, scale="utc")
        except Exception:
            pass
    return None


def _extract_execute_time(execute_text: str) -> Optional[str]:
    if not execute_text:
        return None
    for tok in _extract_time_tokens(execute_text):
        parsed = _parse_time_candidate(tok)
        if parsed is not None:
            return parsed.utc.isot
    return None


def _infer_time_from_entry_loaded(entry_loaded: Dict[str, Any], entry_path: Path) -> Optional[str]:
    """
    Infer model observation time from entry data.
    Priority:
    1) explicit metadata/chromo attrs time fields
    2) base/index metadata text (actual model index header source)
    3) execute command text (requested time)
    4) filename/path fallback
    """
    meta = entry_loaded.get("metadata", {}) if isinstance(entry_loaded, dict) else {}
    if isinstance(meta, dict):
        for key in ("obs_time", "date_obs", "date-obs", "t_rec", "time"):
            if key in meta:
                parsed = _parse_time_candidate(str(meta.get(key)))
                if parsed is not None:
                    return parsed.utc.isot

    # Group attrs can carry true observation time in converted models.
    for group_name in ("chromo", "corona", "base"):
        grp = entry_loaded.get(group_name)
        if isinstance(grp, dict):
            attrs = grp.get("attrs", {})
            if isinstance(attrs, dict):
                for key in ("obs_time", "date_obs", "date-obs", "t_rec", "time"):
                    if key in attrs:
                        parsed = _parse_time_candidate(str(attrs.get(key)))
                        if parsed is not None:
                            return parsed.utc.isot

    # Prefer index/header text over execute (execute can be requested time).
    index_candidates = []
    base = entry_loaded.get("base")
    if isinstance(base, dict) and "index" in base:
        index_candidates.append(base.get("index"))
    if isinstance(meta, dict) and "index" in meta:
        index_candidates.append(meta.get("index"))
    for raw in index_candidates:
        for tok in _extract_time_tokens(_decode_id_text(raw)):
            parsed = _parse_time_candidate(tok)
            if parsed is not None:
                return parsed.utc.isot

    execute_text = _decode_id_text(meta.get("execute", "")) if isinstance(meta, dict) else ""
    inferred_exec = _extract_execute_time(execute_text)
    if inferred_exec:
        return inferred_exec

    return _infer_time_from_path(entry_path)


def _format_coord_tag(lon_deg: float, lat_deg: float, suffix: str = "CR") -> str:
    lon = (lon_deg + 180.0) % 360.0 - 180.0
    lon_dir = "W" if lon >= 0 else "E"
    lat_dir = "N" if lat_deg >= 0 else "S"
    lon_val = f"{abs(int(round(lon))):02d}"
    lat_val = f"{abs(int(round(lat_deg))):02d}"
    return f"{lon_dir}{lon_val}{lat_dir}{lat_val}{suffix}"


def _stage_output_dir(gxmodel_dir: str, obs_time: Time) -> Path:
    date_str = obs_time.to_datetime().strftime("%Y-%m-%d")
    out_dir = Path(gxmodel_dir) / date_str
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _stage_file_base(obs_time: Time, coord_tag: str, projection_tag: str = "CEA") -> str:
    time_tag = obs_time.to_datetime().strftime("%Y%m%d_%H%M%S")
    return f"hmi.M_720s.{time_tag}.{coord_tag}.{projection_tag}"


def _stage_filename(out_dir: Path, base: str, stage_tag: str) -> Path:
    return out_dir / f"{base}.{stage_tag}.h5"


def _last_stage_tag(stop_after: Optional[str]) -> str:
    if stop_after:
        stop = stop_after.lower()
        if stop in ("dl", "download", "downloads"):
            return "DL"
        if stop in ("none", "empty", "empty_box"):
            return "NONE"
        if stop in ("bnd", "bounds"):
            return "BND"
        if stop == "pot":
            return "POT"
        if stop == "nas":
            return "NAS"
        if stop == "gen":
            return "GEN"
        if stop == "chr":
            return "CHR"
    return "CHR"


def _canon_stage_name(tag: Optional[str]) -> str:
    if not tag:
        return "CHR"
    t = tag.lower()
    if t in ("dl", "download", "downloads"):
        return "DL"
    if t in ("none", "empty", "empty_box"):
        return "NONE"
    if t in ("pot", "potential"):
        return "POT"
    if t in ("bnd", "bounds"):
        return "BND"
    if t in ("nas", "nlfff"):
        return "NAS"
    if t in ("gen", "nas.gen", "lines"):
        return "GEN"
    if t in ("chr", "nas.chr", "chromo"):
        return "CHR"
    return t.upper()


def _entry_stage_from_loaded(loaded: Dict[str, Any], entry_path: Path) -> str:
    # Prefer explicit metadata id/model_type when present.
    model_id = _decode_id_text(loaded.get("metadata", {}).get("id", "")).upper()
    if ".POT.GEN.CHR" in model_id:
        return "CHR"
    if ".NAS.GEN.CHR" in model_id:
        return "CHR"
    if ".POT.CHR" in model_id:
        return "CHR"
    if ".POT.GEN" in model_id:
        return "GEN"
    if ".NAS.CHR" in model_id:
        return "CHR"
    if ".NAS.GEN" in model_id:
        return "GEN"
    if ".NAS." in model_id:
        return "NAS"
    if ".POT" in model_id:
        return "POT"
    if ".BND" in model_id:
        return "BND"
    if ".NONE" in model_id:
        return "NONE"

    if "chromo" in loaded:
        return "CHR"
    if "lines" in loaded:
        return "GEN"
    if "corona" in loaded:
        model_type = str(loaded["corona"].get("attrs", {}).get("model_type", "")).lower()
        if model_type in ("none",):
            return "NONE"
        if model_type in ("pot",):
            return "POT"
        if model_type in ("bnd", "bounds"):
            return "BND"
        if model_type in ("nlfff",):
            return "NAS"
        return "NAS"

    # Fallback from filename.
    up = entry_path.name.upper()
    if ".POT.GEN.CHR" in up:
        return "CHR"
    if ".NAS.GEN.CHR" in up:
        return "CHR"
    if ".POT.CHR" in up:
        return "CHR"
    if ".POT.GEN" in up:
        return "GEN"
    if ".NAS.CHR" in up:
        return "CHR"
    if ".NAS.GEN" in up:
        return "GEN"
    if ".NAS." in up:
        return "NAS"
    if ".POT" in up:
        return "POT"
    if ".BND" in up:
        return "BND"
    if ".NONE" in up:
        return "NONE"
    return "NONE"


def _detect_target_stage(cfg: Fov2BoxConfig, entry_stage: Optional[str]) -> str:
    if cfg.rebuild or cfg.rebuild_from_none:
        return "NONE"
    if cfg.jump2potential:
        return "POT"
    if cfg.jump2bounds:
        return "BND"
    if cfg.jump2nlfff:
        return "NAS"
    if cfg.jump2lines:
        return "GEN"
    if cfg.jump2chromo:
        return "CHR"
    if entry_stage:
        return entry_stage
    return "NONE"


def _jump_allowed(entry_stage: str, target_stage: str) -> bool:
    order = {"NONE": 0, "POT": 1, "BND": 2, "NAS": 3, "GEN": 4, "CHR": 5}
    if entry_stage not in order or target_stage not in order:
        return False
    e = order[entry_stage]
    t = order[target_stage]
    if t <= e:
        return True  # same stage or backward jump
    if t == e + 1:
        return True  # forward by one stage
    if entry_stage == "NONE" and target_stage == "BND":
        return True  # convenience shortcut; POT is computed implicitly
    if entry_stage == "POT" and target_stage == "NAS":
        return True  # implicit POT->BND->NAS transition
    if entry_stage == "POT" and target_stage == "GEN":
        return True  # explicit pipeline exception
    if entry_stage == "POT" and target_stage == "CHR":
        return True  # allow direct POT->CHR (no lines group)
    if entry_stage == "NAS" and target_stage == "CHR":
        return True  # allow direct NAS->CHR (no lines group)
    return False


def _warn_impossible_save_requests(cfg: Fov2BoxConfig, start_stage: str) -> None:
    order = {"NONE": 0, "POT": 1, "BND": 2, "NAS": 3, "GEN": 4, "CHR": 5}
    start_ord = order.get(start_stage, 0)
    req = [
        ("NONE", cfg.save_empty_box, "--save-empty-box"),
        ("BND", cfg.save_bounds, "--save-bounds"),
        ("POT", cfg.save_potential, "--save-potential"),
        ("NAS", cfg.save_nas, "--save-nas"),
        ("GEN", cfg.save_gen, "--save-gen"),
        ("CHR", cfg.save_chr, "--save-chr"),
    ]
    impossible = [flag for st, enabled, flag in req if enabled and order.get(st, 0) < start_ord]
    if impossible:
        print(
            "Warning: impossible save request(s) before selected start stage "
            f"{start_stage}: {', '.join(impossible)} (ignored)."
        )


def _stage_tag_from_stage(stage: str) -> str:
    return {
        "NONE": "NONE",
        "POT": "POT",
        "BND": "BND",
        "NAS": "NAS",
        "GEN": "NAS.GEN",
        "CHR": "NAS.CHR",
    }.get(stage, "NAS.CHR")


def _extract_execute_paths(execute_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract data/model directory defaults from metadata/execute.
    Supports both Python CLI form and IDL gx_fov2box form.
    """
    if not execute_text:
        return None, None

    data_dir: Optional[str] = None
    gxmodel_dir: Optional[str] = None
    text = str(execute_text)

    # Python CLI style: gx-fov2box ... --data-dir X --gxmodel-dir Y
    try:
        parts = shlex.split(text)
    except Exception:
        parts = []
    for i, token in enumerate(parts):
        if token == "--data-dir" and i + 1 < len(parts):
            data_dir = parts[i + 1]
        elif token.startswith("--data-dir="):
            data_dir = token.split("=", 1)[1]
        elif token == "--gxmodel-dir" and i + 1 < len(parts):
            gxmodel_dir = parts[i + 1]
        elif token.startswith("--gxmodel-dir="):
            gxmodel_dir = token.split("=", 1)[1]

    # IDL style fallback: TMP_DIR='...' and OUT_DIR='...'
    if data_dir is None:
        m = re.search(r"\bTMP_DIR\s*=\s*['\"]([^'\"]+)['\"]", text, flags=re.IGNORECASE)
        if not m:
            m = re.search(r"\bTMP_DIR\s*=\s*([^,\s]+)", text, flags=re.IGNORECASE)
        if m:
            data_dir = m.group(1).strip().strip("'\"")
    if gxmodel_dir is None:
        m = re.search(r"\bOUT_DIR\s*=\s*['\"]([^'\"]+)['\"]", text, flags=re.IGNORECASE)
        if not m:
            m = re.search(r"\bOUT_DIR\s*=\s*([^,\s]+)", text, flags=re.IGNORECASE)
        if m:
            gxmodel_dir = m.group(1).strip().strip("'\"")

    return data_dir, gxmodel_dir


def _extract_execute_geometry(execute_text: str) -> Tuple[Optional[Tuple[float, float]], Optional[str], Optional[str]]:
    """
    Extract center coordinates, frame and projection hints from metadata/execute text.

    Returns:
      (coords, frame, projection)
      - coords: (x, y) as floats
      - frame: one of {"hpc","hgc","hgs"} or None
      - projection: one of {"cea","top"} or None
    """
    if not execute_text:
        return None, None, None
    text = str(execute_text)
    coords: Optional[Tuple[float, float]] = None
    frame: Optional[str] = None
    projection: Optional[str] = None

    # Python CLI style: --coords X Y [--hpc|--hgc|--hgs] [--cea|--top]
    try:
        parts = shlex.split(text)
    except Exception:
        parts = []
    for i, token in enumerate(parts):
        if token == "--coords" and i + 2 < len(parts):
            try:
                coords = (float(parts[i + 1]), float(parts[i + 2]))
            except Exception:
                pass
        elif token.startswith("--coords="):
            try:
                vals = token.split("=", 1)[1].split(",")
                if len(vals) == 2:
                    coords = (float(vals[0]), float(vals[1]))
            except Exception:
                pass
        elif token == "--hpc":
            frame = "hpc"
        elif token == "--hgc":
            frame = "hgc"
        elif token == "--hgs":
            frame = "hgs"
        elif token == "--cea":
            projection = "cea"
        elif token == "--top":
            projection = "top"

    # IDL canonical pattern:
    # CENTER_ARCSEC=[ -280, -250]
    if coords is None:
        m = re.search(
            r"\bCENTER_ARCSEC\s*=\s*\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\]",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            coords = (float(m.group(1)), float(m.group(2)))
            frame = frame or "hpc"

    # Optional degree-based patterns.
    if coords is None:
        m = re.search(
            r"\bCENTER_HGC\s*=\s*\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\]",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            coords = (float(m.group(1)), float(m.group(2)))
            frame = "hgc"
    if coords is None:
        m = re.search(
            r"\bCENTER_HGS\s*=\s*\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\]",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            coords = (float(m.group(1)), float(m.group(2)))
            frame = "hgs"

    # Projection hint in IDL execute.
    if projection is None:
        up = text.upper()
        if " /TOP" in up or ", TOP=1" in up:
            projection = "top"
        elif " /CEA" in up or ", CEA=1" in up:
            projection = "cea"

    return coords, frame, projection


def _flag_explicit_on_cli(flag: str) -> bool:
    args = sys.argv[1:]
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in args)


def _warn_path_default(role: str, path_text: str) -> None:
    p = Path(path_text).expanduser()
    if p.exists():
        return
    parent = p.parent
    if parent.exists():
        print(f"Warning: {role} from metadata/execute does not exist yet: {p} (will be used as requested).")
    else:
        print(f"Warning: {role} parent directory does not exist on this system: {parent}")


def _lines_quiet(maglib: Any, **kwargs: Any) -> Dict[str, Any]:
    """Suppress noisy stdout/stderr emitted by the underlying line tracer."""
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        return maglib.lines(**kwargs)


def _decode_id_text(v: Any) -> str:
    if hasattr(v, "item"):
        try:
            return _decode_id_text(v.item())
        except Exception:
            pass
    if isinstance(v, (bytes, bytearray)):
        return bytes(v).decode("utf-8", errors="ignore")
    s = str(v or "").strip()
    if s.startswith(("b'", 'b"', "B'", 'B"')):
        try:
            # Normalize uppercase byte literal prefix (e.g. B'CEA') to valid Python syntax.
            lit_src = f"b{s[1:]}" if s[:1] == "B" else s
            lit = ast.literal_eval(lit_src)
            if isinstance(lit, (bytes, bytearray)):
                return bytes(lit).decode("utf-8", errors="ignore")
        except Exception:
            pass
    return s


def _split_stage_id(model_id: str) -> Tuple[str, str]:
    sid = _decode_id_text(model_id)
    up = sid.upper()
    for suffix in (
        ".POT.GEN.CHR", ".NAS.GEN.CHR",
        ".POT.CHR", ".NAS.CHR",
        ".POT.GEN", ".NAS.GEN",
        ".NAS", ".BND", ".POT", ".NONE",
    ):
        if up.endswith(suffix):
            return sid[: -len(suffix)], suffix[1:]
    return sid, ""


def _canonical_lineage_suffix(stage_tag: str) -> str:
    mapping = {
        "NONE": "NONE",
        "POT": "NONE.POT",
        "BND": "NONE.POT.BND",
        "NAS": "NONE.POT.BND.NAS",
        "POT.GEN": "NONE.POT.GEN",
        "POT.CHR": "NONE.POT.CHR",
        "POT.GEN.CHR": "NONE.POT.GEN.CHR",
        "NAS.GEN": "NONE.POT.BND.NAS.GEN",
        "NAS.CHR": "NONE.POT.BND.NAS.CHR",
        "NAS.GEN.CHR": "NONE.POT.BND.NAS.GEN.CHR",
    }
    return mapping.get(stage_tag, stage_tag)


def _merge_lineage(root: str, suffix: str) -> str:
    r = [t for t in str(root or "").split(".") if t]
    s = [t for t in str(suffix or "").split(".") if t]
    if not r:
        return ".".join(s)
    if not s:
        return ".".join(r)
    max_k = min(len(r), len(s))
    overlap = 0
    for i in range(max_k, 0, -1):
        if r[-i:] == s[:i]:
            overlap = i
            break
    return ".".join(r + s[overlap:])


def _lineage_delta_from_entry(entry_stage_tag: str, saved_stage_tag: str) -> str:
    """Return lineage segment from entry stage to saved stage (inclusive of implied steps)."""
    e = [t for t in _canonical_lineage_suffix(entry_stage_tag).split(".") if t]
    s = [t for t in _canonical_lineage_suffix(saved_stage_tag).split(".") if t]
    # Remove shared prefix (e.g., NONE.POT), keep what was computed after entry.
    i = 0
    while i < min(len(e), len(s)) and e[i] == s[i]:
        i += 1
    tail = s[i:]
    if not tail:
        # Clone/self-save: keep explicit stage identity on RHS.
        return saved_stage_tag
    return ".".join(tail)


def _should_save_stage(stage_tag: str, cfg: Fov2BoxConfig) -> bool:
    stop_tag = _last_stage_tag(cfg.stop_after)
    if stop_tag == "GEN" and stage_tag in ("NAS.GEN", "POT.GEN"):
        return True
    if stop_tag == "CHR" and stage_tag.endswith(".CHR"):
        return True
    if stage_tag == stop_tag:
        return True
    if stage_tag == "POT":
        return cfg.save_potential
    if stage_tag == "NAS":
        return cfg.save_nas
    if stage_tag in ("NAS.GEN", "POT.GEN"):
        return cfg.save_gen
    if stage_tag.endswith(".CHR"):
        return cfg.save_chr
    if stage_tag == "NONE":
        return cfg.save_empty_box
    if stage_tag == "BND":
        return cfg.save_bounds
    return False


def _build_execute_cmd(cfg: Fov2BoxConfig) -> str:
    cmd = ["gx-fov2box"]
    if cfg.time:
        cmd += ["--time", cfg.time]
    if cfg.coords:
        cmd += ["--coords", str(cfg.coords[0]), str(cfg.coords[1])]
    if cfg.hpc:
        cmd.append("--hpc")
    elif cfg.hgc:
        cmd.append("--hgc")
    else:
        cmd.append("--hgs")
    if cfg.top:
        cmd.append("--top")
    elif cfg.cea:
        cmd.append("--cea")
    if cfg.box_dims:
        cmd += ["--box-dims", *(str(v) for v in cfg.box_dims)]
    cmd += ["--dx-km", f"{cfg.dx_km:.6f}"]
    cmd += ["--pad-frac", f"{cfg.pad_frac:.4f}"]
    cmd += ["--data-dir", cfg.data_dir]
    cmd += ["--gxmodel-dir", cfg.gxmodel_dir]
    if cfg.euv:
        cmd.append("--euv")
    if cfg.uv:
        cmd.append("--uv")
    if cfg.save_empty_box:
        cmd.append("--save-empty-box")
    if cfg.save_potential:
        cmd.append("--save-potential")
    if cfg.save_bounds:
        cmd.append("--save-bounds")
    if cfg.save_nas:
        cmd.append("--save-nas")
    if cfg.save_gen:
        cmd.append("--save-gen")
    if cfg.save_chr:
        cmd.append("--save-chr")
    if cfg.empty_box_only:
        cmd.append("--empty-box-only")
    if cfg.potential_only:
        cmd.append("--potential-only")
    if cfg.nlfff_only:
        cmd.append("--nlfff-only")
    if cfg.generic_only:
        cmd.append("--generic-only")
    if cfg.use_potential:
        cmd.append("--use-potential")
    if cfg.skip_lines:
        cmd.append("--skip-lines")
    if cfg.center_vox:
        cmd.append("--center-vox")
    if cfg.reduce_passed is not None:
        cmd += ["--reduce-passed", str(cfg.reduce_passed)]
    if cfg.entry_box:
        cmd += ["--entry-box", cfg.entry_box]
    if cfg.sfq:
        cmd.append("--sfq")
    if cfg.observer_name:
        cmd += ["--observer-name", cfg.observer_name]
    if cfg.fov_xc is not None:
        cmd += ["--fov-xc", f"{cfg.fov_xc:.2f}"]
    if cfg.fov_yc is not None:
        cmd += ["--fov-yc", f"{cfg.fov_yc:.2f}"]
    if cfg.fov_xsize is not None:
        cmd += ["--fov-xsize", f"{cfg.fov_xsize:.2f}"]
    if cfg.fov_ysize is not None:
        cmd += ["--fov-ysize", f"{cfg.fov_ysize:.2f}"]
    if cfg.square_fov:
        cmd.append("--square-fov")
    if cfg.stop_after:
        cmd += ["--stop-after", cfg.stop_after]
    if cfg.jump2potential:
        cmd.append("--jump2potential")
    if cfg.jump2bounds:
        cmd.append("--jump2bounds")
    if cfg.jump2nlfff:
        cmd.append("--jump2nlfff")
    if cfg.jump2lines:
        cmd.append("--jump2lines")
    if cfg.jump2chromo:
        cmd.append("--jump2chromo")
    if cfg.rebuild:
        cmd.append("--rebuild")
    if cfg.rebuild_from_none:
        cmd.append("--rebuild-from-none")
    if cfg.clone_only:
        cmd.append("--clone-only")
    return shlex.join(cmd)


def _explicit_observer_fov(cfg: Fov2BoxConfig) -> Optional[dict[str, Any]]:
    parts = [cfg.fov_xc, cfg.fov_yc, cfg.fov_xsize, cfg.fov_ysize]
    if all(v is None for v in parts):
        return None
    if any(v is None for v in parts):
        raise ValueError("Observer FOV requires all of --fov-xc/--fov-yc/--fov-xsize/--fov-ysize.")
    return {
        "frame": "helioprojective",
        "xc_arcsec": float(cfg.fov_xc),
        "yc_arcsec": float(cfg.fov_yc),
        "xsize_arcsec": float(cfg.fov_xsize),
        "ysize_arcsec": float(cfg.fov_ysize),
        "square": bool(cfg.square_fov),
    }


def _observer_metadata_from_source_map(source_map: Optional[Map], cfg: Fov2BoxConfig) -> dict[str, Any]:
    observer_meta: dict[str, Any] = {"name": str(cfg.observer_name or "earth")}
    explicit_fov = _explicit_observer_fov(cfg)
    if explicit_fov is not None:
        observer_meta["fov"] = explicit_fov
    ephemeris: dict[str, Any] = {}
    if source_map is not None:
        obs = getattr(source_map, "observer_coordinate", None)
        if obs is not None:
            try:
                obs_hgs = obs.transform_to(HeliographicStonyhurst(obstime=source_map.date))
                ephemeris["b0_deg"] = float(obs_hgs.lat.to_value(u.deg))
            except Exception:
                pass
            try:
                obs_hgc = obs.transform_to(HeliographicCarrington(observer="earth", obstime=source_map.date))
                ephemeris["l0_deg"] = float(obs_hgc.lon.to_value(u.deg))
            except Exception:
                pass
        if hasattr(source_map, "rsun_meters") and source_map.rsun_meters is not None:
            ephemeris["rsun_cm"] = float(u.Quantity(source_map.rsun_meters).to_value(u.cm))
        if hasattr(source_map, "dsun") and source_map.dsun is not None:
            ephemeris["dsun_cm"] = float(u.Quantity(source_map.dsun).to_value(u.cm))
    if ephemeris:
        observer_meta["ephemeris"] = ephemeris
    return observer_meta


def _observer_metadata_from_entry(entry_loaded: Optional[Dict[str, Any]], cfg: Fov2BoxConfig) -> Optional[dict[str, Any]]:
    if not isinstance(entry_loaded, dict):
        return None
    base_observer = entry_loaded.get("observer", {})
    observer_meta: dict[str, Any] = dict(base_observer) if isinstance(base_observer, dict) else {}
    observer_meta["name"] = str(cfg.observer_name or observer_meta.get("name") or "earth")
    explicit_fov = _explicit_observer_fov(cfg)
    if explicit_fov is not None:
        observer_meta["fov"] = explicit_fov
    ephemeris = dict(observer_meta.get("ephemeris", {})) if isinstance(observer_meta.get("ephemeris"), dict) else {}
    if not ephemeris:
        index_text = None
        base_group = entry_loaded.get("base")
        if isinstance(base_group, dict):
            index_text = base_group.get("index")
        if index_text:
            try:
                header = fits.Header.fromstring(_decode_id_text(index_text), sep="\n")
                if "SOLAR_B0" in header:
                    ephemeris["b0_deg"] = float(header["SOLAR_B0"])
                if "CRLN_OBS" in header:
                    ephemeris["l0_deg"] = float(header["CRLN_OBS"])
                elif "HGLN_OBS" in header:
                    ephemeris["l0_deg"] = float(header["HGLN_OBS"])
                if "RSUN_REF" in header:
                    ephemeris["rsun_cm"] = float(u.Quantity(header["RSUN_REF"], u.m).to_value(u.cm))
                if "DSUN_OBS" in header:
                    ephemeris["dsun_cm"] = float(u.Quantity(header["DSUN_OBS"], u.m).to_value(u.cm))
            except Exception:
                pass
    if ephemeris:
        observer_meta["ephemeris"] = ephemeris
    return observer_meta or None


def _build_index_header(bottom_wcs_header, source_map: Map) -> str:
    """
    Build an IDL-GX compatible INDEX-like FITS header string.

    This mirrors the intent of IDL `wcs2fitshead(..., /structure)` for the
    base map WCS: preserve WCS cards and add key TIME/POSITION cards expected
    by legacy GX Simulator routines.
    """
    header = fits.Header(bottom_wcs_header).copy()

    # Normalize to IDL-GX conventions used in box.index.
    ctype1 = str(header.get("CTYPE1", ""))
    ctype2 = str(header.get("CTYPE2", ""))
    if ctype1.startswith("HGLN-"):
        header["CTYPE1"] = "CRLN-" + ctype1.split("-", 1)[1]
    if ctype2.startswith("HGLT-"):
        header["CTYPE2"] = "CRLT-" + ctype2.split("-", 1)[1]

    header["SIMPLE"] = int(bool(header.get("SIMPLE", 1)))
    header["BITPIX"] = int(header.get("BITPIX", 8))
    header["NAXIS"] = int(header.get("NAXIS", 2))

    if source_map is not None:
        date_obs = source_map.date.isot if source_map.date is not None else None
        if date_obs:
            header["DATE-OBS"] = date_obs
            header["DATE_OBS"] = date_obs
            header["DATE"] = date_obs
        elif "DATE-OBS" in header and "DATE_OBS" not in header:
            header["DATE_OBS"] = header["DATE-OBS"]

        if hasattr(source_map, "dsun") and source_map.dsun is not None:
            header["DSUN_OBS"] = float(source_map.dsun.to_value(u.m))

        obs = source_map.observer_coordinate
        if obs is not None:
            obs_hgs = obs.transform_to(HeliographicStonyhurst(obstime=source_map.date))
            header["HGLN_OBS"] = float(obs_hgs.lon.to_value(u.deg))
            header["HGLT_OBS"] = float(obs_hgs.lat.to_value(u.deg))
            header["SOLAR_B0"] = float(obs_hgs.lat.to_value(u.deg))
            try:
                obs_hgc = obs.transform_to(HeliographicCarrington(observer="earth", obstime=source_map.date))
                header["CRLN_OBS"] = float(obs_hgc.lon.to_value(u.deg))
                header["CRLT_OBS"] = float(obs_hgc.lat.to_value(u.deg))
            except Exception:
                # If Carrington transformation fails (e.g. unsupported observer configuration),
                # proceed without CRLN_OBS/CRLT_OBS rather than aborting header creation.
                # Carrington observer transforms can fail for some observer metadata;
                # keep header generation robust and proceed with available keys.
                pass

        if getattr(source_map, "rsun_meters", None) is not None:
            header["RSUN_REF"] = float(source_map.rsun_meters.to_value(u.m))

        tel = source_map.meta.get("telescop")
        if tel:
            header["TELESCOP"] = str(tel)
        instr = source_map.meta.get("instrume")
        if instr:
            header["INSTRUME"] = str(instr)
        if "WCSNAME" not in header:
            header["WCSNAME"] = "Carrington-Heliographic"

    return header.tostring(sep="\n", endcard=True)


def _print_info(cfg: Fov2BoxConfig) -> None:
    resolved_reduce_passed = cfg.reduce_passed if cfg.reduce_passed is not None else (0 if cfg.center_vox else 1)
    import pyampp
    import sunpy

    runtime_rows = [
        ("python_executable", sys.executable, "Interpreter used for this run"),
        ("python_version", sys.version.split()[0], "Python runtime version"),
        ("gx_fov2box_module", __file__, "Loaded gx_fov2box module path"),
        ("pyampp_module", pyampp.__file__, "Loaded pyampp package path"),
        ("sunpy_version", sunpy.__version__, "Loaded sunpy version"),
        ("gx-fov2box_cli", shutil.which("gx-fov2box"), "Resolved CLI entry point on PATH"),
    ]
    rows = [
        ("time", cfg.time, "Observation time in ISO format"),
        ("coords", cfg.coords, "Center coordinates: arcsec (HPC) or degrees (HGC/HGS)"),
        ("hpc/hgc/hgs", "HPC" if cfg.hpc else "HGC" if cfg.hgc else "HGS", "Coordinate frame"),
        ("cea/top", "TOP" if cfg.top else "CEA", "Basemap projection"),
        ("box_dims", cfg.box_dims, "Box dimensions (nx, ny, nz) in pixels"),
        ("dx_km", cfg.dx_km, "Voxel size in km"),
        ("pad_frac", cfg.pad_frac, "Padding fraction used for context map FOV"),
        ("data_dir", cfg.data_dir, "SDO download/cache directory"),
        ("gxmodel_dir", cfg.gxmodel_dir, "Output gx_models directory"),
        ("entry_box", cfg.entry_box, "Path to precomputed HDF5 box"),
        ("save_empty_box", cfg.save_empty_box, "Save NONE stage"),
        ("save_potential", cfg.save_potential, "Save POT stage"),
        ("save_bounds", cfg.save_bounds, "Save BND stage"),
        ("save_nas", cfg.save_nas, "Save NAS stage"),
        ("save_gen", cfg.save_gen, "Save NAS.GEN stage"),
        ("save_chr", cfg.save_chr, "Save NAS.CHR stage"),
        ("stop_after", cfg.stop_after, "Stop after stage (dl/none/bnd/pot/nas/gen/chr)"),
        ("empty_box_only", cfg.empty_box_only, "Stop after NONE stage"),
        ("potential_only", cfg.potential_only, "Stop after POT stage"),
        ("nlfff_only", cfg.nlfff_only, "Stop after NAS stage"),
        ("generic_only", cfg.generic_only, "Stop after NAS.GEN stage"),
        ("use_potential", cfg.use_potential, "Skip NLFFF; reuse potential field"),
        ("skip_lines", cfg.skip_lines, "Skip GEN line computation and proceed directly to CHR"),
        ("center_vox", cfg.center_vox, "Compute lines only through voxel centers (sets reduce_passed=0 unless overridden)"),
        ("reduce_passed", cfg.reduce_passed, "Expert override: 0|1|2|3 (takes precedence over --center-vox)"),
        ("reduce_passed_resolved", resolved_reduce_passed, "Effective line-reduction mode used by tracer"),
        ("euv", cfg.euv, "Download AIA EUV context maps"),
        ("uv", cfg.uv, "Download AIA UV context maps"),
        ("sfq", cfg.sfq, "Use SFQ disambiguation (method=0)"),
        ("jump2potential", cfg.jump2potential, "Start from entry box and jump to POT"),
        ("jump2bounds", cfg.jump2bounds, "Start from entry box and jump to BND"),
        ("jump2nlfff", cfg.jump2nlfff, "Start from entry box and jump to NAS"),
        ("jump2lines", cfg.jump2lines, "Start from entry box and jump to GEN"),
        ("jump2chromo", cfg.jump2chromo, "Start from entry box and jump to CHR"),
        ("rebuild", cfg.rebuild, "Ignore entry stage payload and rebuild from NONE"),
        ("rebuild_from_none", cfg.rebuild_from_none, "Strip entry to NONE-equivalent and continue forward"),
        ("clone_only", cfg.clone_only, "Convert/copy entry-box to normalized H5 without recomputation"),
    ]
    print("gx-fov2box --info")
    for name, value, desc in runtime_rows + rows:
        print(f"- {name}: {value} :: {desc}")
    missing = []
    if not cfg.time:
        missing.append("--time")
    if not cfg.coords:
        missing.append("--coords")
    if not cfg.box_dims:
        missing.append("--box-dims")
    if missing:
        print("\nWarnings:")
        for item in missing:
            print(f"- Missing required {item}")
        if cfg.entry_box:
            print("- entry_box provided; time/box-dims may be inferred if present")


def _load_hmi_maps_from_downloader(
    time: Time,
    data_dir: Path,
    euv: bool,
    uv: bool,
    disambig_method: int = 2,
    strict_required: bool = True,
) -> tuple[Dict[str, Map], dict]:
    import time as time_mod

    downloader = SDOImageDownloader(time, data_dir=str(data_dir), euv=euv, uv=uv, hmi=True)
    missing_before = []
    if downloader.existence_report:
        for category in ("hmi_b", "hmi_m", "hmi_ic"):
            items = downloader.existence_report.get(category, {})
            missing_before.extend([k for k, exists in items.items() if not exists])
    t0 = time_mod.perf_counter()
    files = downloader.download_images()
    elapsed = time_mod.perf_counter() - t0
    downloaded = len(missing_before) > 0
    required = ["field", "inclination", "azimuth", "disambig", "continuum", "magnetogram"]
    missing = [k for k in required if not files.get(k)]
    if missing and strict_required:
        raise RuntimeError(f"Missing required HMI files: {missing}")
    if missing and not strict_required:
        info = {"downloaded": downloaded, "elapsed": elapsed, "missing_before": missing_before, "missing_after": missing, "files": files}
        return {}, info

    map_field = load_sunpy_map_compat(files["field"])
    map_inclination = load_sunpy_map_compat(files["inclination"])
    map_azimuth = load_sunpy_map_compat(files["azimuth"])
    map_disambig = load_sunpy_map_compat(files["disambig"])
    map_conti = load_sunpy_map_compat(files["continuum"])
    map_losma = load_sunpy_map_compat(files["magnetogram"])

    map_azimuth = hmi_disambig(map_azimuth, map_disambig, method=disambig_method)

    maps = {
        "field": map_field,
        "inclination": map_inclination,
        "azimuth": map_azimuth,
        "disambig": map_disambig,
        "continuum": map_conti,
        "magnetogram": map_losma,
    }
    for key, path in files.items():
        if key in ("field", "inclination", "azimuth", "disambig", "continuum", "magnetogram"):
            continue
        try:
            maps[f"AIA_{key}"] = load_sunpy_map_compat(path)
        except Exception:
            # Skip optional context channels that cannot be loaded.
            continue
    info = {"downloaded": downloaded, "elapsed": elapsed, "missing_before": missing_before}
    return maps, info


def _make_header(map_field: Map) -> Dict[str, Any]:
    header_field = map_field.wcs.to_header()
    field_frame = map_field.center.heliographic_carrington.frame
    lon, lat = field_frame.lon.value, field_frame.lat.value
    obs_time = Time(map_field.date)
    dsun_obs = header_field["DSUN_OBS"]
    return {"lon": lon, "lat": lat, "dsun_obs": dsun_obs, "obs_time": str(obs_time.iso)}


def _make_lines_group(lines: dict, dr3: np.ndarray) -> dict:
    group = {
        "dr": np.asarray(dr3, dtype=float),
    }
    for key in (
        "start_idx",
        "end_idx",
        "seed_idx",
        "apex_idx",
        "codes",
        "av_field",
        "phys_length",
        "voxel_status",
    ):
        if key in lines:
            group[key] = lines[key]
    return group


def _make_chromo_group(chromo_box: dict) -> dict:
    group = {}
    for key in (
        "chromo_idx",
        "chromo_n",
        "chromo_t",
        "n_p",
        "n_hi",
        "n_htot",
        "tr",
        "tr_h",
        "chromo_layers",
        "dz",
        "chromo_mask",
    ):
        if key in chromo_box:
            group[key] = chromo_box[key]

    # Export CHR vectors explicitly in H5 (no BCUBE replacement in schema).
    chromo_bcube = chromo_box.get("chromo_bcube")
    if isinstance(chromo_bcube, np.ndarray) and chromo_bcube.ndim == 4 and chromo_bcube.shape[-1] == 3:
        group["bx"] = chromo_bcube[:, :, :, 0]
        group["by"] = chromo_bcube[:, :, :, 1]
        group["bz"] = chromo_bcube[:, :, :, 2]
    return group


def _to_h5_2d(arr: np.ndarray, ny: int, nx: int) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim != 2:
        return a
    if a.shape == (ny, nx):
        return a
    if a.shape == (nx, ny):
        return a.T
    raise ValueError(f"Cannot normalize 2D map shape {a.shape} to (ny,nx)=({ny},{nx})")


def _to_h5_3d(arr: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """
    Normalize internal cube to H5 canonical order (nz, ny, nx).
    Accepts common permutations used by current pipeline.
    """
    a = np.asarray(arr)
    if a.ndim != 3:
        return a
    if a.shape[1:] == (ny, nx):  # already z,y,x
        return a
    if a.shape[:2] == (ny, nx):  # y,x,z
        return a.transpose((2, 0, 1))
    if a.shape[:2] == (nx, ny):  # x,y,z
        return a.transpose((2, 1, 0))
    if a.shape[1:] == (nx, ny):  # z,x,y
        return a.transpose((0, 2, 1))
    raise ValueError(f"Cannot normalize 3D cube shape {a.shape} to (nz,ny,nx) with (ny,nx)=({ny},{nx})")


def _normalize_stage_for_h5(stage_box: dict) -> dict:
    out = dict(stage_box)
    if "base" not in out:
        return out
    base = dict(out["base"])
    if "bx" in base:
        ny, nx = np.asarray(base["bx"]).shape
    elif "bz" in base:
        ny, nx = np.asarray(base["bz"]).shape
    else:
        out["base"] = base
        return out

    for k in ("bx", "by", "bz", "ic", "chromo_mask"):
        if k in base:
            base[k] = _to_h5_2d(base[k], ny, nx)
    out["base"] = base

    if "corona" in out and isinstance(out["corona"], dict):
        corona = dict(out["corona"])
        for k in ("bx", "by", "bz"):
            if k in corona:
                corona[k] = _to_h5_3d(corona[k], ny, nx)
        out["corona"] = corona

    if "chromo" in out and isinstance(out["chromo"], dict):
        chromo = dict(out["chromo"])
        for k in ("bx", "by", "bz", "dz"):
            if k in chromo:
                chromo[k] = _to_h5_3d(chromo[k], ny, nx)
        for k in ("tr", "tr_h", "chromo_mask"):
            if k in chromo:
                chromo[k] = _to_h5_2d(chromo[k], ny, nx)
        out["chromo"] = chromo

    if "grid" in out and isinstance(out["grid"], dict):
        grid = dict(out["grid"])
        for k in ("voxel_id", "dz"):
            if k in grid and np.asarray(grid[k]).ndim == 3:
                grid[k] = _to_h5_3d(grid[k], ny, nx)
        out["grid"] = grid
    return out


def _h5_corona_to_internal_xyz(corona: dict, axis_order_3d: Optional[str]) -> dict:
    if not isinstance(corona, dict):
        return corona
    axis_order = _decode_id_text(axis_order_3d).lower()
    if axis_order != "zyx":
        return corona
    out = dict(corona)
    for k in ("bx", "by", "bz"):
        if k in out and np.asarray(out[k]).ndim == 3:
            out[k] = np.asarray(out[k]).transpose((2, 1, 0))
    return out


def _sav_cube_to_internal_xyz(arr: np.ndarray, ny: int, nx: int) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim != 3:
        return a
    # SAV restores commonly as (z,y,x) or (x,y,z); normalize to internal (x,y,z).
    if a.shape[1:] == (ny, nx):   # z,y,x
        return a.transpose((2, 1, 0))
    if a.shape[:2] == (nx, ny):   # x,y,z
        return a
    if a.shape[:2] == (ny, nx):   # y,x,z
        return a.transpose((1, 0, 2))
    if a.shape[1:] == (nx, ny):   # z,x,y
        return a.transpose((1, 2, 0))
    return a


def _decode_sav_value(v) -> str:
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="ignore")
    return str(v)


def _load_entry_box_any(entry_path: Path) -> Dict[str, Any]:
    if entry_path.suffix.lower() == ".h5":
        return read_b3d_h5(str(entry_path))
    if entry_path.suffix.lower() != ".sav":
        raise ValueError(f"--entry-box must be .h5 or .sav, got: {entry_path}")

    data = readsav(str(entry_path), python_dict=True, verbose=False)
    if "box" in data:
        box = data["box"][0]
    elif "pbox" in data:
        box = data["pbox"][0]
    else:
        raise ValueError(f"SAV entry box missing box/pbox structure: {entry_path}")

    out: Dict[str, Any] = {}
    names = set(box.dtype.names or [])

    base = None
    if "BASE" in names:
        base_raw = box["BASE"][0]
        bnames = set(base_raw.dtype.names or [])
        if {"BX", "BY", "BZ"}.issubset(bnames):
            base = {
                "bx": np.asarray(base_raw["BX"]),
                "by": np.asarray(base_raw["BY"]),
                "bz": np.asarray(base_raw["BZ"]),
            }
            if "IC" in bnames:
                base["ic"] = np.asarray(base_raw["IC"])
            if "CHROMO_MASK" in bnames:
                base["chromo_mask"] = np.asarray(base_raw["CHROMO_MASK"])
            if "INDEX" in names:
                base["index"] = _decode_sav_value(box["INDEX"])
            out["base"] = base

    ny = nx = None
    if base is not None:
        ny, nx = np.asarray(base["bx"]).shape

    # Build corona from BX/BY/BZ or BCUBE.
    if {"BX", "BY", "BZ"}.issubset(names) and ny is not None and nx is not None:
        out["corona"] = {
            "bx": _sav_cube_to_internal_xyz(np.asarray(box["BX"]), ny, nx),
            "by": _sav_cube_to_internal_xyz(np.asarray(box["BY"]), ny, nx),
            "bz": _sav_cube_to_internal_xyz(np.asarray(box["BZ"]), ny, nx),
            "attrs": {"model_type": "unknown"},
        }
    elif "BCUBE" in names and ny is not None and nx is not None:
        bc = np.asarray(box["BCUBE"])
        # scipy commonly restores BCUBE as (comp, z, y, x)
        if bc.ndim == 4 and bc.shape[0] == 3:
            out["corona"] = {
                "bx": _sav_cube_to_internal_xyz(bc[0], ny, nx),
                "by": _sav_cube_to_internal_xyz(bc[1], ny, nx),
                "bz": _sav_cube_to_internal_xyz(bc[2], ny, nx),
                "attrs": {"model_type": "unknown"},
            }

    # Lines metadata if present.
    lines = {}
    for k in ("STARTIDX", "ENDIDX", "SEED_IDX", "SEEDIDX", "APEX_IDX", "APEXIDX", "CODES",
              "AVFIELD", "PHYSLENGTH", "STATUS", "VOXEL_STATUS"):
        if k in names:
            kk = (
                k.lower()
                .replace("startidx", "start_idx")
                .replace("endidx", "end_idx")
                .replace("seedidx", "seed_idx")
                .replace("apexidx", "apex_idx")
                .replace("avfield", "av_field")
                .replace("physlength", "phys_length")
                .replace("status", "voxel_status")
            )
            lines[kk] = np.asarray(box[k])
    if lines:
        out["lines"] = lines

    # CHR fields if present.
    chromo = {}
    for k in ("CHROMO_IDX", "CHROMO_N", "CHROMO_T", "N_P", "N_HI", "N_HTOT", "TR", "TR_H",
              "CHROMO_LAYERS", "DZ", "CHROMO_MASK", "CHROMO_BCUBE"):
        if k in names:
            kk = k.lower()
            chromo[kk] = np.asarray(box[k])
    if "CHROMO_BCUBE" in names and ny is not None and nx is not None:
        cbc = np.asarray(box["CHROMO_BCUBE"])
        if cbc.ndim == 4 and cbc.shape[0] == 3:
            chromo["bx"] = _sav_cube_to_internal_xyz(cbc[0], ny, nx)
            chromo["by"] = _sav_cube_to_internal_xyz(cbc[1], ny, nx)
            chromo["bz"] = _sav_cube_to_internal_xyz(cbc[2], ny, nx)
    if chromo:
        out["chromo"] = chromo

    sid = _decode_sav_value(box["ID"]) if "ID" in names else entry_path.stem
    execute = _decode_sav_value(box["EXECUTE"]) if "EXECUTE" in names else ""
    out["metadata"] = {"id": sid, "execute": execute}
    return out


def _resolve_box_params(cfg: Fov2BoxConfig) -> Tuple[Time, Tuple[int, int, int], float]:
    obs_time = Time(cfg.time) if cfg.time else None
    box_dims = cfg.box_dims
    dx_km = cfg.dx_km
    if cfg.entry_box:
        entry_path = Path(cfg.entry_box)
        if entry_path.exists() and entry_path.suffix.lower() in (".h5", ".sav"):
            box_b3d = _load_entry_box_any(entry_path)
            corona = box_b3d.get("corona")
            if corona is not None:
                if box_dims is None and "bx" in corona:
                    shape = np.asarray(corona["bx"]).shape
                    axis_order = _decode_id_text(box_b3d.get("metadata", {}).get("axis_order_3d")).lower()
                    if axis_order == "zyx" and len(shape) == 3:
                        box_dims = (int(shape[2]), int(shape[1]), int(shape[0]))
                    else:
                        # internal cubes are x,y,z unless metadata says zyx
                        if len(shape) == 3:
                            box_dims = (int(shape[0]), int(shape[1]), int(shape[2]))
                        else:
                            box_dims = tuple(int(v) for v in shape)
                if "dr" in corona and dx_km == 0:
                    dr3 = corona["dr"]
                    dx_km = float(dr3[0] * sun_consts.radius.to(u.km).value)
        if obs_time is None:
            inferred = _infer_time_from_entry_loaded(box_b3d, entry_path)
            if inferred:
                obs_time = Time(inferred)
    if obs_time is None:
        raise ValueError("--time is required unless it can be inferred from entry_box")
    if box_dims is None:
        raise ValueError("--box-dims is required unless it can be inferred from entry_box")
    return obs_time, tuple(int(v) for v in box_dims), float(dx_km)


@app.command()
def main(
    time: Optional[str] = typer.Option(None, "--time", help="Observation time ISO (e.g. 2024-05-12T00:00:00)"),
    coords: Optional[Tuple[float, float]] = typer.Option(None, "--coords", help="Center coords (x y)"),
    hpc: bool = typer.Option(False, "--hpc/--no-hpc", help="Use helioprojective coordinates"),
    hgc: bool = typer.Option(False, "--hgc/--no-hgc", help="Use heliographic carrington coordinates"),
    hgs: bool = typer.Option(False, "--hgs/--no-hgs", help="Use heliographic stonyhurst coordinates"),
    cea: bool = typer.Option(False, "--cea", help="Use CEA basemap projection"),
    top: bool = typer.Option(False, "--top", help="Use TOP-view basemap projection"),
    box_dims: Optional[Tuple[int, int, int]] = typer.Option(None, "--box-dims", help="Box dims in pixels"),
    dx_km: float = typer.Option(1400.0, "--dx-km", help="Voxel size in km"),
    pad_frac: float = typer.Option(0.10, "--pad-frac", help="Padding fraction for FOV"),
    data_dir: str = typer.Option(DOWNLOAD_DIR, "--data-dir", help="SDO data directory"),
    gxmodel_dir: str = typer.Option(GXMODEL_DIR, "--gxmodel-dir", help="GX model output directory"),
    entry_box: Optional[str] = typer.Option(None, "--entry-box", help="Existing HDF5/SAV box"),
    save_empty_box: bool = typer.Option(False, "--save-empty-box", help="Save NONE stage"),
    save_potential: bool = typer.Option(False, "--save-potential", help="Save POT stage"),
    save_bounds: bool = typer.Option(False, "--save-bounds", help="Save BND stage"),
    save_nas: bool = typer.Option(False, "--save-nas", help="Save NAS stage"),
    save_gen: bool = typer.Option(False, "--save-gen", help="Save NAS.GEN stage"),
    save_chr: bool = typer.Option(False, "--save-chr", help="Save NAS.CHR stage"),
    stop_after: Optional[str] = typer.Option(None, "--stop-after", help="Stop after stage (dl/none/pot/bnd/nas/gen/chr)"),
    empty_box_only: bool = typer.Option(False, "--empty-box-only", help="Stop after NONE"),
    potential_only: bool = typer.Option(False, "--potential-only", help="Stop after POT"),
    nlfff_only: bool = typer.Option(False, "--nlfff-only", help="Stop after NAS"),
    generic_only: bool = typer.Option(False, "--generic-only", help="Stop after NAS.GEN"),
    use_potential: bool = typer.Option(False, "--use-potential", help="Skip NLFFF"),
    skip_lines: bool = typer.Option(False, "--skip-lines", help="Skip GEN line computation and go directly to CHR"),
    center_vox: bool = typer.Option(False, "--center-vox", help="Trace lines through voxel centers"),
    reduce_passed: Optional[int] = typer.Option(
        None,
        "--reduce-passed",
        min=0,
        max=3,
        help="Expert line tracing reduction bitmask: 0|1|2|3 (overrides --center-vox)",
    ),
    euv: bool = typer.Option(False, "--euv", help="Download AIA EUV maps"),
    uv: bool = typer.Option(False, "--uv", help="Download AIA UV maps"),
    sfq: bool = typer.Option(False, "--sfq", help="Use SFQ disambiguation (method=0)"),
    observer_name: str = typer.Option("earth", "--observer-name", help="Observer identifier stored in output metadata"),
    fov_xc: Optional[float] = typer.Option(None, "--fov-xc", help="Observer/image FOV center X in arcsec"),
    fov_yc: Optional[float] = typer.Option(None, "--fov-yc", help="Observer/image FOV center Y in arcsec"),
    fov_xsize: Optional[float] = typer.Option(None, "--fov-xsize", help="Observer/image FOV width in arcsec"),
    fov_ysize: Optional[float] = typer.Option(None, "--fov-ysize", help="Observer/image FOV height in arcsec"),
    square_fov: bool = typer.Option(False, "--square-fov", help="Observer/image FOV is constrained square"),
    jump2potential: bool = typer.Option(False, "--jump2potential", help="Jump to POT"),
    jump2bounds: bool = typer.Option(False, "--jump2bounds", help="Jump to BND"),
    jump2nlfff: bool = typer.Option(False, "--jump2nlfff", help="Jump to NAS"),
    jump2lines: bool = typer.Option(False, "--jump2lines", help="Jump to GEN"),
    jump2chromo: bool = typer.Option(False, "--jump2chromo", help="Jump to CHR"),
    rebuild: bool = typer.Option(False, "--rebuild", help="Recompute from NONE using entry-box parameters"),
    rebuild_from_none: bool = typer.Option(False, "--rebuild-from-none", help="Start from entry-box NONE-equivalent and run forward"),
    clone_only: bool = typer.Option(False, "--clone-only", help="Normalize/copy entry-box to H5 without recomputation"),
    info: bool = typer.Option(False, "--info", help="Show resolved defaults and exit"),
) -> None:
    cfg = Fov2BoxConfig(
        time=time,
        coords=coords,
        hpc=hpc,
        hgc=hgc,
        hgs=hgs,
        cea=cea,
        top=top,
        box_dims=box_dims,
        dx_km=dx_km,
        pad_frac=pad_frac,
        data_dir=data_dir,
        gxmodel_dir=gxmodel_dir,
        entry_box=entry_box,
        save_empty_box=save_empty_box,
        save_potential=save_potential,
        save_bounds=save_bounds,
        save_nas=save_nas,
        save_gen=save_gen,
        save_chr=save_chr,
        stop_after=stop_after,
        empty_box_only=empty_box_only,
        potential_only=potential_only,
        nlfff_only=nlfff_only,
        generic_only=generic_only,
        use_potential=use_potential,
        skip_lines=skip_lines,
        center_vox=center_vox,
        reduce_passed=reduce_passed,
        euv=euv,
        uv=uv,
        sfq=sfq,
        observer_name=observer_name,
        fov_xc=fov_xc,
        fov_yc=fov_yc,
        fov_xsize=fov_xsize,
        fov_ysize=fov_ysize,
        square_fov=square_fov,
        jump2potential=jump2potential,
        jump2bounds=jump2bounds,
        jump2nlfff=jump2nlfff,
        jump2lines=jump2lines,
        jump2chromo=jump2chromo,
        rebuild=rebuild,
        rebuild_from_none=rebuild_from_none,
        clone_only=clone_only,
        info=info,
    )

    if not any([cfg.hpc, cfg.hgc, cfg.hgs]):
        cfg.hpc = True
    if cfg.cea and cfg.top:
        raise ValueError("Select only one projection: --cea or --top")
    if not cfg.cea and not cfg.top:
        cfg.cea = True
    if cfg.square_fov and cfg.fov_xsize is not None:
        cfg.fov_ysize = cfg.fov_xsize
    _explicit_observer_fov(cfg)

    legacy_stop_flags = [
        ("none", cfg.empty_box_only, "--empty-box-only"),
        ("pot", cfg.potential_only, "--potential-only"),
        ("nas", cfg.nlfff_only, "--nlfff-only"),
        ("gen", cfg.generic_only, "--generic-only"),
    ]
    active_legacy = [item for item in legacy_stop_flags if item[1]]
    if len(active_legacy) > 1:
        names = ", ".join(item[2] for item in active_legacy)
        raise ValueError(f"Conflicting legacy stop flags: {names}. Use only one or use --stop-after.")
    if active_legacy:
        legacy_stop = active_legacy[0][0]
        if cfg.stop_after is not None and _last_stage_tag(cfg.stop_after) != _last_stage_tag(legacy_stop):
            raise ValueError(
                f"Conflicting stop controls: --stop-after={cfg.stop_after} and {active_legacy[0][2]}."
            )
        cfg.stop_after = cfg.stop_after or legacy_stop

    if cfg.info:
        _print_info(cfg)
        return

    if cfg.clone_only and not cfg.entry_box:
        raise ValueError("--clone-only requires --entry-box")
    if cfg.clone_only and (cfg.rebuild or cfg.rebuild_from_none):
        raise ValueError("--clone-only cannot be combined with --rebuild or --rebuild-from-none")
    if cfg.rebuild_from_none and not cfg.entry_box:
        raise ValueError("--rebuild-from-none requires --entry-box")
    if cfg.rebuild and cfg.rebuild_from_none:
        raise ValueError("Use only one of --rebuild or --rebuild-from-none")

    jump_flags_set = any([cfg.jump2potential, cfg.jump2bounds, cfg.jump2nlfff, cfg.jump2lines, cfg.jump2chromo])
    if jump_flags_set and not cfg.entry_box:
        print("Warning: jump2* flags are ignored unless --entry-box is provided.")
    if cfg.rebuild_from_none and jump_flags_set:
        raise ValueError("--rebuild-from-none cannot be combined with jump2* flags")
    if cfg.use_potential and (cfg.jump2nlfff or cfg.jump2bounds):
        raise ValueError("--use-potential cannot be combined with --jump2nlfff or --jump2bounds")
    if cfg.skip_lines and _last_stage_tag(cfg.stop_after) == "GEN":
        raise ValueError("--skip-lines cannot be combined with stop-after GEN (or equivalent).")
    if cfg.skip_lines and cfg.center_vox:
        print("Warning: --center-vox is ignored when --skip-lines is set.")
        cfg.center_vox = False

    entry_loaded: Optional[Dict[str, Any]] = None
    entry_stage: Optional[str] = None
    target_stage = "NONE"
    data_dir_explicit = _flag_explicit_on_cli("--data-dir")
    gxmodel_dir_explicit = _flag_explicit_on_cli("--gxmodel-dir")
    if cfg.entry_box:
        entry_path = Path(cfg.entry_box)
        if not entry_path.exists():
            raise ValueError(f"--entry-box not found: {entry_path}")
        entry_loaded = _load_entry_box_any(entry_path)
        entry_stage = _entry_stage_from_loaded(entry_loaded, entry_path)
        execute_text = _decode_id_text(entry_loaded.get("metadata", {}).get("execute", ""))
        exec_data_dir, exec_gxmodel_dir = _extract_execute_paths(execute_text)
        exec_coords, exec_frame, exec_projection = _extract_execute_geometry(execute_text)
        frame_explicit = any(
            _flag_explicit_on_cli(flag) for flag in ("--hpc", "--hgc", "--hgs")
        )
        proj_explicit = any(_flag_explicit_on_cli(flag) for flag in ("--cea", "--top"))
        if exec_data_dir and not data_dir_explicit:
            cfg.data_dir = exec_data_dir
            print(f"Using --data-dir from entry metadata/execute: {cfg.data_dir}")
            _warn_path_default("data_dir", cfg.data_dir)
        if exec_gxmodel_dir and not gxmodel_dir_explicit:
            cfg.gxmodel_dir = exec_gxmodel_dir
            print(f"Using --gxmodel-dir from entry metadata/execute: {cfg.gxmodel_dir}")
            _warn_path_default("gxmodel_dir", cfg.gxmodel_dir)
        if cfg.coords is None and exec_coords is not None:
            cfg.coords = exec_coords
            print(f"Using --coords from entry metadata/execute: {cfg.coords}")
        if not frame_explicit and exec_frame in ("hpc", "hgc", "hgs"):
            cfg.hpc = exec_frame == "hpc"
            cfg.hgc = exec_frame == "hgc"
            cfg.hgs = exec_frame == "hgs"
        if not proj_explicit and exec_projection in ("cea", "top"):
            cfg.cea = exec_projection == "cea"
            cfg.top = exec_projection == "top"
    target_stage = _detect_target_stage(cfg, entry_stage)

    if cfg.entry_box and not cfg.rebuild and not cfg.rebuild_from_none:
        assert entry_stage is not None
        if not _jump_allowed(entry_stage, target_stage):
            raise ValueError(
                f"Incompatible jump request: entry stage {entry_stage} cannot jump to {target_stage}. "
                "Allowed: backward jumps, forward by one stage, NONE->BND (implicit POT), POT->NAS (implicit BND), and POT->GEN."
            )

    observer_metadata = _observer_metadata_from_entry(entry_loaded, cfg) if entry_loaded is not None else None

    if cfg.clone_only:
        assert entry_loaded is not None
        assert entry_stage is not None
        if jump_flags_set and target_stage != entry_stage:
            raise ValueError(
                f"--clone-only allows only no jump or jump-to-self; got entry {entry_stage} -> {target_stage}."
            )
        # Prefer inferred time from entry path/id for output folder naming.
        obs_time = Time(cfg.time) if cfg.time else None
        if obs_time is None and cfg.entry_box:
            inferred = _infer_time_from_entry_loaded(entry_loaded, Path(cfg.entry_box))
            if inferred:
                obs_time = Time(inferred)
        if obs_time is None:
            raise ValueError("--clone-only requires --time or inferable time from --entry-box filename.")

        out_dir = _stage_output_dir(cfg.gxmodel_dir, obs_time)
        src_id = _decode_id_text(entry_loaded.get("metadata", {}).get("id", "")).strip()
        _, inferred_tag = _split_stage_id(src_id) if src_id else ("", "")
        stage_tag = inferred_tag or _stage_tag_from_stage(entry_stage)
        if src_id:
            out_path = out_dir / f"{src_id}.h5"
        else:
            base = _stage_file_base(obs_time, "CLONE", projection_tag="CEA")
            out_path = _stage_filename(out_dir, base, stage_tag)

        clone_box = dict(entry_loaded)
        meta = dict(clone_box.get("metadata", {}))
        meta.setdefault("execute", _build_execute_cmd(cfg))
        meta.setdefault("id", out_path.stem)
        meta["clone_only"] = "true"
        clone_box["metadata"] = meta
        if observer_metadata is not None:
            clone_box["observer"] = observer_metadata
        clone_box = _normalize_stage_for_h5(clone_box)
        write_b3d_h5(str(out_path), clone_box)
        print("\nCompleted gx-fov2box clone-only. Output file:")
        print(f"- {out_path}")
        return

    import time as time_mod
    import warnings
    import logging

    t_start = time_mod.perf_counter()
    warnings.filterwarnings("ignore", message=".*assume_spherical_screen.*")
    logging.getLogger("reproject").setLevel(logging.WARNING)
    logging.getLogger("sunpy").setLevel(logging.WARNING)

    obs_time, box_dims_resolved, dx_km = _resolve_box_params(cfg)
    cfg.dx_km = dx_km
    resume_mode = bool(cfg.entry_box and not cfg.rebuild and entry_loaded is not None)

    if not resume_mode:
        if cfg.coords is None:
            raise ValueError("--coords is required")
        if sum([cfg.hpc, cfg.hgc, cfg.hgs]) != 1:
            raise ValueError("Select exactly one of --hpc, --hgc, --hgs")

    _warn_impossible_save_requests(cfg, target_stage)
    stage_rank = {"NONE": 0, "POT": 1, "BND": 2, "NAS": 3, "GEN": 4, "CHR": 5}
    start_rank = stage_rank.get(target_stage, 0)

    maps = None
    base_bz_arr = None
    base_ic_arr = None
    bottom_bz_data = None
    vert_current_error = None
    projection_tag = "CEA"
    base = None
    dr3 = None

    if resume_mode:
        assert entry_loaded is not None
        base_group = dict(entry_loaded.get("base", {}))
        if not all(k in base_group for k in ("bx", "by", "bz")):
            raise ValueError("Entry-box resume requires base/bx, base/by, and base/bz.")
        if "ic" not in base_group:
            base_group["ic"] = np.asarray(base_group["bz"], dtype=float).copy()
        if "chromo_mask" not in base_group:
            base_group["chromo_mask"] = decompose(np.asarray(base_group["bz"], dtype=float).T,
                                                  np.asarray(base_group["ic"], dtype=float).T)
        refmaps = dict(entry_loaded.get("refmaps", {}))
        base_bz_arr = np.asarray(base_group["bz"], dtype=float)
        base_ic_arr = np.asarray(base_group["ic"], dtype=float)
        bottom_bz_data = base_bz_arr
        projection_tag = _decode_id_text(entry_loaded.get("metadata", {}).get("projection", "CEA")).upper()

        axis_order = entry_loaded.get("metadata", {}).get("axis_order_3d")
        entry_corona_for_dr = _h5_corona_to_internal_xyz(entry_loaded.get("corona"), axis_order)
        if isinstance(entry_corona_for_dr, dict) and "dr" in entry_corona_for_dr:
            d = np.asarray(entry_corona_for_dr["dr"], dtype=float).reshape(-1)
            dr3 = np.array([d[0], d[min(1, d.size - 1)], d[min(2, d.size - 1)]], dtype=float)
        else:
            dr = float((cfg.dx_km * u.km / sun_consts.radius.to(u.km)).value)
            dr3 = np.array([dr, dr, dr], dtype=float)

        src_id = _decode_id_text(entry_loaded.get("metadata", {}).get("id", "")).strip()
        if src_id:
            base, _ = _split_stage_id(src_id)
        if not base:
            base = Path(cfg.entry_box).stem if cfg.entry_box else _stage_file_base(obs_time, "UNKNOWN", projection_tag=projection_tag)
    else:
        observer = get_earth(obs_time)

        data_dir_path = Path(cfg.data_dir).expanduser().resolve()
        disambig_method = 0 if cfg.sfq else 2
        print("Checking/downloading HMI/AIA data...")
        dl_t0 = time_mod.perf_counter()
        maps, download_info = _load_hmi_maps_from_downloader(
            obs_time,
            data_dir_path,
            cfg.euv,
            cfg.uv,
            disambig_method=disambig_method,
            strict_required=(_last_stage_tag(cfg.stop_after) != "DL"),
        )
        dl_elapsed = time_mod.perf_counter() - dl_t0
        print(f"Done in {dl_elapsed:.2f}s")
        if _last_stage_tag(cfg.stop_after) == "DL":
            print("Stopped after download stage by request.")
            total = time_mod.perf_counter() - t_start
            print(f"Total elapsed: {total:.2f}s")
            return

        rsun = u.Quantity(maps["field"].rsun_meters, u.m).to(u.km)

        if cfg.hpc:
            box_origin = SkyCoord(cfg.coords[0] * u.arcsec, cfg.coords[1] * u.arcsec,
                                  obstime=obs_time, observer=observer, rsun=rsun, frame=Helioprojective)
        elif cfg.hgc:
            box_origin = SkyCoord(lon=cfg.coords[0] * u.deg, lat=cfg.coords[1] * u.deg,
                                  radius=rsun, obstime=obs_time, observer=observer,
                                  frame=HeliographicCarrington)
        else:
            box_origin = SkyCoord(lon=cfg.coords[0] * u.deg, lat=cfg.coords[1] * u.deg,
                                  radius=rsun, obstime=obs_time, observer=observer,
                                  frame=HeliographicStonyhurst)

        box_dims_q = u.Quantity(list(box_dims_resolved)) * u.pix
        box_res = (cfg.dx_km * u.km).to(u.Mm)

        frame_obs = Helioprojective(observer=observer, obstime=obs_time, rsun=rsun)
        frame_hcc = Heliocentric(observer=box_origin, obstime=obs_time)
        box_center = box_origin.transform_to(frame_hcc)
        center_z = box_center.z + (box_dims_q[2] / u.pix * box_res) / 2
        box_center = SkyCoord(x=box_center.x, y=box_center.y, z=center_z, frame=frame_hcc)

        box = Box(frame_obs, box_origin, box_center, box_dims_q, box_res)
        if cfg.top:
            bottom_wcs_header = box.bottom_top_header(dsun_obs=maps["field"].dsun)
            projection_tag = "TOP"
        else:
            bottom_wcs_header = box.bottom_cea_header
            projection_tag = "CEA"
        fov_coords = box.bounds_coords_bl_tr(pad_frac=cfg.pad_frac)

        map_bp, map_bt, map_br = hmi_b2ptr(maps["field"], maps["inclination"], maps["azimuth"])

        def submap_with_fov(_map: Map) -> Map:
            bl = fov_coords[0].transform_to(_map.coordinate_frame)
            tr = fov_coords[1].transform_to(_map.coordinate_frame)
            return _map.submap(bl, top_right=tr)

        map_bp = submap_with_fov(map_bp)
        map_bt = submap_with_fov(map_bt)
        map_br = submap_with_fov(map_br)
        map_cont = submap_with_fov(maps["continuum"])
        map_los = submap_with_fov(maps["magnetogram"])

        # Match GX/IDL base convention: bx := bp, by := -bt, bz := br.
        map_bx = map_bp
        map_by = Map(-map_bt.data, map_bt.meta)
        map_bz = map_br

        bottom_bx = map_bx.reproject_to(bottom_wcs_header, algorithm="exact")
        bottom_by = map_by.reproject_to(bottom_wcs_header, algorithm="exact")
        bottom_bz = map_bz.reproject_to(bottom_wcs_header, algorithm="exact")

        base_bz = map_los.reproject_to(bottom_wcs_header, algorithm="exact")
        base_ic = map_cont.reproject_to(bottom_wcs_header, algorithm="exact")
        base_bz_arr = np.asarray(base_bz.data, dtype=float)
        base_ic_arr = np.asarray(base_ic.data, dtype=float)
        bottom_bz_data = np.asarray(bottom_bz.data, dtype=float)

        index = _build_index_header(bottom_wcs_header, bottom_bz)
        chromo_mask = decompose(base_bz_arr.T, base_ic_arr.T)
        base_group = {
            "bx": bottom_bx.data,
            "by": bottom_by.data,
            "bz": bottom_bz.data,
            "ic": base_ic.data,
            "chromo_mask": chromo_mask,
            "index": index,
        }

        refmaps = {}
        def add_refmap(ref_id: str, smap: Map) -> None:
            smap = submap_with_fov(smap)
            # Use WCS header only to avoid non-ASCII FITS meta from some sources.
            header = smap.wcs.to_header()
            refmaps[ref_id] = {"data": smap.data, "wcs_header": header.tostring(sep="\\n", endcard=True)}

        add_refmap("Bz_reference", maps["magnetogram"])
        add_refmap("Ic_reference", maps["continuum"])

        try:
            vc_header = map_bx.wcs.to_header().tostring(sep="\\n", endcard=True)
            rsun_arcsec = maps["magnetogram"].rsun_obs.to_value(u.arcsec)
            crpix1, crpix2 = map_bx.wcs.wcs.crpix
            cdelt1 = map_bx.scale.axis1.to_value(u.arcsec / u.pix)
            cdelt2 = map_bx.scale.axis2.to_value(u.arcsec / u.pix)
            jz = compute_vertical_current(map_bz.data, map_bx.data, map_by.data,
                                          vc_header, rsun_arcsec,
                                          crpix1=crpix1, crpix2=crpix2,
                                          cdelt1_arcsec=cdelt1, cdelt2_arcsec=cdelt2)
            refmaps["Vert_current"] = {"data": jz, "wcs_header": vc_header}
        except Exception as exc:
            vert_current_error = str(exc)

        for pb in AIA_EUV_PASSBANDS + AIA_UV_PASSBANDS:
            key = f"AIA_{pb}"
            if key in maps:
                add_refmap(key, maps[key])

        obs_dr = (cfg.dx_km * u.km) / rsun
        dr3 = np.array([obs_dr.value, obs_dr.value, obs_dr.value])
        coord_tag = _format_coord_tag(
            box_origin.transform_to(HeliographicCarrington(obstime=obs_time)).lon.to_value(u.deg),
            box_origin.transform_to(HeliographicCarrington(obstime=obs_time)).lat.to_value(u.deg),
        )
        base = _stage_file_base(obs_time, coord_tag, projection_tag=projection_tag)
        observer_metadata = _observer_metadata_from_source_map(maps["field"], cfg)

    empty_grid = np.zeros((box_dims_resolved[0], box_dims_resolved[1], box_dims_resolved[2]), dtype=float)
    default_grid = {"dx": float(dr3[0]), "dy": float(dr3[1]), "dz": np.array([float(dr3[2])], dtype=float)}

    out_dir = _stage_output_dir(cfg.gxmodel_dir, obs_time)
    execute_cmd = _build_execute_cmd(cfg)
    lineage_marker = ""
    entry_stage_for_marker = ""
    if resume_mode and entry_loaded is not None:
        entry_meta = dict(entry_loaded.get("metadata", {}))
        lineage_root = _decode_id_text(entry_meta.get("lineage", "")).strip()
        if not lineage_root:
            entry_id = _decode_id_text(entry_meta.get("id", "")).strip()
            _, entry_stage_tag = _split_stage_id(entry_id) if entry_id else ("", "")
            entry_stage_for_marker = entry_stage_tag or _stage_tag_from_stage(entry_stage or target_stage)
            entry_fmt = "SAV" if str(cfg.entry_box or "").lower().endswith(".sav") else "H5"
            lineage_marker = f"ENTRY.{entry_stage_for_marker}.{entry_fmt}"
            lineage_root = lineage_marker
    else:
        lineage_root = "OBS"

    produced = []
    stage_times = {}

    def finalize() -> None:
        if produced:
            print("\nCompleted gx-fov2box. Output files:")
            for path in produced:
                print(f"- {path}")
        if stage_times:
            print("Stage timings:")
            for stage, seconds in stage_times.items():
                print(f"- {stage}: {seconds:.2f}s")
        total = time_mod.perf_counter() - t_start
        print(f"Total elapsed: {total:.2f}s")

    def save_stage(stage_tag: str, stage_box: dict) -> None:
        if not _should_save_stage(stage_tag, cfg):
            return
        stage_box = dict(stage_box)
        if "base" not in stage_box:
            stage_box["base"] = base_group
        if "refmaps" not in stage_box:
            stage_box["refmaps"] = refmaps
        if "corona" in stage_box:
            corona = stage_box["corona"]
            if "dr" not in corona:
                corona["dr"] = dr3
            voxel_id, corona_base = gx_box2id(
                {"corona": corona, "lines": stage_box.get("lines"), "chromo": stage_box.get("chromo")},
                return_corona_base=True,
            )
            if corona_base is not None:
                corona["corona_base"] = int(corona_base)
            stage_box["corona"] = corona
            grid = dict(default_grid)
            if voxel_id is not None:
                grid["voxel_id"] = voxel_id
            else:
                grid["voxel_id"] = gx_box2id({"corona": {"bx": empty_grid, "by": empty_grid, "bz": empty_grid, "dr": dr3}})
            if "chromo" in stage_box and "dz" in stage_box["chromo"]:
                grid["dz"] = stage_box["chromo"]["dz"]
            stage_box["grid"] = grid
        elif "grid" not in stage_box:
            stage_box["grid"] = default_grid
        stage_id = f"{base}.{stage_tag}"
        lineage_suffix = _canonical_lineage_suffix(stage_tag)
        metadata = {
            "execute": execute_cmd,
            "id": stage_id,
            "lineage": (
                f"{lineage_marker}->{_lineage_delta_from_entry(entry_stage_for_marker or stage_tag, stage_tag)}.h5"
                if lineage_marker
                else _merge_lineage(lineage_root, lineage_suffix)
            ),
            "disambiguation": "SFQ" if cfg.sfq else "HMI",
            "projection": _decode_id_text(projection_tag).upper(),
            "axis_order_2d": "yx",
            "axis_order_3d": "zyx",
            "vector_layout": "split_components",
        }
        if vert_current_error:
            metadata["vert_current_error"] = vert_current_error
        stage_box["metadata"] = metadata
        if observer_metadata is not None:
            stage_box["observer"] = dict(observer_metadata)
        stage_box = _normalize_stage_for_h5(stage_box)
        out_path = _stage_filename(out_dir, base, stage_tag)
        write_b3d_h5(str(out_path), stage_box)
        print(f"Saved {stage_tag}: {out_path}")
        produced.append(out_path)

    if start_rank <= 0 and (cfg.save_empty_box or cfg.empty_box_only or cfg.stop_after in ("none", "empty", "empty_box")):
        print("Computing NONE model...")
        t0 = time_mod.perf_counter()
        # NONE stage should preserve boundary conditions at z=0 and keep z>0 empty.
        # Canonical 3D order in H5 is (nz, ny, nx).
        nz, ny, nx = box_dims_resolved[2], box_dims_resolved[1], box_dims_resolved[0]
        corona_bx = np.zeros((nz, ny, nx), dtype=float)
        corona_by = np.zeros((nz, ny, nx), dtype=float)
        corona_bz = np.zeros((nz, ny, nx), dtype=float)
        corona_bx[0, :, :] = np.asarray(base_group["bx"], dtype=float)
        corona_by[0, :, :] = np.asarray(base_group["by"], dtype=float)
        corona_bz[0, :, :] = np.asarray(base_group["bz"], dtype=float)
        stage_box = {
            "corona": {
                "bx": corona_bx,
                "by": corona_by,
                "bz": corona_bz,
                "dr": dr3,
                "attrs": {"model_type": "none"},
            }
        }
        save_stage("NONE", stage_box)
        stage_times["NONE"] = time_mod.perf_counter() - t0
        if cfg.empty_box_only or _last_stage_tag(cfg.stop_after) == "NONE":
            finalize()
            return

    def _make_pot_box() -> dict:
        bnddata = np.asarray(bottom_bz_data, dtype=float).copy()
        bnddata[np.isnan(bnddata)] = 0.0
        maglib_lff = MagFieldLinFFF()
        maglib_lff.set_field(bnddata)
        pot_res = maglib_lff.LFFF_cube(nz=box_dims_resolved[2], alpha=0.0)
        return {
            "bx": pot_res["by"].swapaxes(0, 1),
            "by": pot_res["bx"].swapaxes(0, 1),
            "bz": pot_res["bz"].swapaxes(0, 1),
            "dr": dr3,
            "attrs": {"model_type": "pot"},
        }

    def _make_bnd_from_pot(pot_box: dict) -> dict:
        bnd = {
            "bx": np.array(pot_box["bx"], copy=True),
            "by": np.array(pot_box["by"], copy=True),
            "bz": np.array(pot_box["bz"], copy=True),
            "dr": dr3,
            "attrs": {"model_type": "bnd"},
        }
        # Internal cube layout is (x, y, z); overwrite the z=0 boundary with observed base vectors.
        bnd["bx"][:, :, 0] = np.asarray(base_group["bx"], dtype=float).T
        bnd["by"][:, :, 0] = np.asarray(base_group["by"], dtype=float).T
        bnd["bz"][:, :, 0] = np.asarray(base_group["bz"], dtype=float).T
        return bnd

    def _run_nlfff_from_bnd(bnd_box: dict) -> dict:
        maglib = MagFieldProcessor()
        maglib.load_cube_vars({
            "bx": bnd_box["by"].swapaxes(0, 1),
            "by": bnd_box["bx"].swapaxes(0, 1),
            "bz": bnd_box["bz"].swapaxes(0, 1),
        })
        res_nlf = maglib.NLFFF()
        return {
            "bx": res_nlf["by"].swapaxes(0, 1),
            "by": res_nlf["bx"].swapaxes(0, 1),
            "bz": res_nlf["bz"].swapaxes(0, 1),
            "dr": dr3,
            "attrs": {"model_type": "nlfff"},
        }

    active_jump = None
    if cfg.entry_box and not cfg.rebuild:
        if target_stage == "POT":
            active_jump = "potential"
        elif target_stage == "BND":
            active_jump = "bounds"
        elif target_stage == "NAS":
            active_jump = "nlfff"
        elif target_stage == "GEN":
            active_jump = "lines"
        elif target_stage == "CHR":
            active_jump = "chromo"

    entry_corona = None
    entry_model = None
    entry_lines = None
    entry_axis_order = None
    if active_jump and entry_loaded is not None:
        entry_axis_order = entry_loaded.get("metadata", {}).get("axis_order_3d")
        entry_corona = entry_loaded.get("corona")
        if entry_corona:
            entry_corona = _h5_corona_to_internal_xyz(entry_corona, entry_axis_order)
        if entry_corona:
            entry_model = entry_corona.get("attrs", {}).get("model_type")
        entry_lines = entry_loaded.get("lines")
        if entry_lines is None:
            # Backward compatibility with legacy GEN files that stored line keys under chromo.
            entry_lines = entry_loaded.get("chromo")

    goto_lines = active_jump in ("lines", "chromo")
    goto_chromo = active_jump == "chromo" or cfg.skip_lines

    if goto_lines:
        if not entry_corona:
            raise ValueError("--jump2lines/--jump2chromo requires --entry-box with corona model_type=pot|nlfff")
        nlfff_box = entry_corona

    if not goto_lines:
        pot_box = None
        bnd_box = None

        if active_jump in ("potential", "bounds", "nlfff"):
            if not entry_corona:
                raise ValueError("--entry-box does not contain corona vectors required by selected jump/recompute action.")
            if active_jump == "potential":
                pot_box = entry_corona
                if str(pot_box.get("attrs", {}).get("model_type", "")).lower() != "pot":
                    pot_box["attrs"] = {"model_type": "pot"}
            elif active_jump == "bounds":
                bnd_box = entry_corona
                if str(bnd_box.get("attrs", {}).get("model_type", "")).lower() not in ("bnd", "bounds"):
                    bnd_box["attrs"] = {"model_type": "bnd"}
            elif active_jump == "nlfff":
                # Target NAS can start from POT/BND/NAS entry cubes.
                if entry_stage == "POT":
                    pot_box = entry_corona
                    bnd_box = _make_bnd_from_pot(pot_box)
                elif entry_stage == "BND":
                    bnd_box = entry_corona
                elif entry_stage in ("NAS", "GEN", "CHR"):
                    nlfff_box = entry_corona
                    if str(nlfff_box.get("attrs", {}).get("model_type", "")).lower() != "nlfff":
                        nlfff_box["attrs"] = {"model_type": "nlfff"}
                else:
                    raise ValueError(f"Unsupported entry stage for --jump2nlfff path: {entry_stage}")

        if pot_box is None and bnd_box is None and "nlfff_box" not in locals():
            print("Computing POT model...")
            t0 = time_mod.perf_counter()
            pot_box = _make_pot_box()
            save_stage("POT", {"corona": pot_box})
            stage_times["POT"] = time_mod.perf_counter() - t0
            if cfg.potential_only or _last_stage_tag(cfg.stop_after) == "POT":
                finalize()
                return
        elif pot_box is not None:
            print("Preparing POT model from entry data...")
            t0 = time_mod.perf_counter()
            save_stage("POT", {"corona": pot_box})
            stage_times["POT"] = time_mod.perf_counter() - t0
            if cfg.potential_only or _last_stage_tag(cfg.stop_after) == "POT":
                finalize()
                return

        if bnd_box is None and pot_box is not None and not cfg.use_potential:
            print("Computing BND model...")
            t0 = time_mod.perf_counter()
            bnd_box = _make_bnd_from_pot(pot_box)
            save_stage("BND", {"corona": bnd_box})
            stage_times["BND"] = time_mod.perf_counter() - t0
            if _last_stage_tag(cfg.stop_after) == "BND":
                finalize()
                return
        elif bnd_box is not None:
            print("Preparing BND model from entry data...")
            t0 = time_mod.perf_counter()
            save_stage("BND", {"corona": bnd_box})
            stage_times["BND"] = time_mod.perf_counter() - t0
            if _last_stage_tag(cfg.stop_after) == "BND":
                finalize()
                return

        if "nlfff_box" not in locals():
            if cfg.use_potential:
                if pot_box is None:
                    pot_box = _make_pot_box()
                nlfff_box = {
                    "bx": np.array(pot_box["bx"], copy=True),
                    "by": np.array(pot_box["by"], copy=True),
                    "bz": np.array(pot_box["bz"], copy=True),
                    "dr": dr3,
                    "attrs": {"model_type": "pot"},
                }
            else:
                if bnd_box is None:
                    if pot_box is None:
                        pot_box = _make_pot_box()
                    bnd_box = _make_bnd_from_pot(pot_box)
                print("Computing NAS model...")
                t0 = time_mod.perf_counter()
                nlfff_box = _run_nlfff_from_bnd(bnd_box)
                save_stage("NAS", {"corona": nlfff_box})
                stage_times["NAS"] = time_mod.perf_counter() - t0
                if cfg.nlfff_only or _last_stage_tag(cfg.stop_after) == "NAS":
                    finalize()
                    return
        elif str(nlfff_box.get("attrs", {}).get("model_type", "")).lower() == "nlfff":
            # Jumped to NAS from an existing NAS/GEN/CHR entry box.
            print("Preparing NAS model from entry data...")
            t0 = time_mod.perf_counter()
            save_stage("NAS", {"corona": nlfff_box})
            stage_times["NAS"] = time_mod.perf_counter() - t0
            if cfg.nlfff_only or _last_stage_tag(cfg.stop_after) == "NAS":
                finalize()
                return

    stage_prefix = "NAS"
    is_pot_skip_path = (
        resume_mode
        and entry_stage == "POT"
        and target_stage in ("GEN", "CHR")
    )
    if str(nlfff_box.get("attrs", {}).get("model_type", "")).lower() == "pot" or is_pot_skip_path:
        stage_prefix = "POT"
    gen_stage_tag = f"{stage_prefix}.GEN"
    chr_stage_tag = f"{stage_prefix}.CHR"

    if goto_chromo:
        required_line_keys = [
            "codes", "apex_idx", "start_idx", "end_idx", "seed_idx",
            "av_field", "phys_length", "voxel_status",
        ]
        if entry_lines and all(k in entry_lines for k in required_line_keys):
            lines = entry_lines
        else:
            # Direct CHR jump without GEN metadata: produce chromo-only augmentation.
            lines = None
        compute_lines_time = 0.0
        if cfg.skip_lines:
            print("Skipping GEN line computation by request...")
    else:
        print(f"Computing {gen_stage_tag} model...")
        t0 = time_mod.perf_counter()
        maglib = MagFieldProcessor()
        maglib.load_cube_vars({
            "bx": nlfff_box["by"].swapaxes(0, 1),
            "by": nlfff_box["bx"].swapaxes(0, 1),
            "bz": nlfff_box["bz"].swapaxes(0, 1),
        })
        resolved_reduce_passed = cfg.reduce_passed if cfg.reduce_passed is not None else (0 if cfg.center_vox else 1)
        lines = _lines_quiet(maglib, seeds=None, reduce_passed=resolved_reduce_passed)
        compute_lines_time = time_mod.perf_counter() - t0

    if lines is not None:
        chr_stage_tag = f"{stage_prefix}.GEN.CHR"

    header = _make_header(maps["field"]) if maps is not None else {}
    chromo_box = combo_model(nlfff_box, dr3, base_bz_arr.T, base_ic_arr.T)
    for k in ["codes", "apex_idx", "start_idx", "end_idx", "seed_idx",
              "av_field", "phys_length", "voxel_status"]:
        if lines is not None and k in lines:
            chromo_box[k] = lines[k]
    if "phys_length" in chromo_box:
        chromo_box["phys_length"] *= dr3[0]
    chromo_box["attrs"] = header

    if not goto_chromo:
        lines_group = _make_lines_group(lines, dr3)
        save_stage(gen_stage_tag, {"corona": nlfff_box, "lines": lines_group})
        stage_times[gen_stage_tag] = compute_lines_time
        if cfg.generic_only or _last_stage_tag(cfg.stop_after) == "GEN":
            finalize()
            return

    print(f"Computing {chr_stage_tag} model...")
    t0 = time_mod.perf_counter()
    chr_stage = {"corona": nlfff_box, "chromo": _make_chromo_group(chromo_box)}
    if lines is not None:
        chr_stage["lines"] = _make_lines_group(lines, dr3)
    save_stage(chr_stage_tag, chr_stage)
    stage_times[chr_stage_tag] = time_mod.perf_counter() - t0

    finalize()


if __name__ == "__main__":
    app()
