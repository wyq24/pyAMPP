from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.request import urlopen

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.time import Time

from pyampp.data.downloader import SDOImageDownloader
from pyampp.util.config import jsoc_notify_email


SHARP_CEA_SERIES = "hmi.sharp_cea_720s"
REQUIRED_SEGMENTS = ("Br", "Bt", "Bp", "continuum", "magnetogram")
OPTIONAL_SEGMENTS = ("bitmap",)


@dataclass(frozen=True)
class SharpRecord:
    record: str
    harpnum: int
    t_rec: str
    sample_time: str
    delta_seconds: float
    available_segments: tuple[str, ...]


@dataclass(frozen=True)
class SharpSelection:
    series: str
    requested_time: str
    selected_t_rec: str
    selected_time: str
    delta_seconds: float
    records: tuple[SharpRecord, ...]


def _get_client(client: Any = None):
    if client is not None:
        return client
    try:
        import drms
    except Exception as exc:  # pragma: no cover - exercised only without dependency
        raise RuntimeError("drms is required to query SHARP products") from exc
    return drms.Client()


def _lower_map(columns: Iterable[Any]) -> dict[str, Any]:
    return {str(col).strip().lower(): col for col in columns}


def _segment_available(seg_df, index_value, segment: str) -> bool:
    if seg_df is None:
        return False
    columns = _lower_map(seg_df.columns)
    col = columns.get(str(segment).strip().lower())
    if col is None:
        return False
    raw = seg_df.at[index_value, col]
    text = str(raw).strip().lower()
    return text not in {"", "nan", "none"}


def _coerce_harpnum(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except Exception:
        return None


def _filename_time_token(time_text: str) -> str:
    return SDOImageDownloader._filename_time_token(time_text)


def _download_from_url(url: str, target_path: Path, timeout: int = 60) -> None:
    tmp = target_path.with_name(target_path.name + ".part")
    try:
        with urlopen(url, timeout=timeout) as response, open(tmp, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
        tmp.replace(target_path)
    finally:
        tmp.unlink(missing_ok=True)


def _build_header_from_record(client: Any, record: str) -> fits.Header:
    key_df = client.query(record, key="**ALL**", rec_index=False, convert_numeric=True)
    if key_df is None or len(key_df) == 0:
        raise RuntimeError(f"No DRMS metadata returned for {record}")
    return SDOImageDownloader._build_header_from_keyword_row(key_df.iloc[0])


def _normalize_exported_fits(client: Any, record: str, raw_path: Path, local_path: Path) -> Path:
    header = _build_header_from_record(client, record)
    data = SDOImageDownloader._extract_normalized_image_data(raw_path)
    fits.PrimaryHDU(data=data, header=header).writeto(local_path, overwrite=True)
    return local_path


def _required_segment_query_list() -> list[str]:
    return list(REQUIRED_SEGMENTS) + list(OPTIONAL_SEGMENTS)


def _make_sharp_recordset(t1: Time, t2: Time, series: str) -> str:
    duration_sec = max(1, int(round((Time(t2) - Time(t1)).to_value("sec"))))
    return f"{series}[][{SDOImageDownloader._jsoc_time_string(t1, series)}/{duration_sec}s]"


def find_nearest_sharp_records(
    request_time: Time | str,
    *,
    max_time_delta_hours: float = 12.0,
    harpnums: Sequence[int] | None = None,
    client: Any = None,
    series: str = SHARP_CEA_SERIES,
) -> SharpSelection:
    target = request_time if isinstance(request_time, Time) else Time(request_time)
    client = _get_client(client)
    max_delta_seconds = float(max_time_delta_hours) * 3600.0
    if max_delta_seconds <= 0:
        raise ValueError("--max-time-delta-hours must be positive")

    t1 = target - float(max_time_delta_hours) * u.hour
    t2 = target + float(max_time_delta_hours) * u.hour
    recordset = _make_sharp_recordset(t1, t2, series)
    key_df, seg_df = client.query(
        recordset,
        key=["T_REC", "T_OBS", "HARPNUM"],
        seg=_required_segment_query_list(),
        rec_index=True,
        convert_numeric=False,
    )
    if key_df is None or len(key_df) == 0:
        raise RuntimeError(
            f"No SHARP records found within {max_time_delta_hours:.2f} hours of {target.utc.isot}"
        )

    key_cols = _lower_map(key_df.columns)
    t_rec_col = key_cols.get("t_rec")
    t_obs_col = key_cols.get("t_obs")
    harp_col = key_cols.get("harpnum")
    if t_rec_col is None:
        raise RuntimeError("DRMS SHARP query did not return T_REC")
    if harp_col is None:
        raise RuntimeError("DRMS SHARP query did not return HARPNUM")

    requested_harpnums = None
    if harpnums:
        requested_harpnums = {int(value) for value in harpnums}

    matches: list[dict[str, Any]] = []
    for record_index in key_df.index:
        if not all(_segment_available(seg_df, record_index, segment) for segment in REQUIRED_SEGMENTS):
            continue
        row = key_df.loc[record_index]
        harpnum = _coerce_harpnum(row[harp_col])
        if harpnum is None:
            continue
        t_rec_text = str(row[t_rec_col]).strip()
        t_rec = SDOImageDownloader._parse_jsoc_time(t_rec_text)
        if t_rec is None and t_obs_col is not None:
            t_rec = SDOImageDownloader._parse_jsoc_time(row[t_obs_col])
        if t_rec is None:
            continue
        delta_seconds = abs((t_rec.tai - target.tai).to_value("sec"))
        if delta_seconds > max_delta_seconds + 1e-6:
            continue
        available_segments = tuple(
            segment for segment in _required_segment_query_list() if _segment_available(seg_df, record_index, segment)
        )
        matches.append(
            {
                "record": str(record_index),
                "harpnum": harpnum,
                "t_rec": t_rec_text,
                "sample_time": t_rec.utc.isot,
                "selection_key": t_rec.tai.isot,
                "delta_seconds": float(delta_seconds),
                "available_segments": available_segments,
            }
        )

    if not matches:
        raise RuntimeError(
            f"No SHARP records with required segments found within {max_time_delta_hours:.2f} hours of {target.utc.isot}"
        )

    selected = min(matches, key=lambda item: (item["delta_seconds"], item["sample_time"]))
    selection_key = selected["selection_key"]
    selected_group = [item for item in matches if item["selection_key"] == selection_key]
    if requested_harpnums is not None:
        selected_group = [item for item in selected_group if int(item["harpnum"]) in requested_harpnums]
        if not selected_group:
            requested_text = ", ".join(str(value) for value in sorted(requested_harpnums))
            raise RuntimeError(
                f"Nearest SHARP timestamp {selected['sample_time']} has no requested HARPNUM values: {requested_text}"
            )

    records = tuple(
        SharpRecord(
            record=item["record"],
            harpnum=int(item["harpnum"]),
            t_rec=str(item["t_rec"]),
            sample_time=str(item["sample_time"]),
            delta_seconds=float(item["delta_seconds"]),
            available_segments=tuple(item["available_segments"]),
        )
        for item in sorted(selected_group, key=lambda entry: int(entry["harpnum"]))
    )
    return SharpSelection(
        series=str(series),
        requested_time=target.utc.isot,
        selected_t_rec=str(selected["t_rec"]),
        selected_time=str(selected["sample_time"]),
        delta_seconds=float(selected["delta_seconds"]),
        records=records,
    )


def download_sharp_record_segments(
    record: SharpRecord,
    output_dir: Path | str,
    *,
    client: Any = None,
    series: str = SHARP_CEA_SERIES,
    force_download: bool = False,
    poll_seconds: int = 5,
) -> dict[str, Path]:
    client = _get_client(client)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    downloaded: dict[str, Path] = {}
    for segment in _required_segment_query_list():
        is_optional = segment in OPTIONAL_SEGMENTS
        if is_optional and segment not in record.available_segments:
            continue
        token = _filename_time_token(record.t_rec)
        local_name = f"{series}.{token}.HARP{int(record.harpnum)}.{segment}.fits"
        local_path = target_dir / local_name
        if local_path.exists() and not force_download:
            downloaded[segment] = local_path
            continue

        export_record = f"{record.record}{{{segment}}}"
        request = client.export(
            export_record,
            method="url_quick",
            protocol="fits",
            email=jsoc_notify_email(),
            filenamefmt=False,
        )
        request.wait(sleep=max(1, int(poll_seconds)))
        urls_df = request.urls
        if urls_df is None or len(urls_df) == 0:
            if is_optional:
                continue
            raise RuntimeError(f"DRMS export returned no URL for {export_record}")
        raw_url = str(urls_df.iloc[0]["url"])

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp_file:
            raw_path = Path(tmp_file.name)
        try:
            _download_from_url(raw_url, raw_path)
            _normalize_exported_fits(client, record.record, raw_path, local_path)
        finally:
            raw_path.unlink(missing_ok=True)
        downloaded[segment] = local_path

    missing = [segment for segment in REQUIRED_SEGMENTS if segment not in downloaded]
    if missing:
        raise RuntimeError(
            f"Missing required SHARP segments for HARP{record.harpnum}: {', '.join(missing)}"
        )
    return downloaded
