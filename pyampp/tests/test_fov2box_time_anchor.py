from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.time import Time

from pyampp.gxbox import gx_fov2box


class _FakeMap:
    def __init__(self, date: str):
        self.date = Time(date)


class _FakeWCS:
    def to_header(self):
        header = fits.Header()
        header["CTYPE1"] = "HPLN-TAN"
        header["CTYPE2"] = "HPLT-TAN"
        return header


class _FakeRefMap:
    def __init__(self, date: str):
        self.date = Time(date)
        self.wcs = _FakeWCS()
        self.rsun_obs = 972.3 * u.arcsec
        self.rsun_meters = 6.96e8 * u.m
        self.observer_coordinate = None


class _FakeDownloader:
    calls: list[dict[str, object]] = []

    def __init__(self, time, uv=True, euv=True, hmi=True, data_dir=None, backend="drms", force_download=False, poll_seconds=5):
        self.time = Time(time)
        self.uv = uv
        self.euv = euv
        self.hmi = hmi
        self.existence_report = {
            "hmi_b": {seg: True for seg in ("field", "inclination", "azimuth", "disambig")},
            "hmi_m": {"magnetogram": True},
            "hmi_ic": {"continuum": True},
            "euv": {pb: True for pb in gx_fov2box.AIA_EUV_PASSBANDS},
            "uv": {pb: True for pb in gx_fov2box.AIA_UV_PASSBANDS},
        }
        type(self).calls.append(
            {
                "time": self.time.isot,
                "uv": uv,
                "euv": euv,
                "hmi": hmi,
                "backend": backend,
                "force_download": force_download,
            }
        )

    def download_images(self):
        if self.hmi:
            return {
                "field": "field.fits",
                "inclination": "inclination.fits",
                "azimuth": "azimuth.fits",
                "disambig": "disambig.fits",
                "continuum": "continuum.fits",
                "magnetogram": "magnetogram.fits",
            }
        files = {}
        if self.euv:
            for pb in gx_fov2box.AIA_EUV_PASSBANDS:
                files[str(pb)] = f"aia_{pb}.fits"
        if self.uv:
            for pb in gx_fov2box.AIA_UV_PASSBANDS:
                files[str(pb)] = f"aia_{pb}.fits"
        return files


def _fake_map_loader(path):
    if path == "field.fits":
        return _FakeMap("2025-11-26T15:34:31.400")
    return _FakeMap("2025-11-26T15:34:31.400")


def test_load_hmi_maps_anchors_context_downloads_to_resolved_hmi_time():
    _FakeDownloader.calls.clear()
    requested = Time("2025-11-26T15:47:52")
    with patch.object(gx_fov2box, "SDOImageDownloader", _FakeDownloader), patch.object(
        gx_fov2box, "load_sunpy_map_compat", side_effect=_fake_map_loader
    ), patch.object(gx_fov2box, "hmi_disambig", side_effect=lambda azimuth, _disambig, method=2: azimuth):
        maps, info = gx_fov2box._load_hmi_maps_from_downloader(
            requested,
            Path("/tmp"),
            euv=True,
            uv=True,
            download_backend="drms",
            force_download=False,
        )

    assert len(_FakeDownloader.calls) == 2
    assert _FakeDownloader.calls[0]["time"] == requested.isot
    assert _FakeDownloader.calls[0]["hmi"] is True
    assert _FakeDownloader.calls[0]["euv"] is False
    assert _FakeDownloader.calls[0]["uv"] is False

    assert _FakeDownloader.calls[1]["time"] == "2025-11-26T15:34:31.400"
    assert _FakeDownloader.calls[1]["hmi"] is False
    assert _FakeDownloader.calls[1]["euv"] is True
    assert _FakeDownloader.calls[1]["uv"] is True

    assert info["requested_obs_time"] == requested.isot
    assert info["resolved_obs_time"] == "2025-11-26T15:34:31.400"
    assert maps["field"].date.isot == "2025-11-26T15:34:31.400"
    assert "AIA_94" in maps
    assert "AIA_1700" in maps


def test_refmap_wcs_header_preserves_date_obs():
    header_text = gx_fov2box._refmap_wcs_header(_FakeRefMap("2025-11-26T15:34:31.400"))
    header = fits.Header.fromstring(header_text, sep="\n")
    assert header["DATE-OBS"] == "2025-11-26T15:34:31.400"
    assert header["DATE_OBS"] == "2025-11-26T15:34:31.400"
    assert header["RSUN_OBS"] == 972.3
    assert header["RSUN_REF"] == 6.96e8


def _fake_sav_box_with_refmaps():
    rec_dtype = [
        ("ID", object),
        ("DATA", object),
        ("XC", object),
        ("YC", object),
        ("DX", object),
        ("DY", object),
        ("TIME", object),
        ("XUNITS", object),
        ("YUNITS", object),
        ("RSUN", object),
        ("B0", object),
        ("L0", object),
    ]
    rec = np.empty(1, dtype=rec_dtype)
    rec["ID"][0] = b"AIA_171"
    rec["DATA"][0] = np.arange(6, dtype=np.float32).reshape(2, 3)
    rec["XC"][0] = 10.0
    rec["YC"][0] = -20.0
    rec["DX"][0] = 0.6
    rec["DY"][0] = 0.6
    rec["TIME"][0] = b"2025-11-26T15:34:33.350"
    rec["XUNITS"][0] = b"arcsec"
    rec["YUNITS"][0] = b"arcsec"
    rec["RSUN"][0] = 972.5
    rec["B0"][0] = 1.44
    rec["L0"][0] = 44.92

    ids = np.empty(1, dtype=object)
    ids[0] = b"slot0"
    ptrs = np.empty(1, dtype=object)
    ptrs[0] = rec

    pointer = np.empty(1, dtype=[("IDS", object), ("PTRS", object)])
    pointer["IDS"][0] = ids
    pointer["PTRS"][0] = ptrs

    omap = np.empty(1, dtype=[("POINTER", object)])
    omap["POINTER"][0] = pointer

    refmaps = np.empty(1, dtype=[("OMAP", object)])
    refmaps["OMAP"][0] = omap

    box = np.empty(1, dtype=[("REFMAPS", object), ("ID", object), ("EXECUTE", object)])
    box["REFMAPS"][0] = refmaps
    box["ID"][0] = b"hmi.M_720s.20251126_153431.W28S12CR.CEA.NAS.CHR"
    box["EXECUTE"][0] = b"gx_fov2box, '26-Nov-25 15:47:52'"
    return box


def test_load_entry_box_any_restores_sav_refmaps(tmp_path):
    fake_box = _fake_sav_box_with_refmaps()
    with patch.object(gx_fov2box, "readsav", return_value={"box": fake_box}):
        loaded = gx_fov2box._load_entry_box_any(tmp_path / "entry.sav")

    assert "refmaps" in loaded
    assert list(loaded["refmaps"]) == ["AIA_171"]
    payload = loaded["refmaps"]["AIA_171"]
    assert payload["data"].shape == (2, 3)

    header = fits.Header.fromstring(payload["wcs_header"], sep="\n")
    assert header["DATE-OBS"] == "2025-11-26T15:34:33.350"
    assert header["DATE_OBS"] == "2025-11-26T15:34:33.350"
    assert header["RSUN_OBS"] == 972.5
    assert header["HGLT_OBS"] == 1.44
    assert header["HGLN_OBS"] == 44.92
