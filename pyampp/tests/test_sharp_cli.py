from __future__ import annotations

import sys
import types
from pathlib import Path

if "typer" not in sys.modules:
    class _DummyTyper:
        def __init__(self, *args, **kwargs):
            pass

        def command(self, *args, **kwargs):
            def _decorator(func):
                return func

            return _decorator

    sys.modules["typer"] = types.SimpleNamespace(
        Typer=_DummyTyper,
        Option=lambda default=None, *args, **kwargs: default,
    )

if "PyQt5" not in sys.modules:
    pyqt5_module = types.ModuleType("PyQt5")
    qtwidgets_module = types.ModuleType("PyQt5.QtWidgets")

    class _DummyQApplication:
        @staticmethod
        def instance():
            return None

    qtwidgets_module.QApplication = _DummyQApplication
    qtwidgets_module.QMessageBox = object
    pyqt5_module.QtWidgets = qtwidgets_module
    sys.modules["PyQt5"] = pyqt5_module
    sys.modules["PyQt5.QtWidgets"] = qtwidgets_module

if "pyAMaFiL" not in sys.modules:
    pyamafil_module = types.ModuleType("pyAMaFiL")
    linfff_module = types.ModuleType("pyAMaFiL.mag_field_lin_fff")
    proc_module = types.ModuleType("pyAMaFiL.mag_field_proc")

    class _DummyMagFieldLinFFF:
        pass

    class _DummyMagFieldProcessor:
        pass

    linfff_module.MagFieldLinFFF = _DummyMagFieldLinFFF
    proc_module.MagFieldProcessor = _DummyMagFieldProcessor
    proc_module.mfp_util_invert_index_array = lambda *args, **kwargs: None
    proc_module.mfp_util_transpose_index = lambda arr, *args, **kwargs: arr

    pyamafil_module.mag_field_lin_fff = linfff_module
    pyamafil_module.mag_field_proc = proc_module
    sys.modules["pyAMaFiL"] = pyamafil_module
    sys.modules["pyAMaFiL.mag_field_lin_fff"] = linfff_module
    sys.modules["pyAMaFiL.mag_field_proc"] = proc_module

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.time import Time
from sunpy.coordinates import HeliographicCarrington, HeliographicStonyhurst, get_earth
from sunpy.map import make_fitswcs_header

from pyampp.gxbox.boxutils import read_b3d_h5, write_b3d_h5
from pyampp.sharp import cli as sharp_cli
from pyampp.sharp import drms as sharp_drms
from pyampp.sharp import entry_box as sharp_entry_box
from pyampp.sharp import radio_adapter


def _decode_text(value):
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "ignore")
    if isinstance(value, np.ndarray) and value.shape == ():
        item = value.item()
        if isinstance(item, (bytes, bytearray)):
            return item.decode("utf-8", "ignore")
        return str(item)
    return str(value)


def _fake_sharp_client(records, segments):
    class _Client:
        def query(self, *args, **kwargs):
            return records, segments

    return _Client()


def _write_sharp_like_fits(path: Path, data: np.ndarray, *, obs_time: Time) -> None:
    observer = get_earth(obs_time)
    observer_hgs = observer.transform_to(HeliographicStonyhurst(obstime=obs_time))
    coord = sharp_entry_box.SkyCoord(
        lon=30.0 * u.deg,
        lat=-10.0 * u.deg,
        radius=696000 * u.km,
        frame=HeliographicCarrington(observer=observer, obstime=obs_time),
    )
    header = make_fitswcs_header(
        data.shape,
        coord,
        scale=u.Quantity([0.1, 0.1], u.deg / u.pix),
        projection_code="CEA",
    )
    header = fits.Header(header)
    header["DATE-OBS"] = obs_time.isot
    header["DATE_OBS"] = obs_time.isot
    header["T_REC"] = "2025.11.26_15:36:00_TAI"
    header["DSUN_OBS"] = observer.radius.to_value(u.m)
    header["RSUN_REF"] = 6.96e8
    header["HGLN_OBS"] = observer_hgs.lon.to_value(u.deg)
    header["HGLT_OBS"] = observer_hgs.lat.to_value(u.deg)
    fits.PrimaryHDU(data=np.asarray(data, dtype=np.float32), header=header).writeto(path, overwrite=True)


def _write_freq_source_fits(path: Path, freqs_hz: np.ndarray) -> None:
    primary = fits.PrimaryHDU(data=np.zeros((1, len(freqs_hz), 2, 2), dtype=np.float32))
    primary.header["CRVAL3"] = float(freqs_hz[0])
    primary.header["CDELT3"] = float(freqs_hz[1] - freqs_hz[0]) if len(freqs_hz) > 1 else 0.0
    primary.header["NAXIS3"] = int(len(freqs_hz))
    primary.header["CTYPE3"] = "FREQ"
    primary.header["CUNIT3"] = "Hz"
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="cfreqs", format="E", array=np.asarray(freqs_hz, dtype=np.float32)),
            fits.Column(name="cdelts", format="E", array=np.zeros(len(freqs_hz), dtype=np.float32)),
        ]
    )
    fits.HDUList([primary, table]).writeto(path, overwrite=True)


def _minimal_index_header_text() -> str:
    header = fits.Header()
    header["CRVAL1"] = 30.0
    header["CRVAL2"] = -10.0
    header["DATE-OBS"] = "2025-11-26T15:36:00.000"
    header["DATE_OBS"] = "2025-11-26T15:36:00.000"
    header["DSUN_OBS"] = get_earth(Time("2025-11-26T15:36:00")).radius.to_value(u.m)
    header["HGLN_OBS"] = 0.0
    header["HGLT_OBS"] = 1.2
    header["RSUN_REF"] = 6.96e8
    header["CTYPE1"] = "CRLN-CEA"
    header["CTYPE2"] = "CRLT-CEA"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    return header.tostring(sep="\n", endcard=True)


def test_find_nearest_sharp_records_selects_nearest_timestamp_and_all_patches():
    idx = [
        "hmi.sharp_cea_720s[1111][2025.11.26_15:36:00_TAI]",
        "hmi.sharp_cea_720s[2222][2025.11.26_15:36:00_TAI]",
        "hmi.sharp_cea_720s[3333][2025.11.26_14:24:00_TAI]",
    ]
    key_df = pd.DataFrame(
        {
            "T_REC": [
                "2025.11.26_15:36:00_TAI",
                "2025.11.26_15:36:00_TAI",
                "2025.11.26_14:24:00_TAI",
            ],
            "T_OBS": [
                "2025.11.26_15:35:55_TAI",
                "2025.11.26_15:35:55_TAI",
                "2025.11.26_14:23:55_TAI",
            ],
            "HARPNUM": [1111, 2222, 3333],
        },
        index=idx,
    )
    seg_df = pd.DataFrame(
        {
            "Br": ["br.fits", "br.fits", "br.fits"],
            "Bt": ["bt.fits", "bt.fits", "bt.fits"],
            "Bp": ["bp.fits", "bp.fits", "bp.fits"],
            "continuum": ["ic.fits", "ic.fits", "ic.fits"],
            "magnetogram": ["los.fits", "los.fits", "los.fits"],
            "bitmap": ["bitmap.fits", "", ""],
        },
        index=idx,
    )
    selection = sharp_drms.find_nearest_sharp_records(
        "2025-11-26T15:47:52",
        client=_fake_sharp_client(key_df, seg_df),
        max_time_delta_hours=12.0,
    )

    assert selection.selected_time == "2025-11-26T15:36:00.000"
    assert [record.harpnum for record in selection.records] == [1111, 2222]
    assert "bitmap" in selection.records[0].available_segments


def test_find_nearest_sharp_records_filters_requested_harpnums():
    idx = [
        "hmi.sharp_cea_720s[1111][2025.11.26_15:36:00_TAI]",
        "hmi.sharp_cea_720s[2222][2025.11.26_15:36:00_TAI]",
    ]
    key_df = pd.DataFrame(
        {
            "T_REC": ["2025.11.26_15:36:00_TAI", "2025.11.26_15:36:00_TAI"],
            "T_OBS": ["2025.11.26_15:35:55_TAI", "2025.11.26_15:35:55_TAI"],
            "HARPNUM": [1111, 2222],
        },
        index=idx,
    )
    seg_df = pd.DataFrame(
        {
            "Br": ["br.fits", "br.fits"],
            "Bt": ["bt.fits", "bt.fits"],
            "Bp": ["bp.fits", "bp.fits"],
            "continuum": ["ic.fits", "ic.fits"],
            "magnetogram": ["los.fits", "los.fits"],
            "bitmap": ["", ""],
        },
        index=idx,
    )
    selection = sharp_drms.find_nearest_sharp_records(
        "2025-11-26T15:47:52",
        client=_fake_sharp_client(key_df, seg_df),
        max_time_delta_hours=12.0,
        harpnums=[2222],
    )

    assert [record.harpnum for record in selection.records] == [2222]


def test_find_nearest_sharp_records_errors_when_no_record_is_available():
    key_df = pd.DataFrame({"T_REC": [], "T_OBS": [], "HARPNUM": []})
    seg_df = pd.DataFrame({"Br": [], "Bt": [], "Bp": [], "continuum": [], "magnetogram": []})

    with pytest.raises(RuntimeError, match="No SHARP records found"):
        sharp_drms.find_nearest_sharp_records(
            "2025-11-26T15:47:52",
            client=_fake_sharp_client(key_df, seg_df),
            max_time_delta_hours=12.0,
        )


def test_build_sharp_entry_box_writes_none_stage_and_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(sharp_entry_box, "decompose", lambda bz, ic: np.full_like(bz, 7, dtype=np.int16))
    obs_time = Time("2025-11-26T15:36:00")
    bp = np.arange(12, dtype=np.float32).reshape(3, 4)
    bt = bp + 10.0
    br = bp + 20.0
    continuum = np.full((3, 4), 5.0, dtype=np.float32)
    los = np.full((3, 4), 6.0, dtype=np.float32)

    segment_paths = {}
    for name, data in {
        "Bp": bp,
        "Bt": bt,
        "Br": br,
        "continuum": continuum,
        "magnetogram": los,
    }.items():
        path = tmp_path / f"{name}.fits"
        _write_sharp_like_fits(path, data, obs_time=obs_time)
        segment_paths[name] = path

    result = sharp_entry_box.build_sharp_entry_box(
        segment_paths,
        tmp_path / "entry.h5",
        harpnum=4321,
        t_rec=obs_time.isot,
        requested_time="2025-11-26T15:47:52",
        nz=5,
    )
    box = read_b3d_h5(str(result.output_path))

    assert np.array_equal(box["base"]["bx"], bp)
    assert np.array_equal(box["base"]["by"], -bt)
    assert np.array_equal(box["base"]["bz"], br)
    assert np.array_equal(box["base"]["ic"], continuum)
    assert np.array_equal(box["base"]["chromo_mask"], np.full((3, 4), 7, dtype=np.int32))
    assert np.asarray(box["corona"]["bx"]).shape == (5, 3, 4)
    assert _decode_text(box["metadata"]["disambiguation"]) == "SHARP"
    assert int(box["metadata"]["harpnum"]) == 4321
    assert "CRLN-CEA" in _decode_text(box["base"]["index"])
    assert float(box["observer"]["fov"]["xsize_arcsec"]) > 0.0
    assert result.dx_km > 0.0


def test_build_sharp_entry_box_requires_required_segments(tmp_path):
    with pytest.raises(ValueError, match="Missing required SHARP segment path: continuum"):
        sharp_entry_box.build_sharp_entry_box(
            {"Br": tmp_path / "br.fits", "Bt": tmp_path / "bt.fits", "Bp": tmp_path / "bp.fits", "magnetogram": tmp_path / "los.fits"},
            tmp_path / "entry.h5",
            harpnum=1,
            t_rec="2025-11-26T15:36:00",
            requested_time="2025-11-26T15:47:52",
            nz=5,
        )


def test_build_radio_model_dict_transposes_hdf5_payloads(tmp_path):
    nz, ny, nx = 3, 2, 2
    corona_bx = np.arange(nz * ny * nx, dtype=np.float32).reshape(nz, ny, nx)
    corona_by = corona_bx + 100.0
    corona_bz = corona_bx + 200.0
    chromo_bx = np.arange(1 * ny * nx, dtype=np.float32).reshape(1, ny, nx) + 300.0
    chromo_by = chromo_bx + 100.0
    chromo_bz = chromo_bx + 200.0
    model_path = tmp_path / "model.h5"
    write_b3d_h5(
        str(model_path),
        {
            "base": {
                "bx": np.zeros((ny, nx), dtype=np.float32),
                "by": np.zeros((ny, nx), dtype=np.float32),
                "bz": np.zeros((ny, nx), dtype=np.float32),
                "index": _minimal_index_header_text(),
            },
            "corona": {
                "bx": corona_bx,
                "by": corona_by,
                "bz": corona_bz,
                "dr": np.array([0.01, 0.01, 0.01], dtype=np.float64),
                "corona_base": 1,
            },
            "chromo": {
                "bx": chromo_bx,
                "by": chromo_by,
                "bz": chromo_bz,
                "dz": np.ones((3, ny, nx), dtype=np.float32) * 0.01,
                "chromo_idx": np.array([0, 1, 2, 3], dtype=np.int64),
                "chromo_n": np.ones(4, dtype=np.float32),
                "n_p": np.ones(4, dtype=np.float32) * 2,
                "n_hi": np.ones(4, dtype=np.float32) * 3,
                "chromo_t": np.ones(4, dtype=np.float32) * 4,
                "chromo_layers": np.array([1], dtype=np.int32),
            },
            "lines": {
                "av_field": np.ones((nz, ny, nx), dtype=np.float32) * 9.0,
                "phys_length": np.ones((nz, ny, nx), dtype=np.float32) * 0.25,
                "voxel_status": np.ones((nz, ny, nx), dtype=np.uint8) * 4,
            },
            "observer": {
                "name": "sdo",
                "fov": {
                    "xc_arcsec": 100.0,
                    "yc_arcsec": -50.0,
                    "xsize_arcsec": 20.0,
                    "ysize_arcsec": 10.0,
                },
            },
        },
    )

    model_dict, header, context = radio_adapter.build_radio_model_dict(model_path)
    geometry = radio_adapter.derive_radio_geometry(context["box"], nx=context["nx"], ny=context["ny"])

    assert model_dict["bcube"].shape == (nx, ny, nz, 3)
    assert np.array_equal(model_dict["bcube"][0, 0, :, 0], np.array([0.0, 4.0, 8.0], dtype=np.float32))
    assert model_dict["chromo_bcube"].shape == (nx, ny, 1, 3)
    assert np.array_equal(model_dict["avfield"].shape, (nx, ny, nz))
    assert header["CRVAL1"] == 30.0
    assert geometry.box_dx == 10.0
    assert geometry.box_dy == 5.0


def test_build_radio_model_dict_rebuilds_flattened_line_cubes(tmp_path):
    nz, ny, nx = 3, 2, 2
    av_field_zyx = np.arange(nz * ny * nx, dtype=np.float32).reshape(nz, ny, nx)
    phys_length_zyx = av_field_zyx + 10.0
    voxel_status_zyx = np.ones((nz, ny, nx), dtype=np.uint8) * 4
    model_path = tmp_path / "model_flat_lines.h5"
    write_b3d_h5(
        str(model_path),
        {
            "base": {
                "bx": np.zeros((ny, nx), dtype=np.float32),
                "by": np.zeros((ny, nx), dtype=np.float32),
                "bz": np.zeros((ny, nx), dtype=np.float32),
                "index": _minimal_index_header_text(),
            },
            "corona": {
                "bx": np.arange(nz * ny * nx, dtype=np.float32).reshape(nz, ny, nx),
                "by": np.arange(nz * ny * nx, dtype=np.float32).reshape(nz, ny, nx) + 100.0,
                "bz": np.arange(nz * ny * nx, dtype=np.float32).reshape(nz, ny, nx) + 200.0,
                "dr": np.array([0.01, 0.01, 0.01], dtype=np.float64),
                "corona_base": 1,
            },
            "chromo": {
                "bx": np.arange(1 * ny * nx, dtype=np.float32).reshape(1, ny, nx) + 300.0,
                "by": np.arange(1 * ny * nx, dtype=np.float32).reshape(1, ny, nx) + 400.0,
                "bz": np.arange(1 * ny * nx, dtype=np.float32).reshape(1, ny, nx) + 500.0,
                "dz": np.ones((3, ny, nx), dtype=np.float32) * 0.01,
                "chromo_idx": np.array([0, 1, 2, 3], dtype=np.int64),
                "chromo_n": np.ones(4, dtype=np.float32),
                "n_p": np.ones(4, dtype=np.float32) * 2,
                "n_hi": np.ones(4, dtype=np.float32) * 3,
                "chromo_t": np.ones(4, dtype=np.float32) * 4,
                "chromo_layers": np.array([1], dtype=np.int32),
            },
            "lines": {
                "av_field": av_field_zyx.T.reshape(-1, order="F"),
                "phys_length": phys_length_zyx.T.reshape(-1, order="F"),
                "voxel_status": voxel_status_zyx.T.reshape(-1, order="F"),
            },
            "observer": {
                "name": "sdo",
                "fov": {
                    "xc_arcsec": 0.0,
                    "yc_arcsec": 0.0,
                    "xsize_arcsec": 20.0,
                    "ysize_arcsec": 10.0,
                },
            },
        },
    )

    model_dict, _, _ = radio_adapter.build_radio_model_dict(model_path)

    np.testing.assert_array_equal(model_dict["avfield"], av_field_zyx.transpose((2, 1, 0)))
    np.testing.assert_array_equal(model_dict["physlength"], phys_length_zyx.transpose((2, 1, 0)))
    np.testing.assert_array_equal(model_dict["status"], voxel_status_zyx.transpose((2, 1, 0)))


def test_load_radio_config_requires_all_keys(tmp_path):
    config_path = tmp_path / "radio.yml"
    config_path.write_text("tbase: 1\nnbase: 2\nq0: 3\na: 4\nforce_isothermal: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required keys: b"):
        radio_adapter.load_radio_config(config_path)


def test_resolve_freqs_ghz_accepts_list_and_comma_formats():
    assert sharp_cli._resolve_freqs_ghz(["[5.0, 8.4, 17.0]"]) == [5.0, 8.4, 17.0]
    assert sharp_cli._resolve_freqs_ghz(["5.0,8.4,17.0"]) == [5.0, 8.4, 17.0]
    assert sharp_cli._resolve_freqs_ghz(["[5.0,", "8.4,", "17.0]"]) == [5.0, 8.4, 17.0]
    assert sharp_cli._resolve_freqs_ghz(["5.0", "8.4", "17.0"]) == [5.0, 8.4, 17.0]
    assert sharp_cli._resolve_freqs_ghz([["5.0"], ["8.4", "17.0"]]) == [5.0, 8.4, 17.0]


def test_cli_run_uses_default_freqs_when_flag_is_omitted(tmp_path, monkeypatch):
    freq_source = tmp_path / "tot_flux.fits"
    _write_freq_source_fits(freq_source, np.array([1.25e9, 1.575e9, 1.9e9], dtype=np.float64))
    monkeypatch.setattr(sharp_cli, "DEFAULT_FREQ_SOURCE_FITS", freq_source)
    sharp_cli._default_freqs_ghz.cache_clear()

    mw_lib = tmp_path / "libmw.dylib"
    ebtel = tmp_path / "ebtel.sav"
    radio_config_path = tmp_path / "radio.yml"
    mw_lib.write_text("mw", encoding="utf-8")
    ebtel.write_text("ebtel", encoding="utf-8")
    radio_config_path.write_text(
        "tbase: 1\nnbase: 2\nq0: 3\na: 4\nb: 5\nforce_isothermal: 1\n",
        encoding="utf-8",
    )

    selection = sharp_drms.SharpSelection(
        series=sharp_drms.SHARP_CEA_SERIES,
        requested_time="2025-11-26T15:47:52.000",
        selected_t_rec="2025.11.26_15:36:00_TAI",
        selected_time="2025-11-26T15:36:00.000",
        delta_seconds=712.0,
        records=(
            sharp_drms.SharpRecord(
                record="rec1",
                harpnum=1111,
                t_rec="2025.11.26_15:36:00_TAI",
                sample_time="2025-11-26T15:36:00.000",
                delta_seconds=712.0,
                available_segments=sharp_drms.REQUIRED_SEGMENTS,
            ),
        ),
    )

    monkeypatch.setattr(sharp_cli, "find_nearest_sharp_records", lambda *args, **kwargs: selection)
    monkeypatch.setattr(
        sharp_cli,
        "download_sharp_record_segments",
        lambda record, output_dir, force_download=False: {
            key: Path(output_dir) / f"{key}.fits" for key in sharp_drms.REQUIRED_SEGMENTS
        },
    )
    monkeypatch.setattr(
        sharp_cli,
        "build_sharp_entry_box",
        lambda *args, **kwargs: sharp_entry_box.SharpEntryBoxBuild(
            output_path=tmp_path / f"HARP{kwargs['harpnum']}.NONE.h5",
            model_id=f"HARP{kwargs['harpnum']}.NONE",
            base_id=f"HARP{kwargs['harpnum']}",
            harpnum=int(kwargs["harpnum"]),
            t_rec="2025-11-26T15:36:00.000",
            dx_km=1200.0,
            box_dims=(4, 3, 5),
        ),
    )
    monkeypatch.setattr(sharp_cli, "_run_gx_pipeline", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sharp_cli,
        "_final_chr_path",
        lambda model_dir, base_id, t_rec: tmp_path / f"{base_id}.h5",
    )

    captured = {}

    def _fake_compute(model_path, *, mw_lib, ebtel_file, freqs_ghz, radio_config):
        captured["freqs_ghz"] = list(freqs_ghz)
        return (
            {
                "TI": np.ones((4, 3, len(freqs_ghz)), dtype=np.float64),
                "TV": np.zeros((4, 3, len(freqs_ghz)), dtype=np.float64),
            },
            {
                "geometry": radio_adapter.RadioGeometry(4, 3, 0.0, 0.0, 5.0, 6.0),
                "header": fits.Header({"DATE-OBS": "2025-11-26T15:36:00.000", "DATE_OBS": "2025-11-26T15:36:00.000"}),
                "box": {"observer": {"name": "sdo", "fov": {"xc_arcsec": 0.0, "yc_arcsec": 0.0, "xsize_arcsec": 20.0, "ysize_arcsec": 18.0}}},
            },
        )

    monkeypatch.setattr(sharp_cli, "compute_radio_emission", _fake_compute)
    monkeypatch.setattr(sharp_cli, "write_radio_bundle", lambda output_path, emission, freqs_ghz, metadata: Path(output_path))
    monkeypatch.setattr(
        sharp_cli,
        "write_radio_fits",
        lambda output_dir, emission, freqs_ghz, box, header, geometry, file_stem, extra_header=None: [Path(output_dir) / f"{file_stem}.TI.1.000GHz.fits"],
    )

    rc = sharp_cli.run(
        [
            "--time",
            "2025-11-26T15:47:52",
            "--nz",
            "5",
            "--mw-lib",
            str(mw_lib),
            "--ebtel",
            str(ebtel),
            "--radio-config",
            str(radio_config_path),
            "--out-root",
            str(tmp_path / "out"),
        ]
    )

    assert rc == 0
    assert captured["freqs_ghz"] == [1.25, 1.575, 1.9]
    sharp_cli._default_freqs_ghz.cache_clear()


def test_cli_run_continues_after_per_patch_failure(tmp_path, monkeypatch):
    mw_lib = tmp_path / "libmw.dylib"
    ebtel = tmp_path / "ebtel.sav"
    radio_config_path = tmp_path / "radio.yml"
    mw_lib.write_text("mw", encoding="utf-8")
    ebtel.write_text("ebtel", encoding="utf-8")
    radio_config_path.write_text(
        "tbase: 1\nnbase: 2\nq0: 3\na: 4\nb: 5\nforce_isothermal: 1\n",
        encoding="utf-8",
    )

    selection = sharp_drms.SharpSelection(
        series=sharp_drms.SHARP_CEA_SERIES,
        requested_time="2025-11-26T15:47:52.000",
        selected_t_rec="2025.11.26_15:36:00_TAI",
        selected_time="2025-11-26T15:36:00.000",
        delta_seconds=712.0,
        records=(
            sharp_drms.SharpRecord(
                record="rec1",
                harpnum=1111,
                t_rec="2025.11.26_15:36:00_TAI",
                sample_time="2025-11-26T15:36:00.000",
                delta_seconds=712.0,
                available_segments=sharp_drms.REQUIRED_SEGMENTS,
            ),
            sharp_drms.SharpRecord(
                record="rec2",
                harpnum=2222,
                t_rec="2025.11.26_15:36:00_TAI",
                sample_time="2025-11-26T15:36:00.000",
                delta_seconds=712.0,
                available_segments=sharp_drms.REQUIRED_SEGMENTS,
            ),
        ),
    )

    monkeypatch.setattr(sharp_cli, "find_nearest_sharp_records", lambda *args, **kwargs: selection)
    monkeypatch.setattr(
        sharp_cli,
        "download_sharp_record_segments",
        lambda record, output_dir, force_download=False: {
            key: Path(output_dir) / f"{key}.fits" for key in sharp_drms.REQUIRED_SEGMENTS
        },
    )
    monkeypatch.setattr(
        sharp_cli,
        "build_sharp_entry_box",
        lambda *args, **kwargs: sharp_entry_box.SharpEntryBoxBuild(
            output_path=tmp_path / f"HARP{kwargs['harpnum']}.NONE.h5",
            model_id=f"HARP{kwargs['harpnum']}.NONE",
            base_id=f"HARP{kwargs['harpnum']}",
            harpnum=int(kwargs["harpnum"]),
            t_rec="2025-11-26T15:36:00.000",
            dx_km=1200.0,
            box_dims=(4, 3, 5),
        ),
    )
    monkeypatch.setattr(sharp_cli, "_run_gx_pipeline", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sharp_cli,
        "_final_chr_path",
        lambda model_dir, base_id, t_rec: tmp_path / f"{base_id}.h5",
    )

    captured_harpnums = []

    def _fake_compute_with_failure(model_path, *, mw_lib, ebtel_file, freqs_ghz, radio_config):
        model_text = str(model_path)
        if "2222" in model_text:
            raise RuntimeError("synthetic failure")
        captured_harpnums.append(1111)
        return (
            {
                "TI": np.ones((4, 3, len(freqs_ghz)), dtype=np.float64),
                "TV": np.zeros((4, 3, len(freqs_ghz)), dtype=np.float64),
            },
            {
                "geometry": radio_adapter.RadioGeometry(4, 3, 0.0, 0.0, 5.0, 6.0),
                "header": fits.Header({"DATE-OBS": "2025-11-26T15:36:00.000", "DATE_OBS": "2025-11-26T15:36:00.000"}),
                "box": {"observer": {"name": "sdo", "fov": {"xc_arcsec": 0.0, "yc_arcsec": 0.0, "xsize_arcsec": 20.0, "ysize_arcsec": 18.0}}},
            },
        )

    monkeypatch.setattr(sharp_cli, "compute_radio_emission", _fake_compute_with_failure)
    monkeypatch.setattr(sharp_cli, "write_radio_bundle", lambda output_path, emission, freqs_ghz, metadata: Path(output_path))
    monkeypatch.setattr(
        sharp_cli,
        "write_radio_fits",
        lambda output_dir, emission, freqs_ghz, box, header, geometry, file_stem, extra_header=None: [Path(output_dir) / f"{file_stem}.TI.1.000GHz.fits"],
    )

    rc = sharp_cli.run(
        [
            "--time",
            "2025-11-26T15:47:52",
            "--freq-ghz",
            "5.0",
            "--nz",
            "5",
            "--mw-lib",
            str(mw_lib),
            "--ebtel",
            str(ebtel),
            "--radio-config",
            str(radio_config_path),
            "--out-root",
            str(tmp_path / "out"),
        ]
    )

    assert rc == 1
    assert captured_harpnums == [1111]


def test_cli_run_passes_use_potential_to_pipeline(tmp_path, monkeypatch):
    mw_lib = tmp_path / "libmw.dylib"
    ebtel = tmp_path / "ebtel.sav"
    radio_config_path = tmp_path / "radio.yml"
    mw_lib.write_text("mw", encoding="utf-8")
    ebtel.write_text("ebtel", encoding="utf-8")
    radio_config_path.write_text(
        "tbase: 1\nnbase: 2\nq0: 3\na: 4\nb: 5\nforce_isothermal: 1\n",
        encoding="utf-8",
    )

    selection = sharp_drms.SharpSelection(
        series=sharp_drms.SHARP_CEA_SERIES,
        requested_time="2025-11-26T15:47:52.000",
        selected_t_rec="2025.11.26_15:36:00_TAI",
        selected_time="2025-11-26T15:36:00.000",
        delta_seconds=712.0,
        records=(
            sharp_drms.SharpRecord(
                record="rec1",
                harpnum=1111,
                t_rec="2025.11.26_15:36:00_TAI",
                sample_time="2025-11-26T15:36:00.000",
                delta_seconds=712.0,
                available_segments=sharp_drms.REQUIRED_SEGMENTS,
            ),
        ),
    )

    monkeypatch.setattr(sharp_cli, "find_nearest_sharp_records", lambda *args, **kwargs: selection)
    monkeypatch.setattr(
        sharp_cli,
        "download_sharp_record_segments",
        lambda record, output_dir, force_download=False: {
            key: Path(output_dir) / f"{key}.fits" for key in sharp_drms.REQUIRED_SEGMENTS
        },
    )
    monkeypatch.setattr(
        sharp_cli,
        "build_sharp_entry_box",
        lambda *args, **kwargs: sharp_entry_box.SharpEntryBoxBuild(
            output_path=tmp_path / f"HARP{kwargs['harpnum']}.NONE.h5",
            model_id=f"HARP{kwargs['harpnum']}.NONE",
            base_id=f"HARP{kwargs['harpnum']}",
            harpnum=int(kwargs["harpnum"]),
            t_rec="2025-11-26T15:36:00.000",
            dx_km=1200.0,
            box_dims=(4, 3, 5),
        ),
    )
    pipeline_kwargs = {}

    def _fake_run_gx_pipeline(entry_box, *, t_rec, gxmodel_dir, observer_name="sdo", use_potential=False):
        pipeline_kwargs["use_potential"] = bool(use_potential)

    monkeypatch.setattr(sharp_cli, "_run_gx_pipeline", _fake_run_gx_pipeline)
    monkeypatch.setattr(
        sharp_cli,
        "_final_chr_path",
        lambda model_dir, base_id, t_rec: tmp_path / f"{base_id}.POT.GEN.CHR.h5",
    )
    monkeypatch.setattr(
        sharp_cli,
        "compute_radio_emission",
        lambda *args, **kwargs: (
            {
                "TI": np.ones((4, 3, 1), dtype=np.float64),
                "TV": np.zeros((4, 3, 1), dtype=np.float64),
            },
            {
                "geometry": radio_adapter.RadioGeometry(4, 3, 0.0, 0.0, 5.0, 6.0),
                "header": fits.Header({"DATE-OBS": "2025-11-26T15:36:00.000", "DATE_OBS": "2025-11-26T15:36:00.000"}),
                "box": {"observer": {"name": "sdo", "fov": {"xc_arcsec": 0.0, "yc_arcsec": 0.0, "xsize_arcsec": 20.0, "ysize_arcsec": 18.0}}},
            },
        ),
    )
    monkeypatch.setattr(sharp_cli, "write_radio_bundle", lambda output_path, emission, freqs_ghz, metadata: Path(output_path))
    monkeypatch.setattr(
        sharp_cli,
        "write_radio_fits",
        lambda output_dir, emission, freqs_ghz, box, header, geometry, file_stem, extra_header=None: [Path(output_dir) / f"{file_stem}.TI.1.000GHz.fits"],
    )

    rc = sharp_cli.run(
        [
            "--time",
            "2025-11-26T15:47:52",
            "--freq-ghz",
            "5.0",
            "--nz",
            "5",
            "--use-potential",
            "--mw-lib",
            str(mw_lib),
            "--ebtel",
            str(ebtel),
            "--radio-config",
            str(radio_config_path),
            "--out-root",
            str(tmp_path / "out"),
        ]
    )

    assert rc == 0
    assert pipeline_kwargs["use_potential"] is True
