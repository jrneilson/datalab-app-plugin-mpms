from pathlib import Path

import numpy as np
import pytest

from datalab_app_plugin_mpms import __version__
from datalab_app_plugin_mpms.blocks import MPMSBlock
from datalab_app_plugin_mpms.utils import (
    N_A_MU_B_EMU,
    convert_magnetization,
    inverse,
    parse_qd_dat,
    susceptibility,
)

MPMS3_FILE = Path(__file__).parent / "data" / "CER_1_011_2_CoPdN_thinfilm_CurieTemp.dat"


def test_version():
    assert __version__
    assert MPMSBlock.version == __version__


def test_parse_mpms3():
    df, header = parse_qd_dat(MPMS3_FILE)
    assert header["SAMPLE_MASS"] == "0.347"
    assert header["SAMPLE_MOLECULAR_WEIGHT"] == ""
    assert len(df) == 683 - 44
    for column in ("Time Stamp (sec)", "Temperature (K)", "Magnetic Field (Oe)", "Moment (emu)"):
        assert column in df.columns
    # empty AC columns are dropped
    assert "AC Moment (emu)" not in df.columns


def test_conversions_and_error_propagation():
    moment, err = np.array([1e-3]), np.array([1e-4])  # 10 % error
    sample = dict(mass_mg=1.0, mass_err_mg=0.1, formula_weight=100.0, density=2.0, density_err=0.2)

    m, s = convert_magnetization(moment, err, "emu/g", **sample)
    assert m[0] == pytest.approx(1.0)
    assert s[0] == pytest.approx(np.sqrt(2) * 0.1)

    m, s = convert_magnetization(moment, err, "μB/f.u.", **sample)
    assert m[0] == pytest.approx(100 / 5584.94, rel=1e-5)
    assert s[0] / m[0] == pytest.approx(np.sqrt(2) * 0.1)

    m, s = convert_magnetization(moment, err, "T", **sample)
    assert m[0] == pytest.approx(4 * np.pi * 1e-4 * 2.0)
    assert s[0] / m[0] == pytest.approx(np.sqrt(3) * 0.1)

    with pytest.raises(ValueError):
        convert_magnetization(moment, err, "T", mass_mg=1.0)


def test_susceptibility():
    field = np.array([0.0, 1000.0])
    m_ub = np.array([1.0, 1.0])
    chi, _ = susceptibility(m_ub, np.zeros(2), field, "μB/f.u.")
    assert np.isnan(chi[0])
    assert chi[1] == pytest.approx(N_A_MU_B_EMU / 1000)

    # 1 emu/cm³ in 1 Oe is χ_SI = 4π
    chi, _ = susceptibility(np.array([4 * np.pi * 1e-4]), np.zeros(1), np.array([1.0]), "T")
    assert chi[0] == pytest.approx(4 * np.pi)

    inv, inv_err = inverse(np.array([2.0]), np.array([0.2]))
    assert inv[0] == pytest.approx(0.5)
    assert inv_err[0] == pytest.approx(0.05)


def test_block_plot_and_settings():
    block = MPMSBlock(item_id="test")
    block.generate_plot(location=MPMS3_FILE)
    assert block.data["bokeh_plot_data"]
    assert block.data["mass_mg"] == pytest.approx(0.347)
    assert block._number("formula_weight") is None
    assert block.data["moment_column"] == "Moment (emu)"
    assert block.data["error_column"] == "M. Std. Err. (emu)"

    block.process_events(
        [
            {"event_name": "update_settings", "unit": "emu/g", "block_id": block.block_id},
            {"event_name": "update_settings", "mass_err_mg": "0.005"},
            {"event_name": "update_settings", "quantity": "1/χ"},
            {"event_name": "update_settings", "filter_field_min": "500"},
        ]
    )
    assert "errors" not in block.data
    df, _ = parse_qd_dat(MPMS3_FILE)
    out, unit = block.compute(df)
    assert unit == "emu/g"
    assert (out["Field (Oe)"] >= 500).all()
    assert "M (emu/g)" in out and "M (μB/f.u.)" not in out

    # μB/f.u. without a formula weight falls back to emu with a warning
    block.process_events({"event_name": "update_settings", "unit": "μB/f.u."})
    with pytest.warns(UserWarning, match="formula weight"):
        _, unit = block.compute(df)
    assert unit == "emu"

    block.process_events({"event_name": "update_settings", "mass_mg": "-1"})
    assert block.data["errors"]


def test_cleared_filters_stay_cleared():
    """Emulate datalab's web app, which merges each server response into its stored copy of
    the block (`Object.assign`) and sends that copy back with the next event."""
    stored = MPMSBlock(item_id="test").to_web()
    for event in (
        {"filter_field_min": "1", "filter_field_max": "2"},
        {"filter_field_min": ""},
        {"filter_field_max": ""},
    ):
        block = MPMSBlock.from_web(dict(stored))
        block.process_events({"event_name": "update_settings", **event})
        stored.update(block.to_web())
    block = MPMSBlock.from_web(dict(stored))
    assert block._number("filter_field_min") is None
    assert block._number("filter_field_max") is None
