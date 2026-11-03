"""Parsing and unit conversion for Quantum Design MPMS / MPMS3 `.dat` files."""

from pathlib import Path

import numpy as np
import pandas as pd

N_A_MU_B_EMU = 6.02214076e23 * 9.2740100783e-21
"""Avogadro's number times the Bohr magneton in emu: converts emu/mol to μB per formula unit."""

MU0_EMU_PER_CM3_TO_T = 4 * np.pi * 1e-4
"""μ0 M in T from M in emu/cm³ (1 emu/cm³ = 10³ A/m)."""

MOMENT_CANDIDATES = (
    "Moment (emu)",
    "DC Moment Free Ctr (emu)",
    "DC Moment Fixed Ctr (emu)",
    "Long Moment (emu)",
)
ERROR_CANDIDATES = {
    "Moment (emu)": "M. Std. Err. (emu)",
    "DC Moment Free Ctr (emu)": "DC Moment Err Free Ctr (emu)",
    "DC Moment Fixed Ctr (emu)": "DC Moment Err Fixed Ctr (emu)",
    "Long Moment (emu)": "Long Scan Std Dev",
}
TEMPERATURE_CANDIDATES = ("Temperature (K)", "Average Temp (K)")
FIELD_CANDIDATES = ("Magnetic Field (Oe)", "Field (Oe)")
TIME_CANDIDATES = ("Time Stamp (sec)", "Time")

UNITS = ("emu", "emu/g", "μB/f.u.", "T")
"""Magnetization units: raw moment, mass magnetization, per formula unit, and μ0M."""

SUSCEPTIBILITY_UNITS = {
    "emu": "emu/Oe",
    "emu/g": "emu/(g·Oe)",
    "μB/f.u.": "emu/(mol·Oe)",
    "T": "SI, volume",
}
INVERSE_SUSCEPTIBILITY_UNITS = {
    "emu": "Oe/emu",
    "emu/g": "g·Oe/emu",
    "μB/f.u.": "mol·Oe/emu",
    "T": "SI, volume",
}


def parse_qd_dat(location: str | Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Parse a Quantum Design `.dat` file into a DataFrame of the `[Data]` section
    and a dictionary of the `INFO` header entries (keyed by e.g. `SAMPLE_MASS`).

    Columns that contain no data are dropped.
    """
    with open(location, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    try:
        data_start = next(i for i, line in enumerate(lines) if line.strip() == "[Data]")
    except StopIteration:
        raise ValueError(f"No [Data] section found in {location}; not a Quantum Design .dat file?")

    header: dict[str, str] = {}
    for line in lines[:data_start]:
        parts = [p.strip() for p in line.split(",")]
        if parts[0] == "INFO" and len(parts) >= 3:
            header[parts[-1]] = ",".join(parts[1:-1])

    df = pd.read_csv(location, skiprows=data_start + 1, encoding="utf-8", encoding_errors="replace")
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(axis="columns", how="all")
    return df, header


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return the columns that contain numeric data."""
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()]


def detect_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Return the first candidate column that is present with numeric data."""
    available = numeric_columns(df)
    return next((c for c in candidates if c in available), None)


def detect_error_column(df: pd.DataFrame, moment_column: str | None) -> str | None:
    """Return the standard error column paired with the given moment column, if present."""
    error_column = ERROR_CANDIDATES.get(moment_column or "")
    return error_column if error_column in numeric_columns(df) else None


def convert_magnetization(
    moment: np.ndarray,
    moment_err: np.ndarray | None,
    unit: str,
    mass_mg: float | None = None,
    mass_err_mg: float | None = None,
    formula_weight: float | None = None,
    density: float | None = None,
    density_err: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a moment (emu) and its uncertainty to the requested unit,
    propagating uncertainties in moment, mass and density in quadrature
    (formula weight is treated as exact).

    Parameters:
        moment: Measured moment (emu).
        moment_err: Standard error of the moment (emu), or None.
        unit: One of `UNITS`.
        mass_mg: Sample mass (mg); required for all units other than emu.
        mass_err_mg: Uncertainty in sample mass (mg).
        formula_weight: Formula weight (g/mol); required for μB/f.u.
        density: Sample density (g/cm³); required for T.
        density_err: Uncertainty in density (g/cm³).

    Returns:
        The converted magnetization and its uncertainty.

    """
    moment = np.asarray(moment, dtype=float)
    moment_err = (
        np.zeros_like(moment) if moment_err is None else np.asarray(moment_err, dtype=float)
    )

    if unit not in UNITS:
        raise ValueError(f"Unknown unit {unit!r}; must be one of {UNITS}")
    if unit == "emu":
        return moment, moment_err

    if not mass_mg:
        raise ValueError(f"Sample mass is required to convert to {unit}")
    mass_g = mass_mg * 1e-3
    # relative variance of the conversion factor
    factor_rel_var = ((mass_err_mg or 0) / mass_mg) ** 2

    if unit == "emu/g":
        factor = 1 / mass_g
    elif unit == "μB/f.u.":
        if not formula_weight:
            raise ValueError("Formula weight is required to convert to μB/f.u.")
        factor = formula_weight / (mass_g * N_A_MU_B_EMU)
    else:
        if not density:
            raise ValueError("Density is required to convert to T")
        factor = MU0_EMU_PER_CM3_TO_T * density / mass_g
        factor_rel_var += ((density_err or 0) / density) ** 2

    value = moment * factor
    return value, np.sqrt((moment_err * factor) ** 2 + value**2 * factor_rel_var)


def susceptibility(
    magnetization: np.ndarray,
    magnetization_err: np.ndarray,
    field_oe: np.ndarray,
    unit: str,
    formula_weight: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute χ = M/H (units in `SUSCEPTIBILITY_UNITS`) from magnetization in `unit`.
    Points at zero field are returned as NaN. Field uncertainty is neglected.
    """
    field = np.asarray(field_oe, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_field = np.where(field == 0, np.nan, 1 / field)

    if unit == "μB/f.u.":
        # back to emu/mol before dividing by H
        factor = N_A_MU_B_EMU
    elif unit == "T":
        # μ0M (T) → M (emu/cm³), then χ_SI = 4π M/H
        factor = 4 * np.pi / MU0_EMU_PER_CM3_TO_T
    else:
        factor = 1.0

    return (
        magnetization * factor * inv_field,
        np.abs(magnetization_err * factor * inv_field),
    )


def inverse(value: np.ndarray, err: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return 1/x and its propagated uncertainty."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(value == 0, np.nan, 1 / value)
    return inv, np.abs(err) * inv**2
