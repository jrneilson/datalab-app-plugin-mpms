"""A datalab block for Quantum Design MPMS / MPMS3 magnetometry data."""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pydatalab.blocks.base import DataBlock, event

from datalab_app_plugin_mpms._version import __version__
from datalab_app_plugin_mpms.utils import (
    FIELD_CANDIDATES,
    INVERSE_SUSCEPTIBILITY_UNITS,
    MOMENT_CANDIDATES,
    SUSCEPTIBILITY_UNITS,
    TEMPERATURE_CANDIDATES,
    TIME_CANDIDATES,
    UNITS,
    convert_magnetization,
    detect_column,
    detect_error_column,
    inverse,
    numeric_columns,
    parse_qd_dat,
    susceptibility,
)

TEMPERATURE = "Temperature (K)"
FIELD = "Field (Oe)"
TIME = "Time (s)"
X_AXES = (TEMPERATURE, FIELD, TIME)
QUANTITIES = ("M", "χ", "1/χ")
NO_ERROR = "None"
WIDGET_WIDTH = 240

FILTER_KEYS = {
    TEMPERATURE: "filter_temperature",
    FIELD: "filter_field",
    TIME: "filter_time",
}

NUMERIC_SETTINGS = (
    "mass_mg",
    "mass_err_mg",
    "formula_weight",
    "density",
    "density_err",
    *(f"{key}_{bound}" for key in FILTER_KEYS.values() for bound in ("min", "max")),
)
POSITIVE_SETTINGS = ("mass_mg", "formula_weight", "density")
CHOICE_SETTINGS = {
    "x_axis": X_AXES,
    "quantity": QUANTITIES,
    "unit": UNITS,
    # column choices are validated against the file at plot time
    "moment_column": None,
    "error_column": None,
    "temperature_column": None,
    "field_column": None,
}

EXPORT_CSV_CALLBACK = """
const columns = Object.keys(source.data);
const rows = [columns.map((c) => '"' + c + '"').join(",")];
for (let i = 0; i < source.data[columns[0]].length; i++) {
    rows.push(columns.map((c) => {
        const v = source.data[c][i];
        return (v === null || Number.isNaN(v)) ? "" : v;
    }).join(","));
}
const blob = new Blob([rows.join("\\n") + "\\n"], {type: "text/csv;charset=utf-8"});
const link = document.createElement("a");
link.href = URL.createObjectURL(blob);
link.download = filename;
document.body.appendChild(link);
link.click();
document.body.removeChild(link);
"""

AXIS_MENU_CALLBACK = """
// Open a menu of plot choices when the x or y axis (outside the plot frame) is clicked.
const [x0, x1] = [Math.min(xr.start, xr.end), Math.max(xr.start, xr.end)];
const [y0, y1] = [Math.min(yr.start, yr.end), Math.max(yr.start, yr.end)];
const in_x = cb_obj.x >= x0 && cb_obj.x <= x1;
const in_y = cb_obj.y >= y0 && cb_obj.y <= y1;
let options = null;
if (in_x && cb_obj.y < y0) options = menus.x;
else if (in_y && cb_obj.x < x0) options = menus.y;
if (options === null) return;

document.getElementById("mpms-axis-menu")?.remove();
// tap coordinates are relative to the plot canvas
const canvas = document.querySelector(".mpms-" + block_id + " .bk-canvas-events");
const rect = canvas ? canvas.getBoundingClientRect() : {left: 0, top: 0};
const menu = document.createElement("div");
menu.id = "mpms-axis-menu";
Object.assign(menu.style, {
    position: "absolute", zIndex: 10000, background: "#fff", border: "1px solid #ccc",
    borderRadius: "4px", boxShadow: "0 2px 8px rgba(0,0,0,0.15)", padding: "4px 0",
    fontSize: "14px", whiteSpace: "nowrap",
    left: (rect.left + window.scrollX + cb_obj.sx) + "px",
    top: (rect.top + window.scrollY + cb_obj.sy) + "px",
});
for (const option of options) {
    const item = document.createElement("div");
    item.textContent = (option.current ? "✓ " : "\u2003") + option.label;
    Object.assign(item.style, {padding: "4px 12px", cursor: "pointer"});
    item.onmouseenter = () => { item.style.background = "#e9ecef"; };
    item.onmouseleave = () => { item.style.background = ""; };
    item.onclick = (e) => {
        e.stopPropagation();
        menu.remove();
        const detail = Object.assign({block_id: block_id, event_name: "update_settings"}, option.settings);
        document.dispatchEvent(new CustomEvent("block-event", {detail: detail, bubbles: true}));
    };
    menu.appendChild(item);
}
document.body.appendChild(menu);
setTimeout(() => document.addEventListener("pointerdown", function close(e) {
    if (!menu.contains(e.target)) {
        menu.remove();
        document.removeEventListener("pointerdown", close);
    }
}), 0);
"""

# datalab restyles Bokeh widgets with Bootstrap classes after rendering, which changes their
# size after Bokeh has laid them out; restore Bokeh's geometry within this block.
WIDGET_STYLE = """<style>
.mpms-block .input-group.form-inline.col-sm-6 {
    flex: none; max-width: none; width: 100%; height: 100%; padding: 0;
    display: inline-flex; flex-direction: column; flex-wrap: nowrap;
    align-items: stretch; white-space: nowrap;
}
.mpms-block .input-group > label { justify-content: flex-start; }
.mpms-block .input-group > select.form-control {
    margin-left: 0 !important; width: 100%; height: auto; min-height: 31px;
    padding: 0 12px; font-size: inherit;
}
</style>"""


def _y_columns(quantity: str, unit: str) -> tuple[str, str]:
    """The output column names for the plotted quantity and its uncertainty."""
    if quantity == "M":
        return f"M ({unit})", f"σM ({unit})"
    if quantity == "χ":
        return f"χ ({SUSCEPTIBILITY_UNITS[unit]})", f"σχ ({SUSCEPTIBILITY_UNITS[unit]})"
    return (
        f"1/χ ({INVERSE_SUSCEPTIBILITY_UNITS[unit]})",
        f"σ(1/χ) ({INVERSE_SUSCEPTIBILITY_UNITS[unit]})",
    )


def _js_update_setting(key: str, block_id: str) -> str:
    """A Bokeh JS callback that sends a widget's value to the block's `update_settings` event."""
    detail = {"block_id": block_id, "event_name": "update_settings"}
    return (
        f"const detail = {json.dumps(detail)};\n"
        f"detail[{json.dumps(key)}] = (cb_obj.value ?? cb_obj.text);\n"
        "document.dispatchEvent(new CustomEvent('block-event', {detail: detail, bubbles: true}));"
    )


CLEARED = ""
"""Stored for a cleared numeric setting. datalab drops `None` values when it sends the block
to the browser, which then keeps (and later sends back) the old value; "" replaces it."""


def _parse_float(key: str, value) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid value for {key}: {value!r}. Must be a number.")
    if key in POSITIVE_SETTINGS and value <= 0:
        raise ValueError(f"{key} must be positive")
    if (key.endswith("_err") or key.endswith("_err_mg")) and value < 0:
        raise ValueError(f"{key} must not be negative")
    return value


class MPMSBlock(DataBlock):
    version = __version__
    blocktype: str = "mpms"
    name: str = "MPMS magnetometry"
    description: str = (
        "Plot Quantum Design MPMS / MPMS3 magnetization or (inverse) susceptibility in emu, "
        "emu/g, μB/f.u. or T with propagated uncertainties, filter by temperature, field or "
        "time, and export the converted data."
    )
    accepted_file_extensions = (".dat",)

    defaults = {"x_axis": TEMPERATURE, "quantity": "M", "unit": "emu"}

    @property
    def plot_functions(self):
        return (self.generate_plot,)

    def _number(self, key: str) -> float | None:
        """A numeric setting, or None if it is unset or cleared."""
        value = self.data.get(key)
        return None if value in (None, CLEARED) else float(value)

    @event()
    def update_settings(self, **kwargs):
        """Update one or more block settings from the plot widgets."""
        for key, value in kwargs.items():
            if key in NUMERIC_SETTINGS:
                parsed = _parse_float(key, value)
                self.data[key] = CLEARED if parsed is None else parsed
            elif key in CHOICE_SETTINGS:
                allowed = CHOICE_SETTINGS[key]
                if allowed is not None and value not in allowed:
                    raise ValueError(f"Invalid value for {key}: {value!r}")
                self.data[key] = value
            else:
                raise ValueError(f"Unknown setting {key!r}")

    def _prefill_from_header(self, header: dict[str, str]):
        """Fill sample mass and formula weight from the file header the first time it is read."""
        for key, header_key in (
            ("mass_mg", "SAMPLE_MASS"),
            ("formula_weight", "SAMPLE_MOLECULAR_WEIGHT"),
        ):
            if key not in self.data:
                try:
                    parsed = _parse_float(key, header.get(header_key))
                    self.data[key] = CLEARED if parsed is None else parsed
                except ValueError:
                    self.data[key] = CLEARED

    def _resolve_columns(self, df: pd.DataFrame):
        """Choose the moment, error, temperature and field columns, keeping valid user choices."""
        available = numeric_columns(df)
        for key, candidates in (
            ("moment_column", MOMENT_CANDIDATES),
            ("temperature_column", TEMPERATURE_CANDIDATES),
            ("field_column", FIELD_CANDIDATES),
        ):
            if self.data.get(key) not in available:
                self.data[key] = detect_column(df, candidates)
                if self.data[key] is None:
                    raise ValueError(f"Could not find a {key.replace('_', ' ')} in the file")
        if self.data.get("error_column") not in [*available, NO_ERROR]:
            self.data["error_column"] = (
                detect_error_column(df, self.data["moment_column"]) or NO_ERROR
            )

    def compute(self, df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        """Convert and filter the raw data according to the block settings.

        Returns:
            A DataFrame with the x variables, raw moment and every magnetization and
            susceptibility unit that can be computed from the given sample parameters,
            and the unit actually used for plotting.

        """
        self._resolve_columns(df)

        out = pd.DataFrame()
        time_column = detect_column(df, TIME_CANDIDATES)
        out[TIME] = df[time_column] - df[time_column].min() if time_column else np.nan
        out[TEMPERATURE] = df[self.data["temperature_column"]]
        out[FIELD] = df[self.data["field_column"]]
        moment = df[self.data["moment_column"]].to_numpy(dtype=float)
        error_column = self.data["error_column"]
        moment_err = None if error_column == NO_ERROR else df[error_column].to_numpy(dtype=float)

        sample = {
            "mass_mg": self._number("mass_mg"),
            "mass_err_mg": self._number("mass_err_mg"),
            "formula_weight": self._number("formula_weight"),
            "density": self._number("density"),
            "density_err": self._number("density_err"),
        }

        available_units = []
        for unit in UNITS:
            try:
                m, m_err = convert_magnetization(moment, moment_err, unit, **sample)
            except ValueError:
                continue
            available_units.append(unit)
            chi_unit = SUSCEPTIBILITY_UNITS[unit]
            chi, chi_err = susceptibility(
                m, m_err, out[FIELD].to_numpy(), unit, sample["formula_weight"]
            )
            inv_chi, inv_chi_err = inverse(chi, chi_err)
            out[f"M ({unit})"] = m
            out[f"σM ({unit})"] = m_err
            out[f"χ ({chi_unit})"] = chi
            out[f"σχ ({chi_unit})"] = chi_err
            out[f"1/χ ({INVERSE_SUSCEPTIBILITY_UNITS[unit]})"] = inv_chi
            out[f"σ(1/χ) ({INVERSE_SUSCEPTIBILITY_UNITS[unit]})"] = inv_chi_err

        unit = self.data.get("unit", "emu")
        if unit not in available_units:
            missing = {
                "emu/g": "mass",
                "μB/f.u.": "mass and formula weight",
                "T": "mass and density",
            }
            warnings.warn(
                f"Enter the sample {missing[unit]} to plot in {unit}; showing emu instead."
            )
            unit = "emu"

        mask = np.ones(len(out), dtype=bool)
        for variable, key in FILTER_KEYS.items():
            if (lower := self._number(f"{key}_min")) is not None:
                mask &= out[variable].to_numpy() >= lower
            if (upper := self._number(f"{key}_max")) is not None:
                mask &= out[variable].to_numpy() <= upper

        return out[mask].reset_index(drop=True), unit

    def generate_plot(self, location: str | Path | None = None):
        """Read the attached file (or `location`, if given) and store the Bokeh plot in the block."""
        import bokeh.embed
        from pydatalab.bokeh_plots import DATALAB_BOKEH_THEME

        if location is None:
            if "file_id" not in self.data:
                return
            from pydatalab.file_utils import get_file_info_by_id

            location = get_file_info_by_id(self.data["file_id"], update_if_live=True)["location"]

        df, header = parse_qd_dat(location)
        self._prefill_from_header(header)
        out, unit = self.compute(df)

        layout = self._make_layout(df, out, unit, filename=f"{Path(location).stem}_converted.csv")
        self.data["bokeh_plot_data"] = bokeh.embed.json_item(layout, theme=DATALAB_BOKEH_THEME)

    def _make_layout(self, df: pd.DataFrame, out: pd.DataFrame, unit: str, filename: str):
        from bokeh.layouts import column, row
        from bokeh.models import (
            Button,
            ColumnDataSource,
            CustomJS,
            Div,
            HoverTool,
            Select,
            TextInput,
            Whisker,
        )
        from bokeh.plotting import figure

        def text_input(key: str, title: str) -> TextInput:
            value = self._number(key)
            widget = TextInput(
                title=title, value="" if value is None else f"{value:g}", width=WIDGET_WIDTH
            )
            widget.js_on_change("value", CustomJS(code=_js_update_setting(key, self.block_id)))
            return widget

        def select(key: str, title: str, options) -> Select:
            widget = Select(
                title=title, value=self.data[key], options=list(options), width=WIDGET_WIDTH
            )
            widget.js_on_change("value", CustomJS(code=_js_update_setting(key, self.block_id)))
            return widget

        quantity = self.data["quantity"]
        x_axis = self.data["x_axis"]
        y_label, y_err_label = _y_columns(quantity, unit)

        y = out[y_label].to_numpy()
        y_err = out[y_err_label].to_numpy()
        source = ColumnDataSource(
            {
                "x": out[x_axis].to_numpy(),
                "time": out[TIME].to_numpy(),
                "temperature": out[TEMPERATURE].to_numpy(),
                "field": out[FIELD].to_numpy(),
                "y": y,
                "upper": y + y_err,
                "lower": y - y_err,
                "err": y_err,
            }
        )

        p = figure(
            x_axis_label=f"{x_axis} ▾",
            y_axis_label=f"{y_label} ▾",
            sizing_mode="scale_width",
            aspect_ratio=1.5,
        )
        p.circle("x", "y", source=source, size=4)
        p.add_layout(Whisker(source=source, base="x", upper="upper", lower="lower", line_alpha=0.5))
        p.add_tools(
            HoverTool(
                tooltips=[
                    (TIME, "@time{0.0}"),
                    (TEMPERATURE, "@temperature{0.000}"),
                    (FIELD, "@field{0.00}"),
                    (y_label, "@y{0.0000e} ± @err{0.0e}"),
                ]
            )
        )

        menus = {
            "x": [{"label": x, "settings": {"x_axis": x}, "current": x == x_axis} for x in X_AXES],
            "y": [
                {
                    "label": _y_columns(q, u)[0],
                    "settings": {"quantity": q, "unit": u},
                    "current": (q, u) == (quantity, unit),
                }
                for q in QUANTITIES
                for u in UNITS
                if f"M ({u})" in out
            ],
        }
        p.js_on_event(
            "tap",
            CustomJS(
                args=dict(xr=p.x_range, yr=p.y_range, menus=menus, block_id=self.block_id),
                code=AXIS_MENU_CALLBACK,
            ),
        )

        columns = numeric_columns(df)
        export = Button(
            label="Export converted data (.csv)", button_type="primary", width_policy="min"
        )
        export.js_on_click(
            CustomJS(
                args=dict(source=ColumnDataSource(out), filename=filename), code=EXPORT_CSV_CALLBACK
            )
        )

        return column(
            Div(text=WIDGET_STYLE),
            Div(text="<b>Sample</b> (density is used only for T)"),
            row(text_input("mass_mg", "Mass (mg)"), text_input("mass_err_mg", "± mass (mg)")),
            row(text_input("formula_weight", "Formula weight (g/mol)")),
            row(
                text_input("density", "Density (g/cm³)"),
                text_input("density_err", "± density (g/cm³)"),
            ),
            Div(text="<b>Data columns</b>"),
            row(
                select("moment_column", "Moment column (emu)", columns),
                select("error_column", "Moment error column (emu)", [NO_ERROR, *columns]),
            ),
            row(
                select("temperature_column", "Temperature column", columns),
                select("field_column", "Field column", columns),
            ),
            Div(text="<b>Filter</b> (leave blank for no limit)"),
            *(
                row(
                    text_input(f"{key}_min", f"{variable} ≥"),
                    text_input(f"{key}_max", f"{variable} ≤"),
                )
                for variable, key in FILTER_KEYS.items()
            ),
            export,
            Div(text="Click the x or y axis to change what is plotted."),
            p,
            sizing_mode="scale_width",
            css_classes=["mpms-block", f"mpms-{self.block_id}"],
        )
