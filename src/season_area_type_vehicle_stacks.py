"""
Generate the Season x Area Type (per 100k) plot with vehicle-type stacks.
This is the only plot kept in the project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tueplots import bundles
from matplotlib.patches import Patch
from pathlib import Path
import geopandas as gpd
import zipfile
import xml.etree.ElementTree as ET
import os
import sys

# Plot style (tueplots base)
plt.rcParams.update(bundles.icml2024(column="half"))
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["text.usetex"] = False
font_scale = 1.4
plt.rcParams["font.size"] = int(16 * font_scale)
plt.rcParams["axes.labelsize"] = int(18 * font_scale)
plt.rcParams["axes.titlesize"] = int(18 * font_scale)
plt.rcParams["xtick.labelsize"] = int(15 * font_scale)
plt.rcParams["ytick.labelsize"] = int(15 * font_scale)
plt.rcParams["legend.fontsize"] = int(12 * font_scale)
plt.rcParams["legend.title_fontsize"] = int(12 * font_scale)

# Use the region-based normalization method from src/population_normalization.py
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.append(str(_SRC_DIR))
from population_normalization import AccidentNormalizer

class SevereAccidentAnalyzer:
    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        self.uk_data = None
        self.ger_data = None

    def load_uk_data(self):
        df = pd.read_csv(
            self.data_path
            / "uk"
            / "collisions"
            / "intermediate_steps"
            / "all_data_2024_uk.csv"
            ,
            low_memory=False
        )
        df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
        self.uk_data = df[df["collision_severity"] == 1].copy()
        for col in ["longitude", "latitude"]:
            self.uk_data[col] = (
                self.uk_data[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .astype(float)
            )
        self.uk_data["month"] = self.uk_data["date"].dt.month
        self.uk_data["season"] = self.uk_data["month"].apply(self._get_season)
        self.uk_data["area_type"] = self._map_area_type(self.uk_data, "urban_or_rural_area", "UK")
        return self.uk_data

    def load_germany_data(self):
        df = pd.read_csv(
            self.data_path
            / "germany"
            / "collisions"
            / "preprocessed_ger.csv",
            low_memory=False
        )
        self.ger_data = df[df["casualty_severity"] == 1].copy()
        for col in ["longitude", "latitude"]:
            self.ger_data[col] = (
                self.ger_data[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .astype(float)
            )
        self.ger_data["season"] = self.ger_data["month"].apply(self._get_season)
        self.ger_data["area_type"] = self._map_area_type(self.ger_data, "urban_or_rural_area", "Germany")
        return self.ger_data

    @staticmethod
    def _get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        if month in [3, 4, 5]:
            return "Spring"
        if month in [6, 7, 8]:
            return "Summer"
        if month in [9, 10, 11]:
            return "Autumn"
        return None

    @staticmethod
    def _map_area_type(df: pd.DataFrame, col: str, label: str):
        if col in df.columns:
            return df[col].replace({1: "Urban", 2: "Rural", "1": "Urban", "2": "Rural"})
        print(
            f"[WARN] {label} data missing '{col}'. "
            "Defaulting all rows to 'Urban'. "
            f"Available columns: {sorted(df.columns)}"
        )
        return "Urban"

    @staticmethod
    def _col_letters_to_index(col: str) -> int:
        idx = 0
        for ch in col:
            idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
        return idx - 1

    @staticmethod
    def _to_number(val):
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            v = val.replace(",", "").strip()
            try:
                return float(v)
            except Exception:
                return None
        return None

    @classmethod
    def _extract_total_from_xlsx(cls, path: Path, min_val: float, max_val: float):
        """Scan all sheets and return the max numeric value in [min_val, max_val]."""
        if not path.exists():
            return None
        try:
            with zipfile.ZipFile(path) as zf:
                shared_strings = []
                if "xl/sharedStrings.xml" in zf.namelist():
                    sst = ET.fromstring(zf.read("xl/sharedStrings.xml"))
                    for si in sst.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si"):
                        text_elems = si.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
                        shared_strings.append("".join(t.text or "" for t in text_elems))

                wb = ET.fromstring(zf.read("xl/workbook.xml"))
                sheets = []
                for sheet in wb.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheet"):
                    sheets.append((sheet.get("name"), sheet.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")))

                rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
                rel_map = {}
                for rel in rels.findall(".//{http://schemas.openxmlformats.org/package/2006/relationships}Relationship"):
                    rel_map[rel.get("Id")] = rel.get("Target")

                candidates = []
                for _, rel_id in sheets:
                    target = rel_map.get(rel_id, "")
                    sheet_path = "xl/" + target.lstrip("/")
                    if sheet_path not in zf.namelist():
                        continue
                    sheet_xml = ET.fromstring(zf.read(sheet_path))
                    for c in sheet_xml.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c"):
                        c_type = c.get("t")
                        v = c.find(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
                        if v is None:
                            continue
                        value = v.text or ""
                        if c_type == "s":
                            try:
                                value = shared_strings[int(value)]
                            except Exception:
                                pass
                        num = cls._to_number(value)
                        if num is None:
                            continue
                        if min_val <= num <= max_val:
                            candidates.append(num)
                if not candidates:
                    return None
                return max(candidates)
        except Exception:
            return None

    @staticmethod
    def _read_xlsx_sheet_rows(path: Path, sheet_name: str):
        """Read an .xlsx sheet without openpyxl and return a list of row lists."""
        if not path.exists():
            return []
        with zipfile.ZipFile(path) as zf:
            shared_strings = []
            if "xl/sharedStrings.xml" in zf.namelist():
                sst = ET.fromstring(zf.read("xl/sharedStrings.xml"))
                for si in sst.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si"):
                    text_elems = si.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
                    shared_strings.append("".join(t.text or "" for t in text_elems))

            wb = ET.fromstring(zf.read("xl/workbook.xml"))
            sheets = {}
            for sheet in wb.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheet"):
                name = sheet.get("name")
                rel_id = sheet.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
                sheets[name] = rel_id

            rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
            rel_map = {}
            for rel in rels.findall(".//{http://schemas.openxmlformats.org/package/2006/relationships}Relationship"):
                rel_map[rel.get("Id")] = rel.get("Target")

            rel_id = sheets.get(sheet_name)
            if not rel_id:
                return []
            target = rel_map.get(rel_id, "")
            sheet_path = "xl/" + target.lstrip("/")
            if sheet_path not in zf.namelist():
                return []

            sheet_xml = ET.fromstring(zf.read(sheet_path))
            rows = {}
            max_col = 0
            for row in sheet_xml.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row"):
                r_idx = int(row.get("r", "0")) - 1
                row_cells = {}
                for c in row.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c"):
                    ref = c.get("r", "")
                    col_letters = "".join(ch for ch in ref if ch.isalpha())
                    if not col_letters:
                        continue
                    col_idx = SevereAccidentAnalyzer._col_letters_to_index(col_letters)
                    c_type = c.get("t")
                    v = c.find(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
                    if v is None:
                        continue
                    value = v.text or ""
                    if c_type == "s":
                        try:
                            value = shared_strings[int(value)]
                        except Exception:
                            pass
                    row_cells[col_idx] = value
                    max_col = max(max_col, col_idx)
                if row_cells:
                    rows[r_idx] = row_cells

            result = []
            for r_idx in sorted(rows.keys()):
                row_vals = [""] * (max_col + 1)
                for c_idx, val in rows[r_idx].items():
                    row_vals[c_idx] = val
                result.append(row_vals)
            return result

    def _uk_population_total_from_mye3(self):
        """Sum UK local authority populations from MYE3 using ONS district codes."""
        uk_pop_path = self.data_path.parent / "raw" / "uk" / "population" / "mye24tablesuk.xlsx"
        if not uk_pop_path.exists():
            return None
        raw_rows = self._read_xlsx_sheet_rows(uk_pop_path, "MYE3")
        if not raw_rows:
            return None

        header_row = None
        for i, row in enumerate(raw_rows):
            row_vals = [str(v).strip() for v in row]
            if any(v.lower() == "code" for v in row_vals) and any(
                "estimated population" in v.lower() for v in row_vals
            ):
                header_row = i
                break
        if header_row is None:
            return None

        header = [str(v).strip() for v in raw_rows[header_row]]
        data_rows = raw_rows[header_row + 1 :]
        df = pd.DataFrame(data_rows, columns=header)

        code_col = None
        pop_col = None
        for col in df.columns:
            if str(col).strip().lower() == "code":
                code_col = col
            if "estimated population" in str(col).lower():
                pop_col = col
        if code_col is None or pop_col is None:
            return None

        uk_codes = self.uk_data["local_authority_ons_district"].dropna().unique()
        pop_series = pd.to_numeric(
            df.loc[df[code_col].isin(uk_codes), pop_col], errors="coerce"
        ).dropna()
        if pop_series.empty:
            return None
        return float(pop_series.sum())

    def plot_season_area_type_combined_normalized(self, output_dir: Path = None):
        if output_dir is None:
            base_dir = Path(self.data_path).parent
            graphics_dir = base_dir / "graphics"
            output_dir = graphics_dir if graphics_dir.exists() else (base_dir / "outputs")
        output_dir.mkdir(exist_ok=True)

        season_order = ["Winter", "Spring", "Summer", "Autumn"]
        x = np.arange(len(season_order))
        width = 0.4

        uk_regions_path = self.data_path / "uk" / "geofiles" / "UK_merged.geojson"
        germany_regions_path = self.data_path / "germany" / "geofiles" / "Germany_merged.geojson"

        vehicle_type_cols = {
            "Car": "is_car",
            "Bicycle": "is_cyle",
            "Motorcycle": "is_mcyle",
            "Pedestrian": "is_pedestrian",
            "Goods": "is_goodsv",
            "Other": "is_other",
        }
        vehicle_type_colors = {
            "Car": "#2A9D8F",
            "Bicycle": "#E9C46A",
            "Motorcycle": "#E76F51",
            "Pedestrian": "#8AB17D",
            "Goods": "#7B61FF",
            "Other": "#B5838D",
        }
        default_vehicle_props = {k: 1 / len(vehicle_type_cols) for k in vehicle_type_cols}

        def _vehicle_type_props(df):
            available_cols = [c for c in vehicle_type_cols.values() if c in df.columns]
            if not available_cols:
                return {}
            grouped = df.groupby(["season", "area_type"])[available_cols].sum()
            props = {}
            for idx, row in grouped.iterrows():
                total = row.sum()
                if total == 0:
                    props[idx] = default_vehicle_props
                else:
                    props[idx] = {
                        label: float(row[col]) / total
                        for label, col in vehicle_type_cols.items()
                        if col in grouped.columns
                    }
            return props

        def _draw_vehicle_stack(ax, x_pos, total_height, props, edgecolor):
            bottom = 0.0
            for label in vehicle_type_cols.keys():
                height = total_height * props.get(label, 0.0)
                ax.bar(
                    x_pos,
                    height,
                    width,
                    bottom=bottom,
                    color=vehicle_type_colors[label],
                    edgecolor="none",
                    linewidth=0,
                )
                bottom += height

        def _rates_by_season_area(df, regions, id_col):
            rates = {}
            for (season, area_type), sub in df.groupby(["season", "area_type"]):
                # Use the same normalization pipeline as AccidentNormalizer
                normalizer = AccidentNormalizer(regions.copy())
                gdf = gpd.GeoDataFrame(
                    sub.copy(),
                    geometry=gpd.points_from_xy(sub["longitude"], sub["latitude"]),
                    crs="EPSG:4326",
                ).to_crs(regions.crs)
                normalizer.attach_regions_and_count(gdf)
                norm_regions = normalizer.normalize_by_population(scale=100_000)

                # Population-weighted mean of region rates equals national rate
                pop = norm_regions["population"]
                rate_col = "accidents_per_100k"
                if pop.sum() == 0:
                    rates[(season, area_type)] = 0.0
                else:
                    rates[(season, area_type)] = (
                        (norm_regions[rate_col] * pop).sum() / pop.sum()
                    )
            return pd.Series(rates)

        def _uk_region_rates():
            os.environ.setdefault("OGR_GEOJSON_MAX_OBJ_SIZE", "0")
            regions = gpd.read_file(uk_regions_path)[["region_code", "population", "geometry"]]
            return _rates_by_season_area(self.uk_data, regions, "collision_index")

        def _ger_region_rates():
            os.environ.setdefault("OGR_GEOJSON_MAX_OBJ_SIZE", "0")
            regions = gpd.read_file(germany_regions_path)[["region_code", "population", "geometry"]]
            return _rates_by_season_area(self.ger_data, regions, "c_id")

        uk_mean_rates = _uk_region_rates()
        ger_mean_rates = _ger_region_rates()

        uk_urban = [uk_mean_rates.get((s, "Urban"), 0.0) for s in season_order]
        uk_rural = [uk_mean_rates.get((s, "Rural"), 0.0) for s in season_order]
        ger_urban = [ger_mean_rates.get((s, "Urban"), 0.0) for s in season_order]
        ger_rural = [ger_mean_rates.get((s, "Rural"), 0.0) for s in season_order]

        def _first_existing(*paths):
            for p in paths:
                if p.exists():
                    return p
            return None

        uk_types_path = _first_existing(
            self.data_path
            / "uk"
            / "collisions"
            / "intermediate_steps"
            / "merged_uk_with_kind_and_type.csv",
            self.data_path
            / "uk"
            / "collisions"
            / "intermediate_steps"
            / "merged_uk.csv",
        )
        if uk_types_path is None:
            uk_types_df = self.uk_data.copy()
        else:
            uk_types_df = pd.read_csv(uk_types_path, low_memory=False)
        uk_types_df = uk_types_df[uk_types_df["collision_severity"] == 1].copy()
        uk_types_df["season"] = uk_types_df["month"].apply(self._get_season)
        uk_types_df["area_type"] = self._map_area_type(uk_types_df, "urban_or_rural_area", "UK vehicle types")

        uk_vehicle_props = _vehicle_type_props(uk_types_df)
        ger_vehicle_props = _vehicle_type_props(self.ger_data)

        uk_edge = "#4169E1"
        ger_edge = "#FF8C00"
        urban_alpha = 0.55
        rural_alpha = 0.95

        gap = 1.0
        x_uk = x
        x_ger = x + len(season_order) + gap

        fig, ax = plt.subplots(figsize=(14, 8))
        fig.subplots_adjust(top=0.86)
        for i, season in enumerate(season_order):
            _draw_vehicle_stack(
                ax,
                x_uk[i] - (width / 2),
                uk_urban[i],
                uk_vehicle_props.get((season, "Urban"), default_vehicle_props),
                edgecolor=uk_edge,
            )
            for bar in ax.patches[-len(vehicle_type_cols) :]:
                bar.set_alpha(urban_alpha)
            _draw_vehicle_stack(
                ax,
                x_uk[i] + (width / 2),
                uk_rural[i],
                uk_vehicle_props.get((season, "Rural"), default_vehicle_props),
                edgecolor=uk_edge,
            )
            for bar in ax.patches[-len(vehicle_type_cols) :]:
                bar.set_alpha(rural_alpha)

            _draw_vehicle_stack(
                ax,
                x_ger[i] - (width / 2),
                ger_urban[i],
                ger_vehicle_props.get((season, "Urban"), default_vehicle_props),
                edgecolor=ger_edge,
            )
            for bar in ax.patches[-len(vehicle_type_cols) :]:
                bar.set_alpha(urban_alpha)
            _draw_vehicle_stack(
                ax,
                x_ger[i] + (width / 2),
                ger_rural[i],
                ger_vehicle_props.get((season, "Rural"), default_vehicle_props),
                edgecolor=ger_edge,
            )
            for bar in ax.patches[-len(vehicle_type_cols) :]:
                bar.set_alpha(rural_alpha)

        ax.set_xlabel("Season", fontsize=int(16 * font_scale), fontweight="bold")
        ax.set_ylabel("Accidents per 100,000", fontsize=int(16 * font_scale), fontweight="bold")
        # Intentionally no title per request
        ax.set_xticks(list(x_uk) + list(x_ger))
        ax.set_xticklabels(season_order + season_order, fontsize=int(14 * font_scale))
        ax.tick_params(axis="y", labelsize=int(13 * font_scale))
        ax.axvline((x_uk[-1] + x_ger[0]) / 2, color="#cccccc", linewidth=1, alpha=0.8)
        ax.text(x_uk.mean(), ax.get_ylim()[1] * 1.03, "UK", ha="center", fontsize=int(16 * font_scale), fontweight="bold")
        ax.text(
            x_ger.mean(),
            ax.get_ylim()[1] * 1.03,
            "Germany",
            ha="center",
            fontsize=int(16 * font_scale),
            fontweight="bold",
        )

        # Compact custom legend: Urban swatch | label | Rural swatch (centered, not side-specific)
        from matplotlib.patches import Rectangle

        legend_w = 0.20
        legend_h = 0.28
        legend_ax = ax.inset_axes([0.5 - legend_w / 2, 0.98 - legend_h, legend_w, legend_h])
        legend_ax.axis("off")
        legend_ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#cccccc", linewidth=1))
        urban_x = 0.18
        rural_x = 0.82
        legend_ax.text(urban_x, 0.92, "Urban", ha="center", va="center", fontsize=int(11 * font_scale), fontweight="bold")
        legend_ax.text(rural_x, 0.92, "Rural", ha="center", va="center", fontsize=int(11 * font_scale), fontweight="bold")

        labels = list(vehicle_type_cols.keys())
        row_y = np.linspace(0.76, 0.06, num=len(labels))
        for y, label in zip(row_y, labels):
            legend_ax.add_patch(
                Rectangle((urban_x - 0.05, y - 0.025), 0.10, 0.06, facecolor=vehicle_type_colors[label], alpha=urban_alpha, edgecolor="none")
            )
            legend_ax.text(0.50, y, label, ha="center", va="center", fontsize=max(12, int(10 * font_scale)))
            legend_ax.add_patch(
                Rectangle((rural_x - 0.05, y - 0.025), 0.10, 0.06, facecolor=vehicle_type_colors[label], edgecolor="none")
            )
        ax.grid(axis="y", alpha=0.3)

        output_path = output_dir / "season_area_type_combined_normalized.pdf"
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_dir = BASE_DIR / "data" / "preprocessed"
    analyzer = SevereAccidentAnalyzer(data_dir)
    analyzer.load_uk_data()
    analyzer.load_germany_data()
    analyzer.plot_season_area_type_combined_normalized(BASE_DIR / "outputs")
