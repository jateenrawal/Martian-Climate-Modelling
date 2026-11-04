"""
Plotting utilities for the Mars EBM (mars_ebm_model.py) output CSV.

Usage:
    python plot_mars_ebm.py --csv ..\\Output\\mars_ebm_output.csv --out-dir ..\\Results
"""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# This file lives in <project_root>/Plot/. Model output lives in
# <project_root>/Output/ and plots are saved to <project_root>/Results/,
# resolved relative to this file so the script works from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "Output" / "mars_ebm_output.csv"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "Results"


def _lat_columns(df: pd.DataFrame, prefix: str):
    """Return (latitudes, column_names) for columns like 'T_-45', sorted by latitude."""
    lats = []
    for c in df.columns:
        if not c.startswith(prefix + "_"):
            continue
        try:
            lats.append(float(c[len(prefix) + 1:]))
        except ValueError:
            continue  # skip non-latitude columns, e.g. 'T_global_mean'
    lats = sorted(lats)
    ordered_cols = [f"{prefix}_{lat:.0f}" for lat in lats]
    return np.array(lats), ordered_cols


def plot_global_mean_timeseries(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    axes[0].plot(df["time_years"], df["T_global_mean"], lw=1)
    axes[0].set_ylabel("Global mean\nsurface temperature (K)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["time_years"], df["albedo_global_mean"], color="tab:orange", lw=1, label="Albedo")
    axes[1].plot(df["time_years"], df["frost_area_frac"], color="tab:cyan", lw=1, label="Frost-covered area fraction")
    axes[1].set_ylabel("Fraction")
    axes[1].set_xlabel("Time (Mars years)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Mars EBM: global-mean evolution")
    fig.tight_layout()
    fig.savefig(out_dir / "global_mean_timeseries.png", dpi=150)
    plt.close(fig)


def plot_seasonal_latitude_heatmap(df: pd.DataFrame, out_dir: Path):
    last_year = int(df["time_years"].max())
    year_df = df[(df["time_years"] >= last_year) & (df["time_years"] < last_year + 1)]
    if year_df.empty:
        year_df = df

    lats, cols = _lat_columns(df, "T")
    field = year_df[cols].to_numpy().T          # shape: (n_lat, n_time)
    ls = year_df["Ls_deg"].to_numpy()

    order = np.argsort(ls)
    ls_sorted = ls[order]
    field_sorted = field[:, order]

    fig, ax = plt.subplots(figsize=(9, 5))
    mesh = ax.pcolormesh(ls_sorted, lats, field_sorted, shading="auto", cmap="inferno")
    fig.colorbar(mesh, ax=ax, label="Surface temperature (K)")
    ax.set_xlabel("Solar longitude, Ls (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(f"Mars EBM: seasonal-latitudinal temperature (year {last_year})")
    fig.tight_layout()
    fig.savefig(out_dir / "seasonal_latitude_temperature.png", dpi=150)
    plt.close(fig)


def plot_annual_mean_profile(df: pd.DataFrame, out_dir: Path):
    last_year = int(df["time_years"].max())
    year_df = df[(df["time_years"] >= last_year) & (df["time_years"] < last_year + 1)]
    if year_df.empty:
        year_df = df

    lats, t_cols = _lat_columns(df, "T")
    _, alb_cols = _lat_columns(df, "alb")
    _, frost_cols = _lat_columns(df, "frost")

    t_mean = year_df[t_cols].mean(axis=0).to_numpy()
    alb_mean = year_df[alb_cols].mean(axis=0).to_numpy()
    frost_mean = year_df[frost_cols].mean(axis=0).to_numpy()

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(lats, t_mean, color="tab:red", marker="o", ms=3, label="Temperature")
    ax1.set_xlabel("Latitude (deg)")
    ax1.set_ylabel("Annual-mean surface temperature (K)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(lats, alb_mean, color="tab:blue", marker="s", ms=3, label="Albedo")
    ax2.plot(lats, frost_mean, color="tab:cyan", marker="^", ms=3, label="Frost fraction")
    ax2.set_ylabel("Albedo / frost fraction", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", fontsize=8)

    ax1.set_title(f"Mars EBM: annual-mean latitude profile (year {last_year})")
    fig.tight_layout()
    fig.savefig(out_dir / "annual_mean_latitude_profile.png", dpi=150)
    plt.close(fig)


def plot_polar_cap_extent(df: pd.DataFrame, out_dir: Path):
    lats, frost_cols = _lat_columns(df, "frost")
    north_cols = [c for lat, c in zip(lats, frost_cols) if lat > 0]
    south_cols = [c for lat, c in zip(lats, frost_cols) if lat < 0]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(df["time_years"], df[north_cols].mean(axis=1), label="Northern hemisphere", color="tab:blue")
    ax.plot(df["time_years"], df[south_cols].mean(axis=1), label="Southern hemisphere", color="tab:orange")
    ax.set_xlabel("Time (Mars years)")
    ax.set_ylabel("Mean frost-covered area fraction")
    ax.set_title("Mars EBM: seasonal polar cap 'breathing'")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "polar_cap_extent.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Plot Mars EBM output.")
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="model output CSV file")
    ap.add_argument("--out-dir", default=str(DEFAULT_RESULTS_DIR), help="directory to save PNG plots into")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_global_mean_timeseries(df, out_dir)
    plot_seasonal_latitude_heatmap(df, out_dir)
    plot_annual_mean_profile(df, out_dir)
    plot_polar_cap_extent(df, out_dir)

    print(f"Saved plots to {out_dir}/")


if __name__ == "__main__":
    main()
