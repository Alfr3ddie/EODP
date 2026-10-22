"""L1C georeferenced grid and TOA visualizations.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

DATA_ROOT = Path(r"C:\Users\pleon\EODP_files\SHARED\EODP_TER_2021\EODP-TS-L1C")
MY_OUTPUT = DATA_ROOT / "Myoutputs"
REPORT_OUTPUT = DATA_ROOT / "test_output_L1C_2"
BANDS = ("VNIR-0", "VNIR-1", "VNIR-2", "VNIR-3")


def read_l1c(band):
    filename = MY_OUTPUT / f"l1c_toa_{band}.nc"
    if not filename.is_file():
        raise FileNotFoundError(f"Missing L1C product: {filename}")

    with Dataset(filename, "r") as dataset:
        fields = {}
        for name in ("lat", "lon", "toa"):
            if name not in dataset.variables:
                raise KeyError(f"Variable '{name}' not found in {filename}")
            fields[name] = np.ma.filled(
                np.ma.asarray(dataset.variables[name][:], dtype=float), np.nan
            ).ravel()

    lat, lon, toa = fields["lat"], fields["lon"], fields["toa"]
    if not (len(lat) == len(lon) == len(toa)):
        raise ValueError(f"Inconsistent sizes in {filename}")
    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(toa)
    if not np.any(valid):
        raise ValueError(f"No valid points in {filename}")
    return lon[valid], lat[valid], toa[valid]


def plot_grid(data):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, band in zip(axes.flat, BANDS):
        lon, lat, _ = data[band]
        ax.scatter(lon, lat, s=3, marker=".", color="firebrick", rasterized=True)
        ax.set_title(band)
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("L1C georeferenced grid")
    filepath = REPORT_OUTPUT / "l1c_georeferenced_grid.png"
    fig.savefig(filepath, dpi=200)
    plt.close(fig)
    return filepath


def plot_toa(data):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, band in zip(axes.flat, BANDS):
        lon, lat, toa = data[band]
        im = ax.scatter(lon, lat, c=toa, s=10, marker="s",
                        cmap="jet", linewidths=0, rasterized=True)
        ax.set_title(band)
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        ax.set_aspect("equal", adjustable="datalim")
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label("TOA radiance [mW/(m² sr)]")
    fig.suptitle("L1C georeferenced TOA")
    filepath = REPORT_OUTPUT / "l1c_georeferenced_toa.png"
    fig.savefig(filepath, dpi=200)
    plt.close(fig)
    return filepath


def main():
    if not MY_OUTPUT.is_dir():
        raise FileNotFoundError(f"L1C output folder not found: {MY_OUTPUT}")
    REPORT_OUTPUT.mkdir(parents=True, exist_ok=True)

    data = {}
    print("L1C georeferenced grid and TOA")
    for band in BANDS:
        data[band] = read_l1c(band)
        print(f"{band}: {len(data[band][0])} valid georeferenced points")

    print(f"Saved: {plot_grid(data)}")
    print(f"Saved: {plot_toa(data)}")


if __name__ == "__main__":
    main()
