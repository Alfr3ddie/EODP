"""L1C grid, georeferenced TOA and L1C/L1B geographic overlay.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

DATA_ROOT = Path(r"C:\Users\pleon\EODP_files\SHARED\EODP_TER_2021\EODP-TS-L1C")
L1C_DIR = DATA_ROOT / "Myoutputs"
L1B_DIR = DATA_ROOT / "input" / "l1b_output"
GEO_FILE = DATA_ROOT / "input" / "gm_alt100_act_150" / "geolocation.nc"
OUTPUT_DIR = DATA_ROOT / "test_output_L1C_2"
BANDS = ("VNIR-0", "VNIR-1", "VNIR-2", "VNIR-3")


def read_array(dataset, name):
    if name not in dataset.variables:
        raise KeyError(f"Missing variable '{name}' in {dataset.filepath()}")
    return np.ma.filled(np.ma.asarray(dataset.variables[name][:], dtype=float), np.nan)


def read_l1c(band):
    path = L1C_DIR / f"l1c_toa_{band}.nc"
    with Dataset(path) as ds:
        lat = read_array(ds, "lat").ravel()
        lon = read_array(ds, "lon").ravel()
        toa = read_array(ds, "toa").ravel()
    if not (lat.size == lon.size == toa.size):
        raise ValueError(f"Inconsistent L1C dimensions: {path}")
    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(toa)
    return lon[valid], lat[valid], toa[valid]


def read_l1b(band):
    path = L1B_DIR / f"l1b_toa_{band}.nc"
    with Dataset(path) as ds:
        return read_array(ds, "toa")


def read_geolocation():
    with Dataset(GEO_FILE) as ds:
        group = ds.groups["projection"]
        return read_array(group, "longitude"), read_array(group, "latitude")


def save_grid(l1c_data):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, band in zip(axes.flat, BANDS):
        lon, lat, _ = l1c_data[band]
        ax.scatter(lon, lat, s=3, marker=".", color="firebrick", rasterized=True)
        ax.set(title=band, xlabel="Longitude [deg]", ylabel="Latitude [deg]")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("L1C georeferenced grid")
    path = OUTPUT_DIR / "l1c_georeferenced_grid.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_toa(l1c_data):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, band in zip(axes.flat, BANDS):
        lon, lat, toa = l1c_data[band]
        im = ax.scatter(lon, lat, c=toa, s=10, marker="s", cmap="jet",
                        linewidths=0, rasterized=True)
        ax.set(title=band, xlabel="Longitude [deg]", ylabel="Latitude [deg]")
        ax.set_aspect("equal", adjustable="datalim")
        fig.colorbar(im, ax=ax, shrink=0.85).set_label("TOA radiance [mW/(m² sr)]")
    fig.suptitle("L1C georeferenced TOA")
    path = OUTPUT_DIR / "l1c_georeferenced_toa.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_overlay(l1c_data, geo_lon, geo_lat):
    """Render L1B radiance in geographic coordinates with red L1C grid points."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for ax, band in zip(axes.flat, BANDS):
        l1b_toa = read_l1b(band)
        if l1b_toa.shape != geo_lat.shape or geo_lon.shape != geo_lat.shape:
            raise ValueError(
                f"L1B/geolocation shape mismatch for {band}: "
                f"TOA={l1b_toa.shape}, lat={geo_lat.shape}, lon={geo_lon.shape}"
            )
        # Both layers use the same geographic coordinates (not pixel indices).
        background = ax.pcolormesh(geo_lon, geo_lat, np.ma.masked_invalid(l1b_toa),
                                   shading="nearest", cmap="gray", rasterized=True)
        lon, lat, _ = l1c_data[band]
        # Small red markers: the L1C spacing is defined by geographic coordinates.
        ax.scatter(lon, lat, s=2.5, c="red", marker="o",
                   edgecolors="white", linewidths=0.12, alpha=0.9,
                   zorder=4, rasterized=True, label="L1C grid points")
        ax.set(title=band, xlabel="Longitude [deg]", ylabel="Latitude [deg]")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="upper left", fontsize=8)
        fig.colorbar(background, ax=ax, shrink=0.78).set_label(
            "L1B TOA radiance [mW/(m² sr)]")
    fig.suptitle("L1C grid points over L1B TOA")
    path = OUTPUT_DIR / "l1c_grid_overlay_l1b.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main():
    for path in (L1C_DIR, L1B_DIR, GEO_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    l1c_data = {}
    for band in BANDS:
        l1c_data[band] = read_l1c(band)
        print(f"{band}: {len(l1c_data[band][0])} L1C geographic points")
    geo_lon, geo_lat = read_geolocation()
    print(f"L1B geographic grid: {geo_lat.shape}")
    print("Saved:", save_grid(l1c_data))
    print("Saved:", save_toa(l1c_data))
    print("Saved:", save_overlay(l1c_data, geo_lon, geo_lat))


if __name__ == "__main__":
    main()
