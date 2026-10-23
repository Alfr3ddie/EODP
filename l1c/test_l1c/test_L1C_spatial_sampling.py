"""L1B spatial sampling in ACT/ALT directions using Haversine distances.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

DATA_ROOT = Path(r"C:\Users\pleon\EODP_files\SHARED\EODP_TER_2021\EODP-TS-L1C")
GEO_FILE = DATA_ROOT / "input" / "gm_alt100_act_150" / "geolocation.nc"
OUTPUT_DIR = DATA_ROOT / "test_output_L1C_spatial_sampling"
EARTH_RADIUS_M = 6371000.0


def read_coordinates():
    if not GEO_FILE.is_file():
        raise FileNotFoundError(f"Missing geolocation input: {GEO_FILE}")
    with Dataset(GEO_FILE) as ds:
        projection = ds.groups["projection"]
        lat = np.asarray(np.ma.filled(projection.variables["latitude"][:], np.nan), dtype=float)
        lon = np.asarray(np.ma.filled(projection.variables["longitude"][:], np.nan), dtype=float)
    if lat.shape != lon.shape or lat.ndim != 2:
        raise ValueError(f"Expected matching 2D coordinates, got {lat.shape}, {lon.shape}")
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError("Geolocation coordinates contain NaN or infinity")
    return lat, lon


def haversine(lat1, lon1, lat2, lon2):
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def save_profile(distances, direction, fixed_index):
    x = np.arange(len(distances))
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(x, distances, color="tab:blue", linewidth=1.5)
    ax.set(xlabel=f"Consecutive {direction} pixel pair index",
           ylabel="Ground distance [m]",
           title=f"L1B spatial sampling - {direction} (fixed index={fixed_index})")
    ax.grid(alpha=0.3)
    ax.ticklabel_format(
        axis="y",
        style="plain",
        useOffset=False
    )
    fig.tight_layout()
    path = OUTPUT_DIR / f"l1b_spatial_sampling_{direction.lower()}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    lat, lon = read_coordinates()
    n_alt, n_act = lat.shape
    middle_alt, middle_act = n_alt // 2, n_act // 2
    # ACT: neighbouring columns along central ALT row.
    act = haversine(lat[middle_alt, :-1], lon[middle_alt, :-1],
                    lat[middle_alt, 1:], lon[middle_alt, 1:])
    # ALT: neighbouring rows along central ACT column.
    alt = haversine(lat[:-1, middle_act], lon[:-1, middle_act],
                    lat[1:, middle_act], lon[1:, middle_act])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("L1B ground sampling distances (Haversine, spherical Earth)")
    print(f"Input shape: {lat.shape}, central ALT={middle_alt}, central ACT={middle_act}")
    print("Direction  N pairs   Mean [m]   Median [m]   Min [m]   Max [m]   Std [m]")
    rows = []
    for direction, values, fixed_index in (("ACT", act, middle_alt), ("ALT", alt, middle_act)):
        stats = dict(direction=direction, n_pairs=len(values), mean_m=float(np.mean(values)),
                     median_m=float(np.median(values)), min_m=float(np.min(values)),
                     max_m=float(np.max(values)), std_m=float(np.std(values)))
        rows.append(stats)
        print(f"{direction:>8}  {len(values):>7}   {stats['mean_m']:>8.3f}   "
              f"{stats['median_m']:>10.3f}   {stats['min_m']:>7.3f}   "
              f"{stats['max_m']:>7.3f}   {stats['std_m']:>7.3f}")
        print(f"Saved: {save_profile(values, direction, fixed_index)}")
    csv_path = OUTPUT_DIR / "l1b_spatial_sampling_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {csv_path}")



if __name__ == "__main__":
    main()
