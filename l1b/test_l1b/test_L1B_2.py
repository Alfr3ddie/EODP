"""L1B restored-signal test
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

DATA_ROOT = Path(r"C:\Users\pleon\EODP_files\SHARED\EODP_TER_2021\EODP-TS-L1B")
L1B_DIR = DATA_ROOT / "output_eq"
ISRF_DIR = DATA_ROOT / "input"
REPORT_OUTPUT = DATA_ROOT / "test_output_L1B_2"
BANDS = ["VNIR-0", "VNIR-1", "VNIR-2", "VNIR-3"]


def read_toa(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing NetCDF file: {path}")
    with Dataset(path, "r") as ds:
        if "toa" not in ds.variables:
            raise KeyError(f"Variable 'toa' is missing in {path}")
        data = np.asarray(np.ma.filled(ds.variables["toa"][:], np.nan), dtype=float)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D TOA array in {path}; got {data.shape}")
    return data


def main():
    REPORT_OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    print("L1B Restored Signal")

    for ax, band in zip(axes.flat, BANDS):
        l1b = read_toa(L1B_DIR / f"l1b_toa_{band}.nc")
        isrf = read_toa(ISRF_DIR / f"ism_toa_isrf_{band}.nc")
        if l1b.shape != isrf.shape:
            raise ValueError(f"Shape mismatch for {band}: L1B {l1b.shape}, ISRF {isrf.shape}")
        row = l1b.shape[0] // 2  # Central ALT row, ACT varies horizontally.
        x = np.arange(l1b.shape[1])
        ax.plot(x, l1b[row, :], color="tab:blue", label="L1B TOA", linewidth=1.6)
        ax.plot(x, isrf[row, :], color="tab:red", label="ISRF TOA", linewidth=1.4)
        ax.set_title(band)
        ax.set_xlabel("ACT pixel index")
        ax.set_ylabel("Radiance [mW/m²/sr]")
        ax.grid(alpha=0.25)
        ax.legend()
        print(f"{band}: shape={l1b.shape}, central ALT={row} - plotted")

    fig.suptitle("Restored L1B TOA vs ISRF TOA (central ALT)")
    fig.tight_layout()
    target = REPORT_OUTPUT / "l1b_restored_signal"
    fig.savefig(target, dpi=180)
    plt.close(fig)
    print(f"Saved Figure: {target}")


if __name__ == "__main__":
    main()
