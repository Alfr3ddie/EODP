"""L1B equalization comparison
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

DATA_ROOT = Path(r"C:\Users\pleon\EODP_files\SHARED\EODP_TER_2021\EODP-TS-L1B")
EQ_DIR = DATA_ROOT / "output_eq"
NO_EQ_DIR = DATA_ROOT / "output_not_eq"
ISRF_DIR = DATA_ROOT / "input"
REPORT_OUTPUT = DATA_ROOT / "test_output_L1B_3"
BANDS = ["VNIR-0", "VNIR-1", "VNIR-2", "VNIR-3"]


def read_toa(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    with Dataset(path, "r") as dataset:
        if "toa" not in dataset.variables:
            raise KeyError(f"Missing variable 'toa' in {path}")
        data = np.asarray(np.ma.filled(dataset.variables["toa"][:], np.nan), dtype=float)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D TOA in {path}, got {data.shape}")
    return data


def main():
    REPORT_OUTPUT.mkdir(parents=True, exist_ok=True)
    print("L1B Equalization - ACT profile comparisons")

    for band in BANDS:
        eq = read_toa(EQ_DIR / f"l1b_toa_{band}.nc")
        no_eq = read_toa(NO_EQ_DIR / f"l1b_toa_{band}.nc")
        isrf = read_toa(ISRF_DIR / f"ism_toa_isrf_{band}.nc")
        if eq.shape != no_eq.shape or eq.shape != isrf.shape:
            raise ValueError(f"Shape mismatch in {band}: eq={eq.shape}, no_eq={no_eq.shape}, ISRF={isrf.shape}")
        alt = eq.shape[0] // 2
        act = np.arange(eq.shape[1])

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
        for ax, restored, title, label in [
            (axes[0], eq, f"(a) Equalized TOA {band}", "L1B TOA"),
            (axes[1], no_eq, f"(b) Non-equalized TOA {band}", "No eq. L1B TOA"),
        ]:
            ax.plot(act, restored[alt, :], color="tab:blue", linewidth=1.4, label=label)
            ax.plot(act, isrf[alt, :], color="tab:red", linewidth=1.4, label="ISRF TOA")
            ax.set_title(title)
            ax.set_xlabel("ACT pixel index")
            ax.set_ylabel("Radiance [mW/m²/sr]")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9)

        fig.tight_layout()
        filename = REPORT_OUTPUT / f"l1b_equalization_{band}.png"
        fig.savefig(filename, dpi=180)
        plt.close(fig)
        print(f"{band}: shape={eq.shape}, central ALT={alt}; saved {filename.name}")

    print(f"Saved four equalization plots in: {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
