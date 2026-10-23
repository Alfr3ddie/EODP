"""L1B verification and radiometric calibration profiles.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

DATA_ROOT = Path(r"C:\Users\pleon\EODP_files\SHARED\EODP_TER_2021\EODP-TS-L1B")
REFERENCE_DIR = DATA_ROOT / "output"
EQUALIZED_DIR = DATA_ROOT / "output_eq"
NON_EQUALIZED_DIR = DATA_ROOT / "output_not_eq"
ISRF_DIR = DATA_ROOT / "input"
RESULT_DIR = DATA_ROOT / "test_output_L1B_calibration"
BANDS = ("VNIR-0", "VNIR-1", "VNIR-2", "VNIR-3")
# Guide to the Subject R3.0: strict relative difference threshold, percent.
RELATIVE_TOLERANCE_PERCENT = 1e-3


def read_toa(folder, filename):
    path = folder / filename
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with Dataset(path, "r") as ds:
        if "toa" not in ds.variables:
            raise KeyError(f"Variable 'toa' missing: {path}")
        values = np.ma.filled(np.ma.asarray(ds.variables["toa"][:], dtype=np.float64), np.nan)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D TOA in {path}, got {values.shape}")
    return values


def compare_arrays(actual, reference):
    if actual.shape != reference.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} vs {reference.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("NaN or Inf in arrays: comparison cannot be accepted")
    diff = np.abs(actual - reference)
    # A reference value of zero is accepted only when actual is also exactly zero.
    nonzero = reference != 0
    relative = np.zeros_like(diff)
    relative[nonzero] = 100.0 * diff[nonzero] / np.abs(reference[nonzero])
    relative[~nonzero] = np.where(diff[~nonzero] == 0, 0.0, np.inf)
    maximum_relative = float(np.max(relative))
    maximum_absolute = float(np.max(diff))
    passed = bool(np.all(relative < RELATIVE_TOLERANCE_PERCENT))
    return maximum_absolute, maximum_relative, passed


def plot_profiles(band, isrf, eq, no_eq):
    row_index = isrf.shape[0] // 2
    x = np.arange(isrf.shape[1])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, isrf[row_index], color="tab:red", label="ISRF radiance", linewidth=1.8)
    ax.plot(x, eq[row_index], color="tab:blue", label="L1B equalized", linewidth=1.5)
    ax.plot(x, no_eq[row_index], color="tab:green", label="L1B non-equalized", linewidth=1.1)
    ax.set_title(f"L1B radiometric calibration — {band} (ALT={row_index})")
    ax.set_xlabel("ACT pixel index")
    ax.set_ylabel("TOA radiance [mW/(m² sr)]")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = RESULT_DIR / f"l1b_calibration_{band}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    print("L1B radiometric calibration: equalized / non-equalized / ISRF")
    print(f"Strict relative error threshold: {RELATIVE_TOLERANCE_PERCENT:g}%")
    rows = []
    for band in BANDS:
        restored_name = f"l1b_toa_{band}.nc"
        eq_stage_name = f"l1b_toa_eq_{band}.nc"
        isrf_name = f"ism_toa_isrf_{band}.nc"
        restored_ref = read_toa(REFERENCE_DIR, restored_name)
        restored_eq = read_toa(EQUALIZED_DIR, restored_name)
        intermediate_ref = read_toa(REFERENCE_DIR, eq_stage_name)
        intermediate_eq = read_toa(EQUALIZED_DIR, eq_stage_name)
        restored_no_eq = read_toa(NON_EQUALIZED_DIR, restored_name)
        isrf = read_toa(ISRF_DIR, isrf_name)
        if not (restored_ref.shape == restored_no_eq.shape == isrf.shape):
            raise ValueError(f"Radiance arrays have mismatching shapes for {band}")

        print(f"\n{band} | shape {restored_eq.shape} | central ALT={restored_eq.shape[0] // 2}")
        for label, actual, reference in (
            ("Equalization output", intermediate_eq, intermediate_ref),
            ("Restored L1B TOA", restored_eq, restored_ref),
        ):
            abs_max, rel_max, passed = compare_arrays(actual, reference)
            print(f"  {label}: {'PASS' if passed else 'FAIL'}; "
                  f"max_abs_diff={abs_max:.9g}; max_rel_diff={rel_max:.9g}%")
            rows.append({"band": band, "product": label,
                         "samples": actual.size, "max_abs_diff": abs_max,
                         "max_relative_diff_percent": rel_max,
                         "threshold_percent": RELATIVE_TOLERANCE_PERCENT,
                         "result": "PASS" if passed else "FAIL"})
        path = plot_profiles(band, isrf, restored_eq, restored_no_eq)
        print(f"  Saved calibration profile: {path.name}")

    csv_path = RESULT_DIR / "l1b_crossvalidation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    total_pass = all(row["result"] == "PASS" for row in rows)
    print(f"\nOverall L1B cross-validation: {'PASS' if total_pass else 'FAIL'}")
    print(f"Saved 4 plots and CSV in: {RESULT_DIR}")


if __name__ == "__main__":
    main()
