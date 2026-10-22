"""L1B TOA verification
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

# Existing folders on the user's Windows computer.
DATA_ROOT = Path(r"C:\Users\pleon\EODP_files\SHARED\EODP_TER_2021\EODP-TS-L1B")
MY_OUTPUT = DATA_ROOT / "output_eq"
REFERENCE = DATA_ROOT / "output"
REPORT_OUTPUT = DATA_ROOT / "test_output_L1B_1"

BANDS = ("VNIR-0", "VNIR-1", "VNIR-2", "VNIR-3")
RELATIVE_TOLERANCE_PERCENT = 0.01
REQUIRED_PERCENT = 95.45  # Two-sigma fraction for a normal distribution.


def read_toa(path):
    if not path.is_file():
        raise FileNotFoundError(f"NetCDF file not found: {path}")
    with Dataset(path, "r") as ds:
        if "toa" not in ds.variables:
            raise KeyError(f"Variable 'toa' not found in {path}")
        toa = np.asarray(ds.variables["toa"][:], dtype=np.float64)
    if toa.ndim != 2:
        raise ValueError(f"Expected a 2D TOA in {path}, got {toa.shape}")
    return toa


def check_difference(ours, reference):
    if ours.shape != reference.shape:
        raise ValueError(f"TOA shapes differ: ours={ours.shape}, reference={reference.shape}")

    finite = np.isfinite(ours) & np.isfinite(reference)
    difference = np.abs(ours - reference)
    # Relative percentage error for nonzero reference values.
    # For a zero reference, only an exact zero is counted as a match.
    within = np.zeros(reference.shape, dtype=bool)
    nonzero = finite & (reference != 0)
    within[nonzero] = (100.0 * difference[nonzero] / np.abs(reference[nonzero])
                       < RELATIVE_TOLERANCE_PERCENT)
    zero = finite & (reference == 0)
    within[zero] = (ours[zero] == 0)

    fraction_percent = 100.0 * np.count_nonzero(within) / reference.size
    max_absolute_difference = float(np.max(difference[finite])) if np.any(finite) else float("nan")
    passed = fraction_percent >= REQUIRED_PERCENT
    return passed, fraction_percent, int(np.count_nonzero(~finite)), max_absolute_difference


def main():
    REPORT_OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    all_passed = True

    print("L1B TOA comparison")
    print(f"Threshold: relative difference < {RELATIVE_TOLERANCE_PERCENT}% "
          f"for at least {REQUIRED_PERCENT}% of pixels")

    for ax, band in zip(axes.flat, BANDS):
        name = f"l1b_toa_{band}.nc"
        ours = read_toa(MY_OUTPUT / name)
        ref = read_toa(REFERENCE / name)
        passed, fraction, invalid_count, max_diff = check_difference(ours, ref)
        all_passed &= passed

        # ACT profile at the central ALT row, as in Figure 6-1.
        central_alt = ours.shape[0] // 2
        act = np.arange(ours.shape[1])
        ax.plot(act, ref[central_alt, :], color="tab:blue", label="Ref TOA", linewidth=1.7)
        ax.plot(act, ours[central_alt, :], color="tab:orange", linestyle="--",
                label="Test TOA", linewidth=1.4)
        ax.set(title=f"L1B TOA {band}", xlabel="ACT [-]",
               ylabel="Radiances [mW/m²/sr]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

        print(f"{band}: shape={ours.shape}, central ALT={central_alt} | "
              f"{'PASS' if passed else 'FAIL'} | "
              f"within_tol={fraction:.4f}% | invalid={invalid_count} | "
              f"max_abs_diff={max_diff:.10g}")

    figure_path = REPORT_OUTPUT / "l1b_toa_comparison.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    print(f"Saved Figure: {figure_path}")
    print(f"Overall L1B TOA: {'PASS' if all_passed else 'FAIL'}")


if __name__ == "__main__":
    main()
