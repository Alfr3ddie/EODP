
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset


MY_OUTPUT = Path(r"C:\Users\pleon\EODP_files\SHARED\EODP_TER_2021\EODP-TS-ISM\myoutput")
REFERENCE_OUTPUT = Path(r"C:\Users\pleon\EODP_files\SHARED\EODP_TER_2021\EODP-TS-ISM\output")
REPORT_OUTPUT = Path(r"C:\Users\pleon\EODP_files\SHARED\EODP_TER_2021\EODP-TS-ISM\test_output_ISM_1")


BANDS = ["VNIR-0", "VNIR-1", "VNIR-2", "VNIR-3"]
STAGES = {"ISRF": "ism_toa_isrf_", "Optical": "ism_toa_optical_"}
REL_TOL_PERCENT = 0.001       # 0.001 percent
MIN_PASS_FRACTION = 0.99  # three sigma
PIXEL = (50, 75)             # requested spot check: ALT row, ACT column


def load_toa(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing NetCDF file: {path}")
    with Dataset(path, 'r') as ds:
        arr = np.ma.asarray(ds.variables['toa'][:]).filled(np.nan)
    return np.asarray(arr, dtype=np.float64)


def check_one(stage, band, prefix):
    filename = f"{prefix}{band}.nc"
    ours = load_toa(MY_OUTPUT / filename)
    ref = load_toa(REFERENCE_OUTPUT / filename)
    if ours.shape != ref.shape:
        raise ValueError(f"Shape mismatch {filename}: ours {ours.shape}, reference {ref.shape}")
    if ours.ndim != 2:
        raise ValueError(f"Expected 2D TOA: {filename}, got {ours.ndim}D")

    finite = np.isfinite(ours) & np.isfinite(ref)
    n_total = ours.size
    abs_diff = np.abs(ours - ref)
    # Relative error measured against the reference, where nonzero.
    # At reference == 0, require exactly zero difference instead of dividing by zero.
    relative_pct = np.full(ours.shape, np.inf, dtype=float)
    nonzero = finite & (ref != 0)
    relative_pct[nonzero] = 100 * abs_diff[nonzero] / np.abs(ref[nonzero])
    zeros = finite & (ref == 0)
    relative_pct[zeros] = np.where(abs_diff[zeros] == 0, 0.0, np.inf)
    acceptable = finite & (relative_pct <= REL_TOL_PERCENT)
    percent_good = 100 * np.count_nonzero(acceptable) / n_total
    passed = np.count_nonzero(acceptable) / n_total >= MIN_PASS_FRACTION

    i, j = PIXEL
    if i >= ours.shape[0] or j >= ours.shape[1]:
        raise IndexError(f"Pixel {PIXEL} is outside image {ours.shape} in {filename}")

    label = f"{stage} {band}"
    print(f"{label:20s}  {'PASS' if passed else 'FAIL':4s}  "
          f"within_tol={percent_good:8.4f}%  invalid={n_total - np.count_nonzero(finite)}  "
          f"max_abs_diff={np.nanmax(abs_diff[finite]) if finite.any() else float('nan'):.6g}")
    print(f"  pixel[{i},{j}]: ours={ours[i,j]:.10g}, reference={ref[i,j]:.10g}, "
          f"relative_error={relative_pct[i,j]:.6g}%")

    # ACT profile through central along-track index;
    mid = ours.shape[0] // 2
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(ref[mid, :], color='tab:blue', label='Reference', linewidth=2)
    ax.plot(ours[mid, :], color='black', linestyle='--', label='Our output', linewidth=1.6)
    ax.set(title=f'{stage} TOA comparison - {band} (ALT={mid})',
           xlabel='ACT pixel index',
           ylabel='Radiance [mW/(m² sr)]' if stage == 'ISRF' else 'Irradiance [mW/m²]')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_OUTPUT / f'{stage.lower()}_comparison_{band}.png', dpi=180)
    plt.close(fig)

    return {
        'stage': stage, 'band': band, 'filename': filename,
        'n_pixels': n_total, 'fraction_within_tolerance_pct': round(percent_good, 6),
        'invalid_pixel_count': int(n_total - np.count_nonzero(finite)),
        'max_abs_diff': float(np.max(abs_diff[finite])) if finite.any() else float('nan'),
        'my_pixel_50_75': float(ours[i, j]),
        'reference_pixel_50_75': float(ref[i, j]),
        'relative_error_pixel_50_75_pct': float(relative_pct[i, j]),
        'result': 'PASS' if passed else 'FAIL'
    }


def main():
    for d in (MY_OUTPUT, REFERENCE_OUTPUT):
        if not d.is_dir():
            raise FileNotFoundError(f'Directory not found, edit script paths: {d}')
    REPORT_OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for stage, prefix in STAGES.items():
        for band in BANDS:
            rows.append(check_one(stage, band, prefix))
    with (REPORT_OUTPUT / 'optical_validation_summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print('\nSaved 8 comparison plots and optical_validation_summary.csv to', REPORT_OUTPUT)

if __name__ == '__main__':
    main()
