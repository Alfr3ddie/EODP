"""ISM_4: Detection and Video Conversion verification against ATBD §§7.2–7.3.

Standalone script: reads existing NetCDF outputs, does not change ISM algorithms.
Requires numpy, matplotlib, scipy, netCDF4 and the project's config.ismConfig.
"""
from pathlib import Path
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from scipy.constants import h, c

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from config.ismConfig import ismConfig

# Existing working data folders; do not modify the ISM source files.
DATA_ROOT = Path(r"C:\Users\pleon\EODP_files\SHARED\EODP_TER_2021\EODP-TS-ISM")
MY_OUTPUT = DATA_ROOT / "myoutput"
REFERENCE = DATA_ROOT / "output"
REPORT_OUTPUT = DATA_ROOT / "test_output_ISM_4"
BANDS = ["VNIR-0", "VNIR-1", "VNIR-2", "VNIR-3"]
# Validation thresholds for this standalone test (not presented as an ATBD requirement).
ABS_TOL_ELECTRONS = 0.1  # electrons; accommodate float32 NetCDF storage
ABS_TOL_DN = 0.0        # exact equality for digital counts, when possible
REL_TOL_PERCENT = 0.001
MIN_PASS_PERCENT = 99


def read_toa(folder, stem, band, required=True):
    path = folder / f"{stem}{band}.nc"
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Missing NetCDF file: {path}")
        return None
    with Dataset(path, "r") as ds:
        if "toa" not in ds.variables:
            raise KeyError(f"Variable 'toa' missing in: {path}")
        arr = np.ma.filled(ds.variables["toa"][:], np.nan).astype(np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D TOA, got {arr.shape}: {path}")
    return arr


def statistics(ours, ref, abs_tol):
    if ours.shape != ref.shape:
        raise ValueError(f"Incompatible TOA shapes: {ours.shape} vs {ref.shape}")
    finite = np.isfinite(ours) & np.isfinite(ref)
    n = ours.size
    diff = np.abs(ours - ref)
    good_abs = finite & (diff <= abs_tol)
    # Relative differences exclude reference zeros; absolute tolerance handles them.
    rel = np.full_like(diff, np.inf)
    nonzero = finite & (ref != 0)
    rel[nonzero] = 100.0 * diff[nonzero] / np.abs(ref[nonzero])
    pass_rel = finite & ((rel <= REL_TOL_PERCENT) | ((ref == 0) & (diff <= abs_tol)))
    return {
        "shape": str(ours.shape),
        "invalid": int(n - np.count_nonzero(finite)),
        "max_abs_diff": float(np.max(diff[finite])) if np.any(finite) else float("nan"),
        "exact_equal_pct": 100 * np.count_nonzero(finite & (diff == 0)) / n,
        "within_absolute_tol_pct": 100 * np.count_nonzero(good_abs) / n,
        "within_relative_tol_pct": 100 * np.count_nonzero(pass_rel) / n,
        "reference_min": float(np.min(ref[finite])) if np.any(finite) else float("nan"),
        "reference_max": float(np.max(ref[finite])) if np.any(finite) else float("nan"),
        "pass": bool(np.all(finite) and (100 * np.count_nonzero(pass_rel) / n >= MIN_PASS_PERCENT)),
    }


def save_comparison(ours, ref, band, label, units):
    mid = ours.shape[0] // 2
    x = np.arange(ours.shape[1])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, ref[mid], label="Reference", lw=2)
    ax.plot(x, ours[mid], "k--", label="Our output", lw=1.5)
    ax.set(title=f"{label} comparison - {band} (ALT={mid})",
           xlabel="ACT pixel index", ylabel=units)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_OUTPUT / f"{label.lower()}_comparison_{band}.png", dpi=160)
    plt.close(fig)


def main():
    cfg = ismConfig()
    REPORT_OUTPUT.mkdir(parents=True, exist_ok=True)
    for folder in (MY_OUTPUT, REFERENCE):
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")

    print("ISM_4 - Detection and Video Conversion, ATBD sections 7.2 and 7.3", flush=True)
    print(f"Project config: w={cfg.pix_size:.9g} m, tint={cfg.t_int:.9g} s, "
          f"QE={cfg.QE:.6g}, FWC={cfg.FWC:.6g} e-", flush=True)
    print(f"Effects active: PRNU={cfg.apply_prnu}, DS={cfg.apply_dark_signal}, "
          f"bad/dead={cfg.apply_bad_dead}", flush=True)
    print("Independent comparisons with official NetCDF reference outputs.", flush=True)
    rows = []
    for index, band in enumerate(BANDS):
        print(f"\n--- {band} ---", flush=True)
        optical = read_toa(MY_OUTPUT, "ism_toa_optical_", band)
        electrons = read_toa(MY_OUTPUT, "ism_toa_e_", band)
        detection = read_toa(MY_OUTPUT, "ism_toa_detection_", band)
        final = read_toa(MY_OUTPUT, "ism_toa_", band)
        if not (optical.shape == electrons.shape == detection.shape == final.shape):
            raise ValueError(f"Inconsistent array sizes for {band}")

        wv = float(cfg.wv[index])
        # ATBD §7.2.1: irradiance mW/m2 -> photons.
        factor_ph = 1e-3 * float(cfg.pix_size)**2 * float(cfg.t_int) * wv / (h * c)
        photons = optical * factor_ph
        # ATBD §7.2.2: photons -> electrons and full-well saturation.
        expected_e = np.minimum(photons * float(cfg.QE), float(cfg.FWC))
        s_e = statistics(electrons, expected_e, ABS_TOL_ELECTRONS)
        print(f"Photons / (mW m^-2): {factor_ph:.12g}; electrons / photon: {cfg.QE:.9g}")
        print(f"Electron conversion: max |delta|={s_e['max_abs_diff']:.6g} e-, "
              f"within 0.1 e-={s_e['within_absolute_tol_pct']:.4f}%")
        print(f"Raw expected well-saturated pixels: "
              f"{np.count_nonzero(photons * cfg.QE > cfg.FWC)} / {photons.size}")

        # ATBD §§7.3.1–7.3.2: model the VCU from the saved post-detection TOA.
        factor_ev = float(cfg.OCF) * float(cfg.ADC_gain)
        volts = detection * factor_ev
        max_dn = int(2 ** int(cfg.bit_depth) - 1)
        voltage_range = float(cfg.max_voltage) - float(cfg.min_voltage)
        if voltage_range <= 0:
            raise ValueError("Invalid voltage range in ismConfig")
        dn_unclipped = np.rint(volts / voltage_range * max_dn)
        dn_expected = np.clip(dn_unclipped, 0, max_dn)
        # ATBD §7.3.2 calls for clipping; compare both versions to expose
        # implementations that omit clipping rather than concealing the difference.
        s_dn = statistics(final, dn_expected, ABS_TOL_DN)
        exact_dn = bool(np.array_equal(final, dn_expected))
        matches_unclipped = bool(np.array_equal(final, dn_unclipped))
        print(f"Volts / electron: {factor_ev:.12g}; DN / volt: {max_dn/voltage_range:.12g}")
        print(f"Output DN: min={np.min(final):.6g}, max={np.max(final):.6g}, "
              f"at ADC ceiling={np.count_nonzero(final == max_dn)} / {final.size}")
        print(f"Calculated DN exceeding ADC range before clipping: "
              f"{np.count_nonzero((dn_unclipped < 0) | (dn_unclipped > max_dn))}")
        print(f"ATBD-clipped DN reconstruction: exact={exact_dn}, "
              f"matches un-clipped variant={matches_unclipped}")

        # Validate intermediate detection and final DN output against reference.
        for label, stem, ours, unit, tolerance in (
            ("Detection", "ism_toa_detection_", detection, "Electrons [e-]", ABS_TOL_ELECTRONS),
            ("Final_DN", "ism_toa_", final, "Digital number [DN]", ABS_TOL_DN),
        ):
            ref = read_toa(REFERENCE, stem, band)
            s = statistics(ours, ref, tolerance)
            print(f"{label} vs reference: {'PASS' if s['pass'] else 'CHECK'} "
                  f"within_rel_tol={s['within_relative_tol_pct']:.4f}% "
                  f"exact={s['exact_equal_pct']:.4f}% "
                  f"invalid={s['invalid']} max_abs_diff={s['max_abs_diff']:.6g}", flush=True)
            save_comparison(ours, ref, band, label, unit)
            rows.append({"band": band, "stage": label, **s,
                         "photon_factor_per_mw_m2": factor_ph,
                         "electron_factor_qe": float(cfg.QE),
                         "volts_per_electron": factor_ev,
                         "dn_per_volt": max_dn / voltage_range,
                         "electron_formula_max_abs_diff": s_e["max_abs_diff"],
                         "expected_well_saturated": int(np.count_nonzero(photons * cfg.QE > cfg.FWC)),
                         "final_pixels_at_max_dn": int(np.count_nonzero(final == max_dn)),
                         "calculated_dn_out_of_range": int(np.count_nonzero((dn_unclipped < 0) | (dn_unclipped > max_dn))),
                         "final_matches_atbd_clipped": exact_dn,
                         "final_matches_unclipped": matches_unclipped})
    csv_path = REPORT_OUTPUT / "ism_detection_vcu_validation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved 8 comparison plots and CSV to: {REPORT_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
