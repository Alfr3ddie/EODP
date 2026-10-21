"""EODP-ALG-ISM-1020: radiance-to-irradiance conversion check.
"""

from pathlib import Path
import csv
import sys
import numpy as np
from netCDF4 import Dataset

# This file is intended to be located at EODP/ism/test/test_ism_conversion.py.
# Add the EODP project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.ismConfig import ismConfig

MY_OUTPUT = Path(r"C:\Users\pleon\EODP_files\SHARED\EODP_TER_2021\EODP-TS-ISM\myoutput")
REPORT_OUTPUT = Path(r"C:\Users\pleon\EODP_files\SHARED\EODP_TER_2021\EODP-TS-ISM\test_output_ISM_2")

BANDS = ["VNIR-0", "VNIR-1", "VNIR-2", "VNIR-3"]
PIXEL = (50, 75)  # (ALT, ACT)


def read_toa(file_path):
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    with Dataset(file_path, "r") as ds:
        toa = np.ma.asarray(ds.variables["toa"][:]).filled(np.nan)
    return np.asarray(toa, dtype=np.float64)


def main():
    cfg = ismConfig()  # Use the configuration from YOUR PyCharm project.
    if not MY_OUTPUT.is_dir():
        raise FileNotFoundError(f"Output directory not found: {MY_OUTPUT}")
    REPORT_OUTPUT.mkdir(parents=True, exist_ok=True)

    factor = cfg.Tr * np.pi / 4.0 * (cfg.D / cfg.f) ** 2
    print("EODP-ALG-ISM-1020 - Radiance to irradiance")
    print(f"Tr = {cfg.Tr:.10g} [-]")
    print(f"D  = {cfg.D:.10g} m")
    print(f"f  = {cfg.f:.10g} m")
    print(f"D/f = {cfg.D / cfg.f:.12g} [-]")
    print(f"Conversion factor K = {factor:.12g} [-]")
    print("Note: this factor is common to all bands when Tr, D and f are common.\n")

    i, j = PIXEL
    rows = []
    for band in BANDS:
        isrf_file = MY_OUTPUT / f"ism_toa_isrf_{band}.nc"
        optical_file = MY_OUTPUT / f"ism_toa_optical_{band}.nc"
        isrf = read_toa(isrf_file)
        optical = read_toa(optical_file)
        if isrf.ndim != 2 or optical.shape != isrf.shape:
            raise ValueError(f"Unexpected TOA shape for {band}: {isrf.shape} / {optical.shape}")
        if i >= isrf.shape[0] or j >= isrf.shape[1]:
            raise IndexError(f"Pixel {PIXEL} is outside the TOA matrix")

        radiance = float(isrf[i, j])
        irradiance_before_mtf = radiance * factor
        irradiance_after_mtf = float(optical[i, j])
        if not np.isfinite([radiance, irradiance_before_mtf, irradiance_after_mtf]).all():
            raise ValueError(f"Non-finite value at pixel {PIXEL} for {band}")

        print(f"{band}")
        print(f"  ISRF radiance [{i},{j}]:              {radiance:.12g} mW/(m^2 sr)")
        print(f"  Analytical irradiance BEFORE MTF:    {irradiance_before_mtf:.12g} mW/m^2")
        print(f"  Optical file irradiance AFTER MTF:   {irradiance_after_mtf:.12g} mW/m^2")
        print("  The last two values need not agree: the optical output includes MTF.\n")
        rows.append({
            "band": band,
            "ALT_pixel": i,
            "ACT_pixel": j,
            "Tr": cfg.Tr,
            "D_m": cfg.D,
            "f_m": cfg.f,
            "D_over_f": cfg.D / cfg.f,
            "conversion_factor": factor,
            "isrf_radiance_mW_m2_sr": radiance,
            "analytic_irradiance_before_MTF_mW_m2": irradiance_before_mtf,
            "optical_irradiance_after_MTF_mW_m2": irradiance_after_mtf,
        })

    csv_path = REPORT_OUTPUT / "ism_radiance_to_irradiance.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
