"""L1C TOA validation against the supplied reference products.
"""
from pathlib import Path
import numpy as np
from netCDF4 import Dataset

DATA_ROOT = Path(r"C:\Users\pleon\EODP_files\SHARED\EODP_TER_2021\EODP-TS-L1C")
MY_OUTPUT = DATA_ROOT / "Myoutputs"
REFERENCE = DATA_ROOT / "output"
BANDS = ("VNIR-0", "VNIR-1", "VNIR-2", "VNIR-3")
TOL_PERCENT = 0.001
MIN_PASS_FRACTION = 0.99


def load_points(file_path):
    if not file_path.is_file():
        raise FileNotFoundError(f"Missing NetCDF file: {file_path}")
    with Dataset(file_path, "r") as ds:
        arrays = {
            name: np.asarray(ds.variables[name][:], dtype=np.float64).ravel()
            for name in ("lat", "lon", "toa")
        }
    sizes = {v.size for v in arrays.values()}
    if len(sizes) != 1:
        raise ValueError(f"lat, lon and toa have different lengths in {file_path}")
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError(f"Non-finite lat, lon or toa in {file_path}")
    # The point order can vary between executions.  Sort by both coordinates.
    order = np.lexsort((arrays["lon"], arrays["lat"]))
    return {name: values[order] for name, values in arrays.items()}


def validate_band(band):
    ours = load_points(MY_OUTPUT / f"l1c_toa_{band}.nc")
    ref = load_points(REFERENCE / f"l1c_toa_{band}.nc")

    n = ref["toa"].size
    if ours["toa"].size != n:
        raise ValueError(f"{band}: different number of grid points")

    if not (np.array_equal(ours["lat"], ref["lat"])
            and np.array_equal(ours["lon"], ref["lon"])):
        raise ValueError(f"{band}: geographic coordinate sets do not match")

    # Relative tolerance for non-zero reference values; absolute fallback for zero.
    absolute_difference = np.abs(ours["toa"] - ref["toa"])
    tolerance = np.where(
        ref["toa"] != 0,
        np.abs(ref["toa"]) * TOL_PERCENT / 100.0,
        0.0,
    )
    passing = absolute_difference <= tolerance
    fraction = float(np.mean(passing))
    max_abs = float(np.max(absolute_difference))
    result = "PASS" if fraction >= MIN_PASS_FRACTION else "FAIL"
    print(f"{band}: points={n} | {result} | within_tol={100*fraction:.4f}% "
          f"| max_abs_diff={max_abs:.12g}")
    return result == "PASS"


def main():
    print("L1C TOA comparison with reference")
    print(f"Tolerance: {TOL_PERCENT}% for at least {MIN_PASS_FRACTION * 100:.2f}% of points")
    print("Samples aligned by geographic latitude/longitude before comparison")
    results = [validate_band(band) for band in BANDS]
    print("Overall L1C TOA:", "PASS" if all(results) else "FAIL")


if __name__ == "__main__":
    main()
