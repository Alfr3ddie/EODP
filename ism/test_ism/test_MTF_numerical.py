from pathlib import Path
import warnings

import numpy as np
from scipy.special import j1
from netCDF4 import Dataset

# ============================================================
# 1. DIRECTORIES
# ============================================================

BASE_DIR = Path(
    r"C:\Users\pleon\EODP_files\SHARED\EODP_TER_2021\EODP-TS-ISM"
)

REFERENCE_DIR = BASE_DIR / "output"

OUTPUT_DIR = (
        BASE_DIR / "Test_MTF" / "output_mtf"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. ISM CONFIGURATION
# ============================================================

NLINES = 100
NCOLUMNS = 150

D = 0.15
FOCAL = 0.5262
PIX_SIZE = 3e-5

KLF = 180
WLF = 1e-7

KHF = 300
WHF = 1e-7

DEFOCUS = 2
KSMEAR = 0.191
KMOTION = 0.02

BANDS = {
    "VNIR-0": 490e-9,
    "VNIR-1": 665e-9,
    "VNIR-2": 865e-9,
    "VNIR-3": 945e-9
}

TOLERANCE = 1e-3  # Percentage


# ============================================================
# 3. CALCULATE MTF MATRICES
# ============================================================

def calculate_mtf(wavelength):
    fstep_alt = 1 / (NLINES * PIX_SIZE)
    fstep_act = 1 / (NCOLUMNS * PIX_SIZE)

    eps = 1e-6

    f_alt = np.arange(
        -1 / (2 * PIX_SIZE),
        1 / (2 * PIX_SIZE) - eps,
        fstep_alt
    )

    f_act = np.arange(
        -1 / (2 * PIX_SIZE),
        1 / (2 * PIX_SIZE) - eps,
        fstep_act
    )

    f_alt_2d, f_act_2d = np.meshgrid(
        f_alt,
        f_act,
        indexing="ij"
    )

    f_2d = np.sqrt(
        f_alt_2d ** 2 + f_act_2d ** 2
    )

    f_cutoff = D / (wavelength * FOCAL)

    fn_2d = f_2d * PIX_SIZE
    fr_2d = f_2d / f_cutoff
    fn_alt = f_alt * PIX_SIZE

    # Diffraction MTF
    Hdiff = (2 / np.pi) * (
            np.arccos(fr_2d)
            - fr_2d * np.sqrt(1 - fr_2d ** 2)
    )

    # Defocus MTF
    x = (
            np.pi * DEFOCUS
            * fr_2d * (1 - fr_2d)
    )

    Hdefoc = np.ones_like(x)
    nonzero = np.abs(x) > 1e-12

    Hdefoc[nonzero] = (
            2 * j1(x[nonzero])
            / x[nonzero]
    )

    # Wavefront error MTF
    Hwfe = np.exp(
        -fr_2d * (1 - fr_2d)
        * (
                KLF * (WLF / wavelength) ** 2
                + KHF * (WHF / wavelength) ** 2
        )
    )

    # Detector MTF
    Hdet = np.abs(np.sinc(fn_2d))

    # Smearing MTF
    Hsmear = np.repeat(
        np.sinc(KSMEAR * fn_alt)[:, None],
        NCOLUMNS,
        axis=1
    )

    # Motion MTF
    Hmotion = np.sinc(
        KMOTION * fn_2d
    )

    # System MTF
    Hsys = (
            Hdiff * Hdefoc * Hwfe
            * Hdet * Hsmear * Hmotion
    )

    return {
        "Hdiff": Hdiff,
        "Hdefoc": Hdefoc,
        "Hwfe": Hwfe,
        "Hdet": Hdet,
        "Hsmear": Hsmear,
        "Hmotion": Hmotion,
        "Hsys": Hsys
    }


# ============================================================
# 4. SAVE NETCDF
# ============================================================

def save_nc(path, matrix):
    data = np.asarray(
        matrix,
        dtype=np.float32
    )

    with Dataset(
            path,
            "w",
            format="NETCDF4"
    ) as nc:
        nc.createDimension(
            "alt_lines", NLINES
        )

        nc.createDimension(
            "act_columns", NCOLUMNS
        )

        variable = nc.createVariable(
            "mat",
            "f4",
            ("alt_lines", "act_columns")
        )

        # Suppress only the NumPy 2.5 deprecation warning
        # potentially raised inside netCDF4.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Setting the shape on a NumPy array has been deprecated.*",
                category=DeprecationWarning
            )

            variable[:] = data


# ============================================================
# 5. READ NETCDF
# ============================================================

def read_nc(path):
    with Dataset(path, "r") as nc:
        data = np.asarray(
            nc.variables["mat"][:]
        ).copy()

    return data


# ============================================================
# 6. CROSS-VALIDATION
# ============================================================

def validate(ours, reference):
    if ours.shape != reference.shape:
        return False, np.nan, np.nan

    if (
            not np.all(np.isfinite(ours))
            or not np.all(np.isfinite(reference))
    ):
        return False, np.nan, np.nan

    absolute_diff = np.abs(
        ours.astype(np.float64)
        - reference.astype(np.float64)
    )

    max_abs = float(
        np.max(absolute_diff)
    )

    nonzero = reference != 0

    relative_diff = np.zeros(
        reference.shape,
        dtype=np.float64
    )

    relative_diff[nonzero] = (
            absolute_diff[nonzero]
            / np.abs(reference[nonzero])
            * 100
    )

    if np.any(
            ~nonzero & (absolute_diff != 0)
    ):
        max_rel = float("inf")
    else:
        max_rel = float(
            np.max(relative_diff)
        )

    passed = max_rel < TOLERANCE

    return passed, max_abs, max_rel


# ============================================================
# 7. MAIN TEST
# ============================================================

def main():
    total_tests = 0
    passed_tests = 0

    print("EODP-TS-ISM - MTF CROSS-VALIDATION")
    print(f"Reference: {REFERENCE_DIR}")
    print(f"Generated: {OUTPUT_DIR}")
    print(f"Criterion: max relative difference < {TOLERANCE}%")

    for band, wavelength in BANDS.items():

        print(f"\n--- {band} ---")

        matrices = calculate_mtf(
            wavelength
        )

        for name, matrix in matrices.items():

            filename = f"{name}_{band}.nc"

            our_path = OUTPUT_DIR / filename
            reference_path = REFERENCE_DIR / filename

            save_nc(
                our_path,
                matrix
            )

            total_tests += 1

            if not reference_path.exists():
                print(
                    f"{name:8s} REFERENCE NOT FOUND"
                )

                continue

            ours = read_nc(
                our_path
            )

            reference = read_nc(
                reference_path
            )

            passed, max_abs, max_rel = validate(
                ours,
                reference
            )

            if passed:
                passed_tests += 1

            status = (
                "PASS" if passed else "FAIL"
            )

            print(
                f"{name:8s} {status} | "
                f"max_abs={max_abs:.10g} | "
                f"max_rel={max_rel:.10g}%"
            )

    print("\n======================================")
    print("MTF VALIDATION SUMMARY")
    print("======================================")

    print(
        f"Tests passed: {passed_tests}/{total_tests}"
    )

    if passed_tests == total_tests:
        print("GLOBAL RESULT: PASS")
    else:
        print("GLOBAL RESULT: FAIL")

    print("\nMTF validation completed.")


if __name__ == "__main__":
    main()
