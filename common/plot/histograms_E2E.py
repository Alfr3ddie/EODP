from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# Input and output directories
input_dir = Path(r"C:\Users\pleon\EODP_files\SHARED\EODP-TS-E2E\output_2")
output_dir = Path(r"C:\Users\pleon\EODP_files\SHARED\EODP-TS-E2E\Histograms")

# Create output directory if it does not exist
output_dir.mkdir(parents=True, exist_ok=True)

# TOA bands
bands = ["VNIR-0", "VNIR-1", "VNIR-2", "VNIR-3"]

# Generate histogram for each band
for band in bands:

    file_path = input_dir / f"l1b_toa_{band}.nc"

    # Check if file exists
    if not file_path.exists():
        print(f"File not found: {file_path}")
        continue

    # Read TOA radiance
    with Dataset(file_path, "r") as nc:
        toa = nc.variables["toa"][:]

    # Convert data to 1D array
    toa = np.ma.asarray(toa).compressed()

    # Remove NaN and infinite values
    toa = toa[np.isfinite(toa)]

    if toa.size == 0:
        print(f"No valid data for {band}")
        continue

    # Create histogram
    plt.figure(figsize=(9, 6))

    plt.hist(
        toa,
        bins=50,
        color="steelblue",
        edgecolor="black",
        linewidth=0.5
    )

    plt.xlabel("TOA Radiance [W/(m² sr)]")
    plt.ylabel("Counts")
    plt.title(f"TOA Radiance Histogram - {band}")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    output_path = output_dir / f"toa_histogram_{band}.png"
    plt.savefig(output_path, dpi=300)

    # Display figure
    plt.show()
    plt.close()

    print(f"Histogram saved: {output_path}")