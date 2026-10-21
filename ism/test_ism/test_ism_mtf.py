"""ISM system MTF: six contributors, ACT/ALT profiles and Nyquist analysis.
"""
from pathlib import Path
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from scipy.special import j1

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not (PROJECT_ROOT / 'config' / 'ismConfig.py').exists():
    PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))
from config.ismConfig import ismConfig

MY_OUTPUT = Path(r'C:\Users\pleon\EODP_files\SHARED\EODP_TER_2021\EODP-TS-ISM\myoutput')
REPORT_OUTPUT = Path(r'C:\Users\pleon\EODP_files\SHARED\EODP_TER_2021\EODP-TS-ISM\test_output_ISM_3')
BANDS = ['VNIR-0', 'VNIR-1', 'VNIR-2', 'VNIR-3']
CONTRIBUTORS = ['Diffraction', 'Defocus', 'WFE', 'Detector', 'Smearing', 'Motion']
COLORS = {
    'Diffraction': 'tab:blue', 'Defocus': 'tab:orange',
    'WFE': 'tab:green', 'Detector': 'tab:red',
    'Smearing': 'tab:purple', 'Motion': 'tab:brown',
    'System': 'black'
}

def image_shape(band):
    path = MY_OUTPUT / f'ism_toa_optical_{band}.nc'
    if not path.is_file():
        raise FileNotFoundError(f'Optical TOA file not found: {path}')
    with Dataset(path, 'r') as ds:
        shape = ds.variables['toa'].shape
    if len(shape) != 2:
        raise ValueError(f'Expected a 2D TOA in {path}, got {shape}')
    return shape


def calculate_mtf(shape, wavelength, cfg):
    n_alt, n_act = shape
    w = float(cfg.pix_size)
    fn_alt = np.fft.fftshift(np.fft.fftfreq(n_alt))
    fn_act = np.fft.fftshift(np.fft.fftfreq(n_act))
    freq_alt, freq_act = np.meshgrid(fn_alt, fn_act, indexing='ij')
    fn_2d = np.hypot(freq_alt, freq_act)
    optical_cutoff = cfg.D / (wavelength * cfg.f)
    fr = fn_2d / (w * optical_cutoff)

    hd = np.zeros_like(fr)
    inside = fr < 1.0
    q = fr[inside]
    hd[inside] = 2 / np.pi * (np.arccos(q) - q * np.sqrt(1 - q*q))

    x = np.pi * cfg.defocus * fr * (1 - fr)
    hf = np.ones_like(x)
    mask = np.abs(x) > 1e-12
    hf[mask] = 2 * j1(x[mask]) / x[mask]

    k_wfe = cfg.kLF * (cfg.wLF / wavelength)**2 + cfg.kHF * (cfg.wHF / wavelength)**2
    hw = np.exp(-fr * (1 - fr) * k_wfe)
    hdet = np.abs(np.sinc(fn_2d))
    hsmear = np.broadcast_to(np.sinc(cfg.ksmear * fn_alt)[:, None], shape)
    hmotion = np.sinc(cfg.kmotion * fn_2d)
    hsys = hd * hf * hw * hdet * hsmear * hmotion

    components = dict(zip(CONTRIBUTORS + ['System'],
                          (hd, hf, hw, hdet, hsmear, hmotion, hsys)))
    return fn_alt, fn_act, components, optical_cutoff


def save_plot(frequencies, curves, band, direction, central_pixel):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    center = len(frequencies) // 2
    x = frequencies[center:]
    for name in CONTRIBUTORS + ['System']:
        ax.plot(x, curves[name][center:], color=COLORS[name],
                linewidth=2.7 if name == 'System' else 1.4,
                linestyle='--' if name == 'System' else '-', label=f'{name} MTF')
    ax.axvline(0.5, color='gray', linestyle=':', linewidth=1.5, label='Nyquist')
    ax.set(xlabel='Normalized spatial frequency [cycles/pixel]',
           ylabel='MTF', title=f'System MTF - {direction} - {band} ({central_pixel})')
    ax.set_xlim(0, 0.51)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    filename = REPORT_OUTPUT / f'mtf_{direction.lower()}_{band}.png'
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def main():
    if not MY_OUTPUT.is_dir():
        raise FileNotFoundError(f'MY_OUTPUT folder not found: {MY_OUTPUT}')
    cfg = ismConfig()
    REPORT_OUTPUT.mkdir(parents=True, exist_ok=True)
    print('EODP-ALG-ISM-1030 - System MTF and Nyquist')
    print(f'Pixel size = {cfg.pix_size:.9g} m')
    print(f'D = {cfg.D:.9g} m; f = {cfg.f:.9g} m')
    print('Defining MTF: smallest contributor at the Nyquist frequency')

    rows = []
    for band_index, band in enumerate(BANDS):
        shape = image_shape(band)
        wavelength = float(cfg.wv[band_index])
        fn_alt, fn_act, components, cutoff = calculate_mtf(shape, wavelength, cfg)
        mid_alt, mid_act = shape[0] // 2, shape[1] // 2

        act_curves = {name: mat[mid_alt, :] for name, mat in components.items()}
        alt_curves = {name: mat[:, mid_act] for name, mat in components.items()}
        save_plot(fn_act, act_curves, band, 'ACT', f'ALT={mid_alt}')
        save_plot(fn_alt, alt_curves, band, 'ALT', f'ACT={mid_act}')

        # Even-sized FFT arrays sample Nyquist at -0.5 (index zero).
        act_values = {name: float(components[name][mid_alt, 0]) for name in CONTRIBUTORS}
        alt_values = {name: float(components[name][0, mid_act]) for name in CONTRIBUTORS}
        defining_act = min(act_values, key=act_values.get)
        defining_alt = min(alt_values, key=alt_values.get)
        nyquist_act = float(components['System'][mid_alt, 0])
        nyquist_alt = float(components['System'][0, mid_act])

        print(f'\n{band} | wavelength={wavelength*1e9:.1f} nm | shape={shape}')
        print('  Contributor           ACT at Nyquist    ALT at Nyquist')
        for name in CONTRIBUTORS:
            print(f'  {name:<20s} {act_values[name]:>13.6f}    {alt_values[name]:>13.6f}')
        print(f'  System               {nyquist_act:>13.6f}    {nyquist_alt:>13.6f}')
        print(f'  Defining MTF (ACT): {defining_act}')
        print(f'  Defining MTF (ALT): {defining_alt}')

        row = {
            'band': band, 'wavelength_nm': wavelength*1e9,
            'pixel_size_m': float(cfg.pix_size), 'cutoff_frequency_inv_m': cutoff,
            'nyquist_frequency_inv_m': 1/(2*float(cfg.pix_size)),
            'mtf_nyquist_ACT': nyquist_act, 'mtf_nyquist_ALT': nyquist_alt,
            'defining_MTF_ACT': defining_act, 'defining_MTF_ALT': defining_alt,
        }
        for name in CONTRIBUTORS:
            row[f'{name.lower()}_ACT_nyquist'] = act_values[name]
            row[f'{name.lower()}_ALT_nyquist'] = alt_values[name]
        rows.append(row)

    csv_path = REPORT_OUTPUT / 'ism_mtf_nyquist.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nSaved 8 MTF plots and CSV to: {REPORT_OUTPUT}')


if __name__ == '__main__':
    main()
