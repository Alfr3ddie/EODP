from ism.src.initIsm import initIsm
import numpy as np
from common.plot.plotMat2D import plotMat2D
from common.plot.plotF import plotF


class videoChainPhase(initIsm):

    def __init__(self, auxdir, indir, outdir):
        super().__init__(auxdir, indir, outdir)

    def compute(self, toa, band):
        self.logger.info("EODP-ALG-ISM-3000: Video Chain")

        # The video chain receives the signal in electrons from Detection.
        alt, act = toa.shape[0] // 2, toa.shape[1] // 2
        electrons_read = float(toa[alt, act])

        # Electrons to Voltage - read-out and amplification (ATBD 7.3.1).
        self.logger.info("EODP-ALG-ISM-3010: Electrons to Voltage - Read-out and Amplification")
        voltage = self.electr2Volt(toa, self.ismConfig.OCF, self.ismConfig.ADC_gain)
        voltage_read = float(voltage[alt, act])
        self.logger.debug("TOA [0,0] " + str(voltage[0, 0]) + " [V]")

        # Voltage to Digital Numbers (ATBD 7.3.2).
        self.logger.info("EODP-ALG-ISM-3020: Voltage to Digital Numbers - Digitisation")
        toa_dn = self.digitisation(voltage,
                                   self.ismConfig.bit_depth,
                                   self.ismConfig.min_voltage,
                                   self.ismConfig.max_voltage)
        dn_read = float(toa_dn[alt, act])
        self.logger.debug("TOA [0,0] " + str(toa_dn[0, 0]) + " [DN]")

        # Conversion summary for the central pixel. Only stages available
        # in this module are printed; upstream conversions belong to
        # opticalPhase and detectionPhase.
        factor_ev = float(self.ismConfig.OCF) * float(self.ismConfig.ADC_gain)
        voltage_range = float(self.ismConfig.max_voltage) - float(self.ismConfig.min_voltage)
        factor_vdn = 4095.0 / voltage_range
        voltage_analytic = electrons_read * factor_ev
        dn_analytic_before_limit = float(np.round(voltage_read * factor_vdn))
        dn_analytic_after_limit = float(np.clip(dn_analytic_before_limit, 0, 4095))

        print(f"\nVCU conversion table - {band} - pixel [ALT={alt}, ACT={act}]")
        print(f"{'TOA stage':<28} {'Conv. factor':<24} {'TOA analytic':>18} {'TOA read':>18}  Unit")
        print("-" * 105)
        print(f"{'Electrons (input)':<28} {'--':<24} {'--':>18} {electrons_read:>18.9g}  e-")
        print(f"{'Electrons to Voltage':<28} {factor_ev:<24.12g} {voltage_analytic:>18.9g} {voltage_read:>18.9g}  V")
        print(f"{'Voltage to DN (raw)':<28} {factor_vdn:<24.12g} {dn_analytic_before_limit:>18.9g} {'--':>18}  DN")
        print(f"{'DN after ADC saturation':<28} {'0 ... 4095 DN':<24} {dn_analytic_after_limit:>18.9g} {dn_read:>18.9g}  DN")
        print(f"Conversion factor units: {factor_ev:.12g} V/e-; {factor_vdn:.12g} DN/V")
        print(f"ADC output limits: min={np.min(toa_dn):.0f} DN, max={np.max(toa_dn):.0f} DN\n")

        if self.ismConfig.save_vcu_stage:
            saveas_str = self.globalConfig.ism_toa_vcu + band
            title_str = 'TOA after the VCU phase [DN]'
            xlabel_str = 'ACT'
            ylabel_str = 'ALT'
            plotMat2D(toa_dn, title_str, xlabel_str, ylabel_str, self.outdir, saveas_str)

            idalt = int(toa_dn.shape[0] / 2)
            saveas_str = saveas_str + '_alt' + str(idalt)
            plotF([], toa_dn[idalt, :], title_str, xlabel_str, ylabel_str, self.outdir, saveas_str)

        return toa_dn

    def electr2Volt(self, toa, OCF, gain_adc):
        """Convert electrons [e-] to voltage [V], including amplification."""
        return toa * OCF * gain_adc

    def digitisation(self, toa, bit_depth, min_voltage, max_voltage):
        """Digitise a voltage array into 12-bit DN with saturation [0, 4095]."""
        if int(bit_depth) != 12:
            raise ValueError("The requested ADC saturation of 4095 DN requires bit_depth=12")
        if max_voltage <= min_voltage:
            raise ValueError("max_voltage must be greater than min_voltage")

        satval = 4095
        toa_dn = np.round(toa * satval / (max_voltage - min_voltage))
        toa_dn = np.clip(toa_dn, 0, satval)
        return toa_dn
