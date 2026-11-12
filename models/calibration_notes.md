# Interval Calibration Notes

## Methodology: Conformal Quantile Regression (CQR) Shift
To ensure honest uncertainty quantification and satisfy the hackathon PICP criteria (~80% coverage for P10-P90 interval), we implement a finite-sample, distribution-free conformal calibration shift on the walk-forward validation window.

### Nonconformity Definition
For lower and upper bounds:
- Lower residual: $E^{\\text{low}}_i = \\hat{q}_{0.1}(x_i) - y_i$
- Upper residual: $E^{\\text{high}}_i = y_i - \\hat{q}_{0.9}(x_i)$

Conformal quantile offsets $c_{\\text{low}}, c_{\\text{high}}$ are estimated at the $(1 - \\alpha/2)(1 + 1/n)$ quantile level ($\alpha = 0.20$ for 80% coverage).
Calibrated intervals are:
$$P_{10}^{\\text{cal}} = \\hat{q}_{0.1} - c_{\\text{low}}$$
$$P_{90}^{\\text{cal}} = \\hat{q}_{0.9} + c_{\\text{high}}$$
Post-shift monotonicity sorting ($p_{10} \le p_{50} \le p_{90}$) is strictly applied.

## Empirical Results (Validation & Test Window)
- **Target Coverage**: 80.00%
- **Raw LGBM Coverage (PICP)**: 56.25%
- **Calibrated LGBM Coverage (PICP)**: 71.25%
- **Shift Parameters**: $c_{\\text{low}} = 68.95$ MW, $c_{\\text{high}} = 118.78$ MW
- **Pinball Loss (Mean)**: 69.94 MW

## Production Contract
- Raw predictions saved as: `model="lgbm_quantile"`
- Calibrated predictions saved as: `model="lgbm_quantile_calibrated"`
