"""LS-DYNA R12 MAT_077_H with direct polynomial coefficients in SI units."""

import math
from numbers import Integral, Real

from engine.export.keyword_numbers import long_card

COEFFICIENTS = ('C10', 'C01', 'C11', 'C20', 'C02', 'C30')


def format_hyperelastic(material_id, density_kg_m3, poisson_ratio, coefficients_pa,
                        relaxation_moduli_pa=(), relaxation_rates_per_s=()):
    """Serialize solid material parameters"""
    if isinstance(material_id, bool) or not isinstance(material_id, Integral) or not 1 <= material_id <= 99999999:
        raise ValueError('material ID must be a positive integer with at most eight digits')
    if not isinstance(coefficients_pa, dict) or set(coefficients_pa) != set(COEFFICIENTS):
        raise ValueError('coefficients_pa must explicitly define C10, C01, C11, C20, C02, C30')
    coefficients = [coefficients_pa[key] for key in COEFFICIENTS]
    moduli, rates = tuple(relaxation_moduli_pa), tuple(relaxation_rates_per_s)
    if len(moduli) != len(rates) or len(moduli) > 12:
        raise ValueError('supply matching relaxation moduli and rates with at most twelve terms')
    values = [density_kg_m3, poisson_ratio, *coefficients, *moduli, *rates]
    if any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x) for x in values):
        raise ValueError('material parameters must be finite numbers')
    if density_kg_m3 <= 0 or not .49 < poisson_ratio < .5:
        raise ValueError('density must be positive and Poisson ratio must lie between 0.49 and 0.5')
    if coefficients[0] + coefficients[1] <= 0:
        raise ValueError('C10 + C01 must give positive infinitesimal shear stiffness')
    if any(x <= 0 for x in (*moduli, *rates)):
        raise ValueError('relaxation moduli and decay rates must be positive')
    converted = [density_kg_m3 * 1e-12, *(x * 1e-6 for x in coefficients + list(moduli))]
    if any(x != 0 and y == 0 for x, y in zip([density_kg_m3, *coefficients, *moduli], converted)):
        raise ValueError('material parameters underflow the output units')
    lines = ['*MAT_077_H +',
             long_card((material_id, converted[0], poisson_ratio, 0, 0, 0, 0, 0)),
             long_card((*converted[1:7], 0))]
    lines.extend(long_card((modulus * 1e-6, rate, 0, 0)) for modulus, rate in zip(moduli, rates))
    return '\n'.join(lines)
