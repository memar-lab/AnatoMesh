"""LS-DYNA R12 MAT_061 serialization from explicit SI parameters."""

import math
from numbers import Integral, Real

from engine.export.keyword_numbers import long_card


def format_kelvin_maxwell(material_id, density_kg_m3, bulk_pa,
                          short_time_shear_pa, long_time_shear_pa, *,
                          maxwell_decay_per_s=None, kelvin_relaxation_s=None,
                          strain_output=0):
    """Return a solid-only MAT_061 card in tonne, mm, second units."""
    if isinstance(material_id, bool) or not isinstance(material_id, Integral) or not 1 <= material_id <= 99999999:
        raise ValueError('material ID must be a positive integer with at most eight digits')
    if isinstance(strain_output, bool) or not isinstance(strain_output, Integral) or strain_output not in (0, 1, 2):
        raise ValueError('strain_output must be 0, 1, or 2')
    if (maxwell_decay_per_s is None) == (kelvin_relaxation_s is None):
        raise ValueError('supply exactly one Maxwell decay rate or Kelvin relaxation time')
    formulation = int(kelvin_relaxation_s is not None)
    constant = kelvin_relaxation_s if formulation else maxwell_decay_per_s
    values = (density_kg_m3, bulk_pa, short_time_shear_pa, long_time_shear_pa, constant)
    if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) for value in values):
        raise ValueError('material parameters must be finite numbers')
    if density_kg_m3 <= 0 or bulk_pa <= 0 or short_time_shear_pa <= 0:
        raise ValueError('density, bulk modulus, and short-time shear modulus must be positive')
    if not 0 <= long_time_shear_pa <= short_time_shear_pa:
        raise ValueError('long-time shear modulus must lie between zero and the short-time modulus')
    if constant < 0 or (formulation and constant == 0):
        raise ValueError('Maxwell decay must be nonnegative; Kelvin relaxation time must be positive')
    row = (material_id, density_kg_m3 * 1e-12, bulk_pa * 1e-6,
           short_time_shear_pa * 1e-6, long_time_shear_pa * 1e-6,
           constant, formulation, strain_output)
    if any(value == 0 for value in row[1:4]):
        raise ValueError('converted material parameters underflow the output units')
    return '*MAT_KELVIN-MAXWELL_VISCOELASTIC +\n' + long_card(row)
