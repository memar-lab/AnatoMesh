"""Finite LS-DYNA numbers bounded by the declared field width."""

import math


def format_real(value, width=20):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('keyword numbers must be finite')
    for precision in range(17, 0, -1):
        candidate = format(value, f'.{precision}g')
        if 'e' in candidate:
            mantissa, exponent = candidate.split('e')
            candidate = f'{mantissa}e{int(exponent)}'
        if len(candidate) <= width and math.isfinite(float(candidate)):
            return candidate
    raise ValueError(f'number cannot fit a {width}-character keyword field')


def long_card(values):
    values = tuple(values)
    if len(values) > 8:
        raise ValueError('material cards allow at most eight long fields')
    return ''.join(format_real(value).rjust(20) for value in values)
