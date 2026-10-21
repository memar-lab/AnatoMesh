"""Analytic loading pulses in seconds, radians, and SI angular derivatives."""

from dataclasses import dataclass

import numpy as np

SHAPES = ("half_sine", "haversine", "triangular", "biphasic", "hermite")


@dataclass(frozen=True)
class Pulse:
    shape: str
    peak: float
    duration: float
    rise: float | None = None
    decay: float | None = None
    onset: float = 0.0
    onset_jerk: float | None = None
    quantity: str = "acceleration"

    def __post_init__(self):
        if self.shape not in SHAPES or self.quantity not in {"acceleration", "velocity"}:
            raise ValueError("unknown pulse shape or quantity")
        for name in ("peak", "duration", "rise", "decay", "onset", "onset_jerk"):
            value = getattr(self, name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.duration <= 0 or self.onset < 0 or self.peak == 0:
            raise ValueError("duration must be positive, peak nonzero, and onset nonnegative")
        rise = self.rise
        if rise is None:
            rise = self.duration / 2 if self.decay is None else self.duration - self.decay
        decay = self.duration - rise if self.decay is None else self.decay
        if rise <= 0 or decay <= 0 or not np.isclose(rise + decay, self.duration, rtol=1e-12, atol=0):
            raise ValueError("positive rise and decay times must sum to duration")
        object.__setattr__(self, "rise", float(rise))
        object.__setattr__(self, "decay", float(decay))
        if self.onset_jerk is not None:
            if self.quantity != "acceleration":
                raise ValueError("onset_jerk is defined here for acceleration pulses only")
            if self.shape == "hermite":
                if not 0 <= self.onset_jerk * self.rise / self.peak <= 3:
                    raise ValueError("onset jerk would cause a Hermite rise overshoot")
            elif not np.isclose(self.onset_jerk, self.initial_slope, rtol=1e-10, atol=1e-12):
                raise ValueError("onset jerk conflicts with the specified analytic pulse")

    @property
    def initial_slope(self):
        if self.shape == "half_sine":
            return self.peak * np.pi / (2 * self.rise)
        if self.shape == "biphasic":
            return self.peak * np.pi / self.rise
        if self.shape == "triangular":
            return self.peak / self.rise
        if self.shape == "hermite":
            return 0.0 if self.onset_jerk is None else self.onset_jerk
        return 0.0

    def evaluate(self, time):
        """Return pulse value, derivative, and time integral at arbitrary times."""
        time = np.asarray(time, dtype=float)
        if not np.isfinite(time).all():
            raise ValueError("sample times must be finite")
        x = np.clip(time - self.onset, 0, self.duration)
        u = np.minimum(x / self.rise, 1)
        v = np.clip((x - self.rise) / self.decay, 0, 1)
        rising = x <= self.rise
        p, r, d = self.peak, self.rise, self.decay
        if self.shape == "half_sine":
            value = np.where(rising, p * np.sin(np.pi * u / 2), p * np.cos(np.pi * v / 2))
            derivative = np.where(rising, p * np.pi / (2 * r) * np.cos(np.pi * u / 2),
                                  -p * np.pi / (2 * d) * np.sin(np.pi * v / 2))
            integral = 2 * p * r / np.pi * (1 - np.cos(np.pi * u / 2)) + 2 * p * d / np.pi * np.sin(np.pi * v / 2)
        elif self.shape == "haversine":
            value = np.where(rising, p / 2 * (1 - np.cos(np.pi * u)), p / 2 * (1 + np.cos(np.pi * v)))
            derivative = np.where(rising, p * np.pi / (2 * r) * np.sin(np.pi * u),
                                  -p * np.pi / (2 * d) * np.sin(np.pi * v))
            integral = p * r / 2 * (u - np.sin(np.pi * u) / np.pi) + p * d / 2 * (v + np.sin(np.pi * v) / np.pi)
        elif self.shape == "triangular":
            value = np.where(rising, p * u, p * (1 - v))
            derivative = np.where(rising, p / r, -p / d)
            integral = p * r * u**2 / 2 + p * d * (v - v**2 / 2)
        elif self.shape == "biphasic":
            value = np.where(rising, p * np.sin(np.pi * u), -p * r / d * np.sin(np.pi * v))
            derivative = np.where(rising, p * np.pi / r * np.cos(np.pi * u),
                                  -p * r * np.pi / d**2 * np.cos(np.pi * v))
            integral = p * r / np.pi * (1 - np.cos(np.pi * u)) - p * r / np.pi * (1 - np.cos(np.pi * v))
        else:
            j = self.initial_slope * r / p
            value = np.where(rising, p * ((j - 2) * u**3 + (3 - 2 * j) * u**2 + j * u),
                             p * (1 - 3 * v**2 + 2 * v**3))
            derivative = np.where(rising, p / r * (3 * (j - 2) * u**2 + 2 * (3 - 2 * j) * u + j),
                                  p / d * (-6 * v + 6 * v**2))
            integral = p * r * ((j - 2) * u**4 / 4 + (3 - 2 * j) * u**3 / 3 + j * u**2 / 2)
            integral += p * d * (v - v**3 + v**4 / 2)
        outside = (time < self.onset) | (time > self.onset + self.duration)
        value = np.where(outside | (x == 0) | (x == self.duration), 0, value)
        derivative = np.where(outside, 0, derivative)
        return value, derivative, integral

    def sample(self, count=1001, settle=0.0):
        if not isinstance(count, (int, np.integer)) or count < 5:
            raise ValueError("at least five samples are required")
        if not np.isfinite(settle) or settle < 0:
            raise ValueError("settle must be finite and nonnegative")
        knots = [0, self.onset, self.onset + self.rise, self.onset + self.duration]
        if self.shape == "biphasic":
            knots += [self.onset + self.rise / 2, self.onset + self.rise + self.decay / 2]
        if settle:
            knots += [self.onset + self.duration + settle]
        time = np.linspace(self.onset, self.onset + self.duration, count)
        for knot in knots:
            closest = int(np.argmin(np.abs(time - knot)))
            tolerance = 8 * np.finfo(float).eps * max(abs(knot), self.duration)
            if abs(time[closest] - knot) <= tolerance:
                time[closest] = knot
            else:
                time = np.append(time, knot)
        time = np.unique(time)
        value, derivative, integral = self.evaluate(time)
        if not all(np.isfinite(values).all() for values in (time, value, derivative, integral)):
            raise ValueError("pulse parameters exceed finite floating-point range")
        return {"time_s": time, "value": value, "derivative": derivative, "integral": integral}
