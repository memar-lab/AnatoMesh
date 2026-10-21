"""Rotational loading"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from engine.paths import safe_open_write

logger = logging.getLogger(__name__)

PLANE_AXES: dict[str, tuple[float, float, float]] = {
    "sagittal": (1.0, 0.0, 0.0),
    "coronal": (0.0, 1.0, 0.0),
    "axial": (0.0, 0.0, 1.0),
}
PLANE_DESCRIPTIONS: dict[str, str] = {
    "sagittal": "flexion/extension, rotation about the left-right axis",
    "coronal": "lateral bending, rotation about the anterior-posterior axis",
    "axial": "horizontal rotation, about the superior-inferior axis",
}
PULSE_SHAPES = ("half_sine", "trapezoid", "haversine", "triangular", "biphasic")

CURVE_ID_BASE = 100  # +1, +2, +3 for the X, Y, Z angular velocity curves
CURVE_ID_ACCEL_BASE = 200  # the matching angular acceleration curves
LOAD_NODE_SET_ID = 910  # node set the prescribed motion is applied to


@dataclass
class RotationSpec:
    """One rotational loading case."""

    planes: Sequence[str] = field(default_factory=lambda: ["sagittal"])
    peak_angular_velocity: Optional[float] = None
    peak_angular_acceleration: Optional[float] = None
    duration_ms: Optional[float] = None
    shape: str = "half_sine"
    n_samples: int = 200
    settle_ms: float = 20.0

    def __post_init__(self) -> None:
        bad = [p for p in self.planes if p not in PLANE_AXES]
        if bad:
            raise ValueError(f"unknown rotation plane(s) {bad}; choose from {sorted(PLANE_AXES)}")
        if not len(self.planes):
            raise ValueError("at least one rotation plane is required")
        if self.shape not in PULSE_SHAPES:
            raise ValueError(f"unknown pulse shape {self.shape!r}; choose from {PULSE_SHAPES}")
        for name in ("peak_angular_velocity", "peak_angular_acceleration", "duration_ms", "settle_ms"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")
        if self.n_samples < 2:
            raise ValueError("n_samples must be at least 2")

        given = [self.peak_angular_velocity, self.peak_angular_acceleration, self.duration_ms]
        if sum(v is not None and v > 0 for v in given) < 2:
            raise ValueError(
                "give at least two of peak_angular_velocity, peak_angular_acceleration "
                "and duration_ms -- the third is derived from them")
        if all(v is not None and v > 0 for v in given):
            expected = self.peak_angular_acceleration * self._shape_integral * self.duration_ms / 1000
            if not np.isclose(self.peak_angular_velocity, expected, rtol=1e-8, atol=0):
                raise ValueError("velocity, acceleration, and duration contradict the pulse shape")

    @property
    def axis(self) -> np.ndarray:
        """Unit rotation axis, in RAS world coordinates."""
        vec = np.sum([PLANE_AXES[p] for p in self.planes], axis=0)
        norm = np.linalg.norm(vec)
        if norm == 0:  # only reachable if PLANE_AXES ever gained opposing entries
            raise ValueError("the selected planes cancel to a zero rotation axis")
        return vec / norm

    @property
    def _shape_integral(self) -> float:
        """Mean-to-peak ratio of the pulse."""
        return {"half_sine": 2.0 / np.pi, "trapezoid": 0.8, "haversine": 0.5,
                "triangular": 0.5, "biphasic": 1.0 / np.pi}[self.shape]

    @property
    def duration(self) -> float:
        """Pulse duration in seconds, derived when not given outright."""
        if self.duration_ms:
            return self.duration_ms / 1000.0
        return self.peak_angular_velocity / (self.peak_angular_acceleration * self._shape_integral)

    @property
    def alpha(self) -> float:
        """Peak angular acceleration, rad/s^2."""
        if self.peak_angular_acceleration:
            return self.peak_angular_acceleration
        return self.peak_angular_velocity / (self._shape_integral * self.duration)

    @property
    def omega(self) -> float:
        """Peak angular velocity, rad/s."""
        if self.peak_angular_velocity:
            return self.peak_angular_velocity
        return self.alpha * self._shape_integral * self.duration

    @property
    def termination_time(self) -> float:
        """Solve end time, seconds"""
        return self.duration + self.settle_ms / 1000.0

    def sample(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample the pulse."""
        n_pulse = max(int(self.n_samples), 2)
        if self.shape in {"haversine", "triangular", "biphasic"}:
            from engine.pipeline.curves import Pulse

            sampled = Pulse(self.shape, self.alpha, self.duration).sample(
                max(n_pulse, 5), settle=self.settle_ms / 1000)
            return sampled["time_s"], sampled["value"], sampled["integral"]
        t_pulse = np.linspace(0.0, self.duration, n_pulse)
        if self.shape == "half_sine":
            a = self.alpha * np.sin(np.pi * t_pulse / self.duration)
        else:
            ramp = 0.2 * self.duration
            a = np.clip(np.minimum(t_pulse / ramp,
                                    (self.duration - t_pulse) / ramp), 0.0, 1.0) * self.alpha
        t_tail = (np.linspace(self.duration, self.termination_time, 8)[1:]
                  if self.settle_ms else np.empty(0))
        t = np.concatenate([t_pulse, t_tail])
        a = np.concatenate([a, np.zeros(len(t_tail))])
        w = np.concatenate([[0.0], np.cumsum(np.diff(t) * 0.5 * (a[1:] + a[:-1]))])
        return t, a, w

    def summary(self) -> dict:
        """Plain-dict description, for the run report and the web UI."""
        axis = self.axis
        return {
            "planes": list(self.planes),
            "axis_ras": [round(float(v), 6) for v in axis],
            "peak_angular_velocity_rad_s": round(float(self.omega), 4),
            "peak_angular_acceleration_rad_s2": round(float(self.alpha), 2),
            "pulse_shape": self.shape,
            "pulse_duration_ms": round(float(self.duration * 1000.0), 3),
            "termination_time_s": round(float(self.termination_time), 6),
        }


@dataclass
class PrescribedPulse:
    pulse: dict
    planes: Sequence[str] = field(default_factory=lambda: ["sagittal"])
    n_samples: int = 1001
    settle_ms: float = 20.0

    def __post_init__(self):
        from engine.pipeline.curves import Pulse

        self.profile = Pulse(**self.pulse)
        if not self.planes or any(plane not in PLANE_AXES for plane in self.planes):
            raise ValueError("select valid anatomical rotation planes")
        self.profile.sample(self.n_samples, self.settle_ms / 1000)

    @property
    def axis(self):
        axis = np.sum([PLANE_AXES[plane] for plane in self.planes], axis=0)
        return axis / np.linalg.norm(axis)

    @property
    def duration(self):
        return self.profile.duration

    @property
    def shape(self):
        return self.profile.shape

    @property
    def termination_time(self):
        return self.profile.onset + self.duration + self.settle_ms / 1000

    def sample(self):
        values = self.profile.sample(self.n_samples, self.settle_ms / 1000)
        if self.profile.quantity == "acceleration":
            return values["time_s"], values["value"], values["integral"]
        return values["time_s"], values["derivative"], values["value"]

    @property
    def alpha(self):
        return float(np.max(np.abs(self.sample()[1])))

    @property
    def omega(self):
        return float(np.max(np.abs(self.sample()[2])))

    def summary(self):
        return {"planes": list(self.planes), "axis_ras": self.axis.tolist(),
                "pulse": self.pulse, "termination_time_s": self.termination_time,
                "peak_angular_velocity_rad_s": self.omega,
                "peak_angular_acceleration_rad_s2": self.alpha}


def rotation_from_dict(parameters):
    return PrescribedPulse(**parameters) if "pulse" in parameters else RotationSpec(**parameters)


def _fmt_curve(curve_id: int, title: str, t: np.ndarray, v: np.ndarray) -> list[str]:
    """One *DEFINE_CURVE_TITLE block."""
    from engine.export.keyword_numbers import format_real
    lines = ["*DEFINE_CURVE_TITLE", title,
             f"{curve_id},0,1.0,1.0,0.0,0.0,0"]
    points = [(format_real(ti), format_real(vi)) for ti, vi in zip(t, v)]
    if any(float(b[0]) <= float(a[0]) for a, b in zip(points, points[1:])):
        raise ValueError('curve time points collide at keyword field precision')
    lines += [f"{ti},{vi}" for ti, vi in points]
    return lines


def format_curve_deck(spec: RotationSpec) -> str:
    """Format the *DEFINE_CURVE cards for a rotation spec."""
    t, a, w = spec.sample()
    axis = spec.axis
    lines: list[str] = [
        f"$# rotation planes: {', '.join(spec.planes)}",
        f"$# axis (RAS): {axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}",
        f"$# peak angular velocity {spec.omega:.4g} rad/s, "
        f"peak angular acceleration {spec.alpha:.6g} rad/s^2, "
        f"{spec.shape} pulse of {spec.duration * 1000.0:.4g} ms",
    ]
    for dof, name in enumerate("XYZ"):
        component = float(axis[dof])
        if abs(component) < 1e-12:
            continue
        lines += _fmt_curve(CURVE_ID_BASE + dof + 1,
                            f"angular velocity about {name} (rad/s)", t, w * component)
        lines += _fmt_curve(CURVE_ID_ACCEL_BASE + dof + 1,
                            f"angular acceleration about {name} (rad/s^2)", t, a * component)
    return "\n".join(lines)


def write_curve_file(out_path: Path, spec: RotationSpec) -> Path:
    """Write the standalone `curve.k` include file."""
    text = "\n".join(["*KEYWORD", format_curve_deck(spec), "*END", ""])
    with safe_open_write(out_path, "w") as f:
        f.write(text)
    logger.info("wrote %s (%s, %.4g rad/s peak, %.6g rad/s^2 peak)",
                out_path, "+".join(spec.planes), spec.omega, spec.alpha)
    return out_path


def format_prescribed_motion(spec: RotationSpec, node_ids: np.ndarray,
                             use_acceleration: bool = False) -> str:
    """Format the node set and *BOUNDARY_PRESCRIBED_MOTION_SET cards."""
    node_ids = np.asarray(node_ids)
    if not len(node_ids):
        return ""
    lines = ["*SET_NODE_LIST_TITLE", "prescribed rotation node set",
             f"{LOAD_NODE_SET_ID},0,0,0,0,0,0,0"]
    ids = (np.unique(node_ids) + 1).tolist()
    for i in range(0, len(ids), 8):
        lines.append(",".join(str(v) for v in ids[i:i + 8]))

    base = CURVE_ID_ACCEL_BASE if use_acceleration else CURVE_ID_BASE
    vad = 1 if use_acceleration else 0
    axis = spec.axis
    for dof, _name in enumerate("XYZ"):
        if abs(float(axis[dof])) < 1e-12:
            continue
        lines.append("*BOUNDARY_PRESCRIBED_MOTION_SET")
        lines.append(f"{LOAD_NODE_SET_ID},{5 + dof},{vad},{base + dof + 1},1.0,0,1e28,0.0")
    return "\n".join(lines)


def demo() -> None:
    """Self-check"""
    spec = RotationSpec(planes=["sagittal"], peak_angular_velocity=40.0,
                        peak_angular_acceleration=10000.0)
    t, a, w = spec.sample()
    assert abs(a.max() - 10000.0) < 1.0, a.max()
    assert abs(w.max() - 40.0) / 40.0 < 0.01, w.max()
    assert abs(spec.duration - np.pi * 40.0 / (2 * 10000.0)) < 1e-9
    assert np.all(np.diff(t) > 0), "curve abscissa must be strictly increasing"
    assert abs(w[-1] - w.max()) < 1e-9

    spec2 = RotationSpec(planes=["axial"], peak_angular_acceleration=8000.0, duration_ms=6.0)
    assert abs(spec2.omega - 8000.0 * (2 / np.pi) * 0.006) < 1e-9, spec2.omega

    spec3 = RotationSpec(planes=["sagittal", "coronal"], peak_angular_velocity=30.0,
                         duration_ms=5.0)
    assert np.allclose(spec3.axis, [2 ** -0.5, 2 ** -0.5, 0.0])
    deck = format_curve_deck(spec3)
    assert deck.count("*DEFINE_CURVE_TITLE") == 4, "two axes x (velocity, acceleration)"

    motion = format_prescribed_motion(spec3, np.arange(20))
    assert motion.count("*BOUNDARY_PRESCRIBED_MOTION_SET") == 2
    assert ",5,0," in motion and ",6,0," in motion, "rotational DOFs must be 5/6/7"

    for bad in (dict(planes=["oblique"], peak_angular_velocity=1.0, duration_ms=1.0),
                dict(planes=["axial"], peak_angular_velocity=1.0),
                dict(planes=["axial"], peak_angular_velocity=-1.0, duration_ms=1.0),
                dict(planes=["axial"], shape="ramp", peak_angular_velocity=1.0, duration_ms=1.0)):
        try:
            RotationSpec(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid spec: {bad}")
    print("loading.demo OK")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
