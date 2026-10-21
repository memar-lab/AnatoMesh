"""Fail-closed topology and sampled Jacobian gate for solid hexahedra."""

import numpy as np

from engine.mesh.hex_geometry import element_volumes, jacobian_metrics, SIGNS
from engine.mesh.topology_gate import domain_topology
from engine.mesh.warpage import hex_warpage
from engine.mesh.jacobian_certificate import certify_hex_jacobians


class QualityGateError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__("Mesh failed topology, Jacobian, or shape acceptance criteria: "
                         f"topology={report['topology']['passed']}, "
                         f"min_SJ={report['minimum_scaled_jacobian']}, "
                         f"mean_SJ={report['mean_scaled_jacobian']}, "
                         f"aspect_failures={report['aspect_ratio_above_4']}, "
                         f"skew_failures={report['skew_above_050']}, "
                         f"warping_failures={report['warping_factor_above_040']}, "
                         f"nonpositive_certificates={report['nonpositive_jacobian_certificates']}, "
                         f"unresolved_certificates={report['unresolved_jacobian_certificates']}")


def inspect_hex_mesh(nodes, hexes, labels):
    topology = domain_topology(nodes, hexes, labels)
    determinants, scaled = jacobian_metrics(nodes, hexes)
    volumes = element_volumes(nodes, hexes)
    certificate = certify_hex_jacobians(nodes, hexes)
    warping, warping_angles = hex_warpage(nodes, hexes)
    vertices = np.asarray(nodes, dtype=float)[np.asarray(hexes)]
    center = np.einsum("eia,ib->eab", vertices - vertices[:, :1], SIGNS / 8)
    lengths = np.linalg.norm(center, axis=1)
    aspect = np.divide(lengths.max(axis=1), lengths.min(axis=1),
                       out=np.full(len(hexes), np.inf), where=lengths.min(axis=1) > 0)
    unit = np.divide(center, lengths[:, None, :], out=np.zeros_like(center),
                     where=lengths[:, None, :] > 0)
    skew = np.max(np.abs(np.stack([np.sum(unit[:, :, a] * unit[:, :, b], axis=1)
                                   for a, b in [(0, 1), (1, 2), (2, 0)]], axis=1)), axis=1)
    finite = bool(np.isfinite(determinants).all() and np.isfinite(scaled).all()
                  and np.isfinite(volumes).all())
    report = {"topology": topology, "element_count": len(hexes),
              "nonpositive_jacobians": int(np.count_nonzero(determinants <= 0)),
              "nonpositive_volumes": int(np.count_nonzero(volumes <= 0)),
              "scaled_jacobian_below_020": int(np.count_nonzero(scaled < .20)),
              "minimum_scaled_jacobian": float(scaled.min()) if len(scaled) else None,
              "mean_scaled_jacobian": float(scaled.mean()) if len(scaled) else None,
              "finite": finite,
              "nonpositive_jacobian_certificates": int(np.count_nonzero(certificate < 0)),
              "unresolved_jacobian_certificates": int(np.count_nonzero(certificate == 0)),
              "warping_factor_above_040": int(np.count_nonzero(warping > .4)),
              "maximum_warping_factor": float(warping.max()) if len(warping) and np.isfinite(warping).all() else None,
              "maximum_warping_angle_degrees": float(warping_angles.max()) if len(warping_angles) else None,
              "aspect_ratio_above_4": int(np.count_nonzero(aspect > 4)),
              "skew_above_050": int(np.count_nonzero(skew > .5)),
              "maximum_center_aspect_ratio": float(aspect.max()) if len(aspect) and np.isfinite(aspect).all() else None,
              "maximum_center_skew": float(skew.max()) if len(skew) else None,
              "sampling": "eight Gauss points, eight corners, and center"}
    report["passed"] = bool(topology["passed"] and finite and len(hexes)
                            and np.all(determinants > 0) and np.all(volumes > 0)
                            and np.all(scaled >= .20) and scaled.mean() >= .70
                            and np.all(aspect <= 4) and np.all(skew <= .5)
                            and np.all(warping <= .4) and np.all(certificate == 1))
    return report


def require_hex_quality(nodes, hexes, labels):
    report = inspect_hex_mesh(nodes, hexes, labels)
    if not report["passed"]:
        raise QualityGateError(report)
    return report
