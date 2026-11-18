"""Fixed-dimension right-prism geometry. All fitting is in physical mm.

P1/P2 and P3/P4 bound the hypotenuse face. The imaging face shares P1/P2.
The default depth equals face length (equal-leg right-prism assumption).
No scaling or reflections are permitted in the fitted rigid transform.
"""
from __future__ import annotations
import numpy as np

FACE_KEYS = ('shared_left_P1', 'shared_right_P2', 'derived_left', 'derived_right')


def fit_prism(clicked_mm, width_mm, length_mm, depth_mm=None):
    depth = length_mm if depth_mm is None else depth_mm
    dimensions = np.asarray([width_mm, length_mm, depth], dtype=float)
    if not np.all(np.isfinite(dimensions)) or np.any(dimensions <= 0):
        raise ValueError('Prism dimensions must be finite and positive.')
    points = np.asarray(clicked_mm, dtype=float)
    if points.shape != (4, 3) or not np.all(np.isfinite(points)):
        raise ValueError('Expected four finite XYZ points in P1, P2, P3, P4 order.')
    diagonal = np.hypot(length_mm, depth)
    model = np.array([[0, 0, 0], [width_mm, 0, 0],
                      [0, diagonal, 0], [width_mm, diagonal, 0]], dtype=float)
    center_model, center_points = model.mean(0), points.mean(0)
    covariance = (model - center_model).T @ (points - center_points)
    u, singular, vt = np.linalg.svd(covariance)
    if singular[0] <= 1e-12 or singular[1] <= singular[0] * 1e-8:
        raise ValueError('Clicks do not define two independent prism directions; recheck P1–P4.')
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    translation = center_points - rotation @ center_model
    fitted = model @ rotation.T + translation
    along = length_mm ** 2 / diagonal
    offset = length_mm * depth / diagonal
    candidates = {}
    for name, sign in [('A', 1), ('B', -1)]:
        face = np.array([[0, 0, 0], [width_mm, 0, 0],
                         [0, along, sign * offset], [width_mm, along, sign * offset]])
        candidates[name] = (face @ rotation.T + translation).tolist()
    errors = np.linalg.norm(fitted - points, axis=1)
    return dict(model='right_prism', width_mm=float(width_mm),
                face_length_mm=float(length_mm), depth_mm=float(depth),
                equal_leg_assumption=bool(length_mm == depth),
                rotation=rotation.tolist(), translation_mm=translation.tolist(),
                fitted_clicked_mm=fitted.tolist(), input_clicked_mm=points.tolist(),
                candidates_mm=candidates, per_corner_error_mm=errors.tolist(),
                rms_error_mm=float(np.sqrt(np.mean(errors ** 2))),
                max_error_mm=float(errors.max()))


def point_record(mm, spacing, kind='fitted'):
    mm = np.asarray(mm, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('XYZ spacing must be finite and positive.')
    voxel = mm / spacing
    return {**dict(zip(('x', 'y', 'z'), voxel.tolist())),
            **dict(zip(('x_mm', 'y_mm', 'z_mm'), mm.tolist())),
            'num_orientation_clicks': kind}


def face_records(face_mm, spacing):
    return {key: point_record(point, spacing) for key, point in zip(FACE_KEYS, face_mm)}


def clicked_mm_from_tracker(data):
    """Use unrounded per-view observations if available, else saved voxel points."""
    spacing = np.asarray(data.get('spacing_mm_xyz', [data.get('spacing_mm', .025)] * 3))
    result = []
    for pid in ('P1', 'P2', 'P3', 'P4'):
        observations = [v[pid] for v in data.get('per_orientation', {}).values()
                        if v.get(pid) is not None]
        if not observations:
            observations = [data.get('blue_clicked_points', data.get('green_points', {})).get(pid)]
        if not observations or any(p is None for p in observations):
            raise ValueError(f'Missing observations for {pid}.')
        result.append(np.mean([[p[k] for k in ('x', 'y', 'z')] for p in observations], axis=0))
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('Invalid source spacing.')
    return np.asarray(result) * spacing, spacing


def slice_intersection(polygon_xyz, axis, position, tolerance=1e-8):
    """Intersect a convex, ordered planar polygon with an axis-aligned slice."""
    vertices = np.asarray(polygon_xyz)
    points = []
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        da, db = a[axis] - position, b[axis] - position
        if abs(da) <= tolerance:
            points.append(a)
        if da * db < 0:
            points.append(a + da / (da - db) * (b - a))
    unique = []
    for p in points:
        if not any(np.linalg.norm(p - q) < tolerance for q in unique):
            unique.append(p)
    return np.asarray(unique).reshape(-1, 3)
