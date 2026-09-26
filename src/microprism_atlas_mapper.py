"""
Map a tracked microprism imaging face into the Allen CCF annotation volume.

Run after microprism_corner_tracker.py. The script:
1. loads the four imaging-face corners;
2. maps them into Allen CCF coordinates;
3. identifies the annotation region at each corner;
4. fits the corresponding oblique plane through the atlas;
5. renders the complete atlas plane with the prism face outlined.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.path import Path as MatplotlibPath
from matplotlib.patches import Polygon, Rectangle
from matplotlib.widgets import Button, Slider
import numpy as np

from allen_ccf_resources import (
    DEFAULT_RESOURCE_DIR,
    ensure_allen_ccf_resources,
    ensure_allen_template,
)
from project_paths import resolve_project_path


OUTSIDE_ID = 0
OUT_OF_BOUNDS_ID = -1
DEFAULT_SPACING_MM = 0.025
DEFAULT_SPACING_XYZ_MM = (DEFAULT_SPACING_MM, DEFAULT_SPACING_MM, DEFAULT_SPACING_MM)
SLICE_POINT_TOLERANCE_VOXELS = 0.75
PRISM_OVERLAY_COLOR = "#ff3030"
PRISM_FILL_COLOR = (1.0, 0.1, 0.1, 0.18)
PRISM_MARKER_COLOR = "#ff5a5a"
TWO_PHOTON_OVERLAY_ALPHA = 0.65
TWO_PHOTON_ZOOM_OVERLAY_ALPHA = 1.0
TWO_PHOTON_ZOOM_ALIGN_ALPHA = 0.55

# --- interface design tokens -------------------------------------------
# One palette and two font sizes, so every control is built the same way.
UI_BG = "#14161a"
UI_PANEL = "#1d222a"
UI_BANNER = "#242a33"
UI_SURFACE = "#2a2f38"
UI_SURFACE_HOVER = "#3a414d"
UI_ACTIVE = "#2f5a9e"
UI_ACTIVE_HOVER = "#3a6cbb"
UI_BORDER = "#454c59"
UI_TEXT = "#e6e9ef"
UI_TEXT_DARK = "#101216"
UI_MUTED = "#8b93a1"
UI_ACCENT_VIEW = "#5b9dff"
UI_FONT_SMALL = 8.6
UI_FONT_BODY = 10.0
TWO_PHOTON_BUTTON_COLOR = "#f0a84b"
TWO_PHOTON_BUTTON_HOVER_COLOR = "#ffc072"
TWO_PHOTON_MARKER_COLOR = "#00d5ff"
TWO_PHOTON_ZOOM_BUTTON_COLOR = "#5ecb8b"
TWO_PHOTON_ZOOM_BUTTON_HOVER_COLOR = "#8fe0b0"
TWO_PHOTON_ZOOM_OUTLINE_COLOR = "#5ecb8b"
UI_BUTTON_KINDS = {
    "default": (UI_SURFACE, UI_SURFACE_HOVER, UI_TEXT),
    "active": (UI_ACTIVE, UI_ACTIVE_HOVER, UI_TEXT),
    "twop": (
        TWO_PHOTON_BUTTON_COLOR, TWO_PHOTON_BUTTON_HOVER_COLOR,
        UI_TEXT_DARK,
    ),
    "zoom": (
        TWO_PHOTON_ZOOM_BUTTON_COLOR, TWO_PHOTON_ZOOM_BUTTON_HOVER_COLOR,
        UI_TEXT_DARK,
    ),
}
DEFAULT_ZOOM_MAGNIFICATION = 2.0
MAX_TWO_PHOTON_RENDER_PX = 2048
CONTROLS_HELP = (
    "Viewer controls\n"
    "  Views      In-plane/I, Coronal/C, Sagittal/S, Axial/A\n"
    "  Finish     press Q, or close the window; the summary then prints\n"
    "  Toggles    Colors, Labels and Fill light up blue when on; Fill is\n"
    "             the translucent red tint inside the prism outline\n"
    "  Navigate   drag or arrows to pan, scroll or +/- to zoom, "
    "0 fits the slice, R resets\n"
    "  Display    G atlas colours, L corner labels, F prism fill, "
    "Panel hides the region list\n"
    "  2p         Map 2p loads the unzoomed then the zoomed image; "
    "Layers shows the overlay toggles and opacity sliders\n"
    "  Aligning   Check scores the match against the image data and "
    "names a better offset if one exists\n"
    "             drag or arrows (Shift = 10 px), Mag changes the "
    "magnification, Enter confirms, R re-centres, Escape cancels\n"
    "  Corners    click D1, D2, P1, P2; Z undoes, P pans\n"
    "  MicroCT    uCT compares against the source volume; "
    "scroll the panel to change slice"
)


def style_button(button: Button, kind: str) -> None:
    """Re-skin an existing button, used to mark the active view."""
    face, hover, text_colour = UI_BUTTON_KINDS[kind]
    button.ax.set_facecolor(face)
    button.color = face
    button.hovercolor = hover
    button.label.set_color(text_colour)
    button.label.set_fontweight("bold" if kind != "default" else "normal")


def make_button(
    figure: Any,
    rect: tuple[float, float, float, float],
    label: str,
    kind: str = "default",
    fontsize: float = UI_FONT_SMALL,
) -> Button:
    """Build a toolbar button with the shared look.

    Every control in the viewer goes through here, so colours, borders and
    type sizes stay consistent instead of being respecified at each call site.
    """
    face, hover, text_colour = UI_BUTTON_KINDS[kind]
    button_axis = figure.add_axes(list(rect))
    button_axis.set_facecolor(face)
    button = Button(button_axis, label, color=face, hovercolor=hover)
    button.label.set_color(text_colour)
    button.label.set_fontsize(fontsize)
    button.label.set_fontweight("bold" if kind != "default" else "normal")
    for spine in button_axis.spines.values():
        spine.set_color(UI_BORDER)
        spine.set_linewidth(0.8)
    return button


def make_slider(
    figure: Any,
    rect: tuple[float, float, float, float],
    label: str,
    valinit: float,
    colour: str,
) -> Slider:
    """Build a 0..1 opacity slider with the shared look."""
    slider_axis = figure.add_axes(list(rect))
    slider = Slider(
        slider_axis, label, 0.0, 1.0,
        valinit=float(np.clip(valinit, 0.0, 1.0)), valstep=0.05, color=colour,
    )
    slider.label.set_position((0.0, 1.9))
    slider.label.set_horizontalalignment("left")
    slider.label.set_verticalalignment("bottom")
    slider.label.set_color(UI_MUTED)
    slider.label.set_fontsize(UI_FONT_SMALL - 0.6)
    slider.valtext.set_color(UI_TEXT)
    slider.valtext.set_fontsize(UI_FONT_SMALL - 0.6)
    return slider


def spacing_xyz(
    base: float | None,
    x: float | None,
    y: float | None,
    z: float | None,
    fallback: Iterable[float] = DEFAULT_SPACING_XYZ_MM,
) -> np.ndarray:
    """Return physical voxel size in coordinate order (X, Y, Z)."""
    fallback_values = tuple(float(value) for value in fallback)
    if len(fallback_values) != 3:
        raise ValueError("Spacing fallback must contain exactly three values.")
    if base is None:
        values = list(fallback_values)
    else:
        values = [float(base), float(base), float(base)]
    for index, override in enumerate((x, y, z)):
        if override is not None:
            values[index] = float(override)
    if any(value <= 0 for value in values):
        raise ValueError("All spacing values must be positive.")
    return np.asarray(values, dtype=float)


def tracker_spacing_xyz(tracker_data: dict[str, Any]) -> tuple[float, float, float]:
    values = tracker_data.get("spacing_mm_xyz")
    if isinstance(values, (list, tuple)) and len(values) == 3:
        return tuple(float(value) for value in values)
    scalar = float(tracker_data.get("spacing_mm", DEFAULT_SPACING_MM))
    return (scalar, scalar, scalar)


def load_registration_metrics(transform_path: Path | None) -> dict[str, Any]:
    """Load metadata saved next to a SimpleITK transform, if present."""
    if transform_path is None:
        return {}
    metrics_path = transform_path.with_name("registration_metrics.json")
    if not metrics_path.exists():
        return {}
    try:
        with metrics_path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError):
        return {}


def metadata_transform_direction(metrics: dict[str, Any]) -> str | None:
    """Return saved transform direction recorded by landmark registration."""
    direction = metrics.get("transform_saved_direction")
    if direction in {"fixed-to-moving", "moving-to-fixed"}:
        return str(direction)
    return None


def warn_transform_provenance(
    transform_path: Path | None,
    project: Path,
    metrics: dict[str, Any],
    source_spacing_xyz: np.ndarray,
) -> None:
    """Warn when the mapper appears to be using stale registration outputs."""
    if transform_path is None:
        return

    landmarks_path = project / "data" / "landmarks.json"
    if landmarks_path.exists():
        if transform_path.stat().st_mtime < landmarks_path.stat().st_mtime:
            print(
                "\nWARNING: registration transform is older than landmarks.json. "
                "The atlas mapper may be using a stale transform."
            )
            print(f"  Transform: {transform_path}")
            print(f"  Landmarks: {landmarks_path}")
        try:
            with landmarks_path.open("r", encoding="utf-8") as stream:
                landmark_data = json.load(stream)
            landmark_count = int(landmark_data.get("num_pairs", 0))
            metric_count = int(metrics.get("num_landmarks", 0))
            if metric_count and landmark_count and metric_count != landmark_count:
                print(
                    "\nWARNING: registration metrics and landmarks.json disagree "
                    "on landmark count."
                )
                print(
                    f"  metrics: {metric_count} landmark(s), "
                    f"landmarks.json: {landmark_count} landmark(s)"
                )
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    moving_spacing_um = metrics.get("moving_spacing_um")
    if moving_spacing_um is not None:
        metrics_spacing = np.array([float(moving_spacing_um) / 1000.0] * 3)
        if not np.allclose(metrics_spacing, source_spacing_xyz, rtol=0.02, atol=1e-6):
            print(
                "\nWARNING: mapper source spacing differs from registration "
                "moving spacing."
            )
            print(f"  registration spacing xyz: {metrics_spacing.tolist()} mm")
            print(f"  mapper source spacing xyz: {source_spacing_xyz.tolist()} mm")


@dataclass(frozen=True)
class RegionInfo:
    region_id: int
    acronym: str
    name: str
    color_hex: str

    @property
    def rgb(self) -> tuple[float, float, float]:
        value = self.color_hex.strip().lstrip("#")
        if len(value) != 6:
            value = "808080"
        return tuple(int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


@dataclass(frozen=True)
class PlaneGeometry:
    """An orthonormal 2D coordinate system embedded in atlas XYZ space."""

    origin_xyz: np.ndarray
    width_unit_xyz: np.ndarray
    length_unit_xyz: np.ndarray
    normal_unit_xyz: np.ndarray
    face_uv: dict[str, np.ndarray]
    fit_error_voxels: dict[str, float]

    def xyz_to_uv(self, points_xyz: np.ndarray) -> np.ndarray:
        relative = np.asarray(points_xyz, dtype=float) - self.origin_xyz
        return np.stack(
            (
                relative @ self.width_unit_xyz,
                relative @ self.length_unit_xyz,
            ),
            axis=-1,
        )

    def uv_to_xyz(self, points_uv: np.ndarray) -> np.ndarray:
        points_uv = np.asarray(points_uv, dtype=float)
        return (
            self.origin_xyz
            + points_uv[..., 0, None] * self.width_unit_xyz
            + points_uv[..., 1, None] * self.length_unit_xyz
        )


class AllenOntology:
    """Read Allen structure IDs, names, acronyms, and display colors."""

    def __init__(self, path: Path):
        self.regions: dict[int, RegionInfo] = {
            OUTSIDE_ID: RegionInfo(OUTSIDE_ID, "OUT", "Outside brain", "000000"),
            OUT_OF_BOUNDS_ID: RegionInfo(
                OUT_OF_BOUNDS_ID, "OOB", "Outside annotation volume", "000000"
            ),
        }
        with path.open("r", encoding="utf-8") as stream:
            self._parse(json.load(stream))

    def _parse(self, value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                self._parse(item)
            return
        if not isinstance(value, dict):
            return
        if "id" in value:
            region_id = int(value["id"])
            self.regions[region_id] = RegionInfo(
                region_id,
                str(value.get("acronym") or f"ID {region_id}"),
                str(value.get("name") or f"Region {region_id}"),
                str(value.get("color_hex_triplet") or "808080"),
            )
        for key in ("msg", "children"):
            if key in value:
                self._parse(value[key])

    def get(self, region_id: int) -> RegionInfo:
        return self.regions.get(
            region_id,
            RegionInfo(region_id, f"ID {region_id}", "Unknown region", "808080"),
        )


def first_existing(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def search_project(project: Path, names: tuple[str, ...]) -> Path | None:
    direct = [
        *(DEFAULT_RESOURCE_DIR / name for name in names),
        *(project / "outputs" / name for name in names),
        *(project / "data" / "ccf" / name for name in names),
        *(project / "data" / "processed" / name for name in names),
        *(project / name for name in names),
    ]
    found = first_existing(direct)
    if found:
        return found
    for name in names:
        matches = sorted(project.rglob(name))
        if matches:
            return matches[0].resolve()
    return None


def resolve_input(
    explicit: str | None,
    project: Path,
    names: tuple[str, ...],
    description: str,
    required: bool = True,
) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{description} not found: {path}")
        return path
    path = search_project(project, names)
    if required and path is None:
        raise FileNotFoundError(
            f"Could not find {description} ({', '.join(names)}) under {project}. "
            "Pass it explicitly on the command line."
        )
    return path


def load_annotation(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = np.load(path)
    elif suffix == ".nrrd":
        try:
            import nrrd
        except ImportError as exc:
            raise RuntimeError(
                "Reading NRRD files requires `pip install pynrrd`."
            ) from exc
        data, _ = nrrd.read(str(path))
    elif suffix in {".tif", ".tiff"}:
        try:
            import tifffile
        except ImportError as exc:
            raise RuntimeError("Reading TIFF files requires tifffile.") from exc
        data = tifffile.imread(path)
    else:
        raise ValueError(f"Unsupported annotation format: {path.suffix}")
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(f"Annotation must be 3D, got shape {data.shape}")
    return data


def load_volume(path: Path, description: str) -> np.ndarray:
    """Load a 3D image volume from NPY, NRRD, or TIFF."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = np.load(path)
    elif suffix == ".nrrd":
        try:
            import nrrd
        except ImportError as exc:
            raise RuntimeError(
                f"Reading {description} NRRD files requires `pip install pynrrd`."
            ) from exc
        data, _ = nrrd.read(str(path))
    elif suffix in {".tif", ".tiff"}:
        try:
            import tifffile
        except ImportError as exc:
            raise RuntimeError(f"Reading {description} TIFF files requires tifffile.") from exc
        data = tifffile.imread(path)
    else:
        raise ValueError(f"Unsupported {description} format: {path.suffix}")
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(f"{description} must be 3D, got shape {data.shape}")
    return data


def resolve_microct_for_comparison(
    explicit: str | None,
    tracker_data: dict[str, Any],
    project: Path,
) -> Path | None:
    """Find the microCT volume used by the corner tracker for side-by-side QC."""
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"MicroCT comparison image not found: {path}")
        return path

    source = tracker_data.get("source_image")
    if source:
        source_path = Path(str(source)).expanduser()
        candidates = [source_path]
        if not source_path.is_absolute():
            candidates.append(project / source_path)
        found = first_existing(candidates)
        if found:
            return found

    return search_project(
        project,
        (
            "microct_aligned_corrected.tif",
            "microct_aligned_corrected.tiff",
            "aligned_corrected.tif",
            "aligned_corrected.tiff",
            "microct_aligned.tif",
            "microct_aligned.tiff",
            "microct_registered.tif",
            "microct_registered.tiff",
        ),
    )


def point_xyz(point: dict[str, Any], label: str) -> np.ndarray:
    try:
        return np.array(
            [float(point["x"]), float(point["y"]), float(point["z"])],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid or missing coordinates for {label}") from exc


def load_face_corners(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        tracker_data = json.load(stream)
    face = tracker_data.get("imaging_face_points") or {}
    clicked = tracker_data.get("blue_clicked_points") or {}
    derived = tracker_data.get("derived_points") or {}
    values = {
        "bottom_left": face.get("shared_left_P1") or face.get("P1") or clicked.get("P1"),
        "bottom_right": face.get("shared_right_P2") or face.get("P2") or clicked.get("P2"),
        "top_left": face.get("derived_left") or derived.get("derived_left"),
        "top_right": face.get("derived_right") or derived.get("derived_right"),
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise ValueError(
            "microprism_corners.json does not contain a complete imaging face. "
            "Atlas mapper needs P1, P2, derived_left, and derived_right. "
            "Rerun microprism_corner_tracker.py and enter the prism dimensions "
            "so the two derived top points can be saved. "
            f"Missing: {', '.join(missing)}"
        )
    return (
        {name: point_xyz(value, name) for name, value in values.items()},
        tracker_data,
    )


def infer_coordinate_space(
    requested: str,
    tracker_data: dict[str, Any],
    annotation_shape: tuple[int, int, int],
) -> str:
    if requested != "auto":
        return requested
    source_name = Path(str(tracker_data.get("source_image", ""))).name.lower()
    source_shape = tuple(int(v) for v in tracker_data.get("image_shape_zyx", []))
    if "registered" in source_name or source_shape == annotation_shape:
        return "registered"
    return "aligned"


def transform_corners_to_ccf(
    corners: dict[str, np.ndarray],
    coordinate_space: str,
    transform_path: Path | None,
    source_spacing_xyz_mm: np.ndarray,
    atlas_spacing_xyz_mm: np.ndarray,
    saved_transform_maps: str,
    annotation_shape: tuple[int, int, int],
) -> tuple[dict[str, np.ndarray], str]:
    if coordinate_space == "registered":
        print("Using direct registered-image voxel mapping into CCF.")
        return ({name: value.copy() for name, value in corners.items()}, "registered")
    if transform_path is None:
        raise FileNotFoundError(
            "Aligned-space points require --transform. Use "
            "--coordinate-space registered only if tracking used "
            "microct_registered.tif."
        )
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError("Applying transforms requires SimpleITK.") from exc

    saved_transform = sitk.ReadTransform(str(transform_path))

    def apply(candidate) -> dict[str, np.ndarray]:
        mapped = {}
        for name, point in corners.items():
            source_mm = tuple((point * source_spacing_xyz_mm).tolist())
            ccf_mm = np.asarray(candidate.TransformPoint(source_mm), dtype=float)
            mapped[name] = ccf_mm / atlas_spacing_xyz_mm
        return mapped

    def bounds_score(mapped: dict[str, np.ndarray]) -> tuple[int, float]:
        nz, ny, nx = annotation_shape
        upper = np.array([nx - 1, ny - 1, nz - 1], dtype=float)
        inside = 0
        outside_distance = 0.0
        for point in mapped.values():
            is_inside = bool(np.all(point >= 0) and np.all(point <= upper))
            inside += int(is_inside)
            below = np.maximum(-point, 0.0)
            above = np.maximum(point - upper, 0.0)
            outside_distance += float(np.linalg.norm(below + above))
        return inside, -outside_distance

    direct = apply(saved_transform)
    if saved_transform_maps == "moving-to-fixed":
        print("Using saved transform directly: aligned microCT -> CCF.")
        return direct, "direct"

    try:
        inverse = apply(saved_transform.GetInverse())
    except RuntimeError:
        inverse = None

    if saved_transform_maps == "fixed-to-moving":
        if inverse is None:
            raise RuntimeError("The saved registration transform is not invertible.")
        print("Using inverse saved transform: aligned microCT -> CCF.")
        return inverse, "inverse"

    candidates = [("direct", direct)]
    if inverse is not None:
        candidates.append(("inverse", inverse))
    direction, selected = max(candidates, key=lambda item: bounds_score(item[1]))
    direct_score = bounds_score(direct)
    inverse_score = bounds_score(inverse) if inverse is not None else None
    print(
        "Transform direction auto-check: "
        f"direct={direct_score[0]}/4 corners in bounds, "
        f"inverse={inverse_score[0] if inverse_score else 0}/4."
    )
    print(f"Using {direction} transform mapping for aligned microCT -> CCF.")
    return selected, direction


def normalized(vector: np.ndarray, label: str) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < 1e-8:
        raise ValueError(f"Cannot define plane: {label} has near-zero length.")
    return vector / length


def build_plane_geometry(
    corners_xyz: dict[str, np.ndarray],
    target_length_to_width: float | None = None,
) -> PlaneGeometry:
    """Fit an orthonormal plane and its closest axis-aligned rectangle."""
    bl = corners_xyz["bottom_left"]
    br = corners_xyz["bottom_right"]
    tl = corners_xyz["top_left"]
    tr = corners_xyz["top_right"]
    origin = np.mean(np.stack((bl, br, tl, tr)), axis=0)

    width_hint = 0.5 * ((br - bl) + (tr - tl))
    width_unit = normalized(width_hint, "average width")
    length_hint = 0.5 * ((tl - bl) + (tr - br))
    length_orthogonal = (
        length_hint - np.dot(length_hint, width_unit) * width_unit
    )
    length_unit = normalized(length_orthogonal, "average length")
    if np.dot(length_unit, length_hint) < 0:
        length_unit = -length_unit
    normal_unit = normalized(np.cross(width_unit, length_unit), "normal")

    provisional = PlaneGeometry(
        origin, width_unit, length_unit, normal_unit, {}, {}
    )
    measured_uv = {
        name: provisional.xyz_to_uv(point)
        for name, point in corners_xyz.items()
    }
    measured_left_u = 0.5 * (
        measured_uv["bottom_left"][0] + measured_uv["top_left"][0]
    )
    measured_right_u = 0.5 * (
        measured_uv["bottom_right"][0] + measured_uv["top_right"][0]
    )
    measured_bottom_v = 0.5 * (
        measured_uv["bottom_left"][1] + measured_uv["bottom_right"][1]
    )
    measured_top_v = 0.5 * (
        measured_uv["top_left"][1] + measured_uv["top_right"][1]
    )
    center_u = 0.5 * (measured_left_u + measured_right_u)
    center_v = 0.5 * (measured_bottom_v + measured_top_v)
    width = abs(measured_right_u - measured_left_u)
    height = abs(measured_top_v - measured_bottom_v)
    if target_length_to_width is not None:
        ratio = float(target_length_to_width)
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError("Prism length/width ratio must be positive.")
        # Closest dimensions to the registered measurements subject to h=r*w.
        width = (width + ratio * height) / (1.0 + ratio * ratio)
        height = ratio * width
    left_u = center_u - width / 2.0
    right_u = center_u + width / 2.0
    bottom_v = center_v - height / 2.0
    top_v = center_v + height / 2.0
    face_uv = {
        "bottom_left": np.array((left_u, bottom_v)),
        "bottom_right": np.array((right_u, bottom_v)),
        "top_left": np.array((left_u, top_v)),
        "top_right": np.array((right_u, top_v)),
    }
    fit_error = {
        name: float(np.dot(point - origin, normal_unit))
        for name, point in corners_xyz.items()
    }
    return PlaneGeometry(
        origin, width_unit, length_unit, normal_unit, face_uv, fit_error
    )


def rectified_face_corners_xyz(
    geometry: PlaneGeometry,
) -> dict[str, np.ndarray]:
    """Return the closest rectangular face represented by the fitted plane."""
    return {
        key: geometry.uv_to_xyz(point_uv)
        for key, point_uv in geometry.face_uv.items()
    }


def measured_face_uv_from_corners(
    geometry: PlaneGeometry,
    corners_xyz: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Project the actual transformed corners into the fitted plane coordinates."""
    return {
        key: geometry.xyz_to_uv(point)
        for key, point in corners_xyz.items()
    }


def with_face_uv(
    geometry: PlaneGeometry,
    face_uv: dict[str, np.ndarray],
) -> PlaneGeometry:
    return PlaneGeometry(
        geometry.origin_xyz,
        geometry.width_unit_xyz,
        geometry.length_unit_xyz,
        geometry.normal_unit_xyz,
        face_uv,
        geometry.fit_error_voxels,
    )


def unique_points(points: list[np.ndarray], tolerance: float = 1e-6) -> list[np.ndarray]:
    unique = []
    for point in points:
        if not any(np.linalg.norm(point - existing) <= tolerance for existing in unique):
            unique.append(point)
    return unique


def face_slice_intersection_xyz(
    corners_xyz: dict[str, np.ndarray],
    face_order: tuple[str, ...],
    axis: int,
    slice_index: float,
    tolerance: float = 1e-6,
) -> list[np.ndarray]:
    """Return where the prism face intersects one x/y/z slice plane."""
    intersections: list[np.ndarray] = []
    ordered = [corners_xyz[key] for key in face_order]
    for start, end in zip(ordered, ordered[1:] + ordered[:1]):
        start_delta = float(start[axis] - slice_index)
        end_delta = float(end[axis] - slice_index)
        start_on = abs(start_delta) <= tolerance
        end_on = abs(end_delta) <= tolerance

        if start_on:
            intersections.append(start.copy())
        if start_on and end_on:
            intersections.append(end.copy())
        elif start_delta * end_delta < 0.0:
            fraction = start_delta / (start_delta - end_delta)
            intersections.append(start + fraction * (end - start))
        elif end_on:
            intersections.append(end.copy())

    return unique_points(intersections)


def project_xyz_to_view_mm(
    point_xyz: np.ndarray,
    mode_name: str,
    spacing_xyz_mm: np.ndarray,
) -> np.ndarray:
    sx, sy, sz = [float(value) for value in spacing_xyz_mm]
    x, y, z = [float(value) for value in point_xyz]
    if mode_name == "coronal":
        return np.array((x * sx, -y * sy), dtype=float)
    if mode_name == "sagittal":
        # anterior-posterior horizontal, dorsal up: the brain lies on its
        # side rather than standing on end.
        return np.array((z * sz, -y * sy), dtype=float)
    if mode_name == "axial":
        return np.array((z * sz, -x * sx), dtype=float)
    raise ValueError(f"Cannot project mode {mode_name!r}")


def project_xyz_list_to_view_mm(
    points_xyz: list[np.ndarray],
    mode_name: str,
    spacing_xyz_mm: np.ndarray,
) -> np.ndarray:
    if not points_xyz:
        return np.empty((0, 2), dtype=float)
    return np.stack([
        project_xyz_to_view_mm(point, mode_name, spacing_xyz_mm)
        for point in points_xyz
    ])


def near_slice_corner_keys(
    corners_xyz: dict[str, np.ndarray],
    face_order: tuple[str, ...],
    axis: int,
    slice_index: float,
    tolerance: float = SLICE_POINT_TOLERANCE_VOXELS,
) -> list[str]:
    return [
        key for key in face_order
        if abs(float(corners_xyz[key][axis] - slice_index)) <= tolerance
    ]


def angle_between_degrees(first: np.ndarray, second: np.ndarray) -> float:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm < 1e-10 or second_norm < 1e-10:
        return float("nan")
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def report_face_geometry(
    label: str,
    corners_xyz: dict[str, np.ndarray],
    spacing_xyz_mm: np.ndarray,
) -> None:
    """Print a compact sanity check for the prism face geometry."""
    p1 = corners_xyz["bottom_left"]
    p2 = corners_xyz["bottom_right"]
    dl = corners_xyz["top_left"]
    dr = corners_xyz["top_right"]

    p1_mm, p2_mm, dl_mm, dr_mm = [
        np.asarray(point, dtype=float) * spacing_xyz_mm
        for point in (p1, p2, dl, dr)
    ]
    bottom = p2_mm - p1_mm
    top = dr_mm - dl_mm
    left = dl_mm - p1_mm
    right = dr_mm - p2_mm
    diag_a = dr_mm - p1_mm
    diag_b = dl_mm - p2_mm

    normal = np.cross(bottom, left)
    normal_len = float(np.linalg.norm(normal))
    plane_error_mm = float("nan")
    if normal_len > 1e-10:
        normal_unit = normal / normal_len
        plane_error_mm = float(np.dot(dr_mm - p1_mm, normal_unit))

    sagittal_p1 = project_xyz_to_view_mm(p1, "sagittal", spacing_xyz_mm)
    sagittal_dl = project_xyz_to_view_mm(dl, "sagittal", spacing_xyz_mm)
    sagittal_side = sagittal_dl - sagittal_p1
    sagittal_angle_from_vertical = float("nan")
    if np.linalg.norm(sagittal_side) > 1e-10:
        sagittal_angle_from_vertical = float(np.degrees(np.arctan2(
            abs(sagittal_side[0]), abs(sagittal_side[1])
        )))

    print(f"\n{label} prism-face geometry check:")
    print(
        "  side lengths mm: "
        f"P1-P2={np.linalg.norm(bottom):.3f}, "
        f"dL-dR={np.linalg.norm(top):.3f}, "
        f"P1-dL={np.linalg.norm(left):.3f}, "
        f"P2-dR={np.linalg.norm(right):.3f}"
    )
    print(
        "  diagonals mm: "
        f"P1-dR={np.linalg.norm(diag_a):.3f}, "
        f"P2-dL={np.linalg.norm(diag_b):.3f}"
    )
    print(
        "  opposite-edge angle differences: "
        f"bottom_vs_top={angle_between_degrees(bottom, top):.2f} deg, "
        f"left_vs_right={angle_between_degrees(left, right):.2f} deg"
    )
    print(f"  fourth-corner plane error: {plane_error_mm:.4f} mm")
    print(
        "  sagittal P1-to-dL direction: "
        f"dY={sagittal_side[0]:.3f} mm, "
        f"dZ_display={sagittal_side[1]:.3f} mm, "
        f"{sagittal_angle_from_vertical:.1f} deg from vertical"
    )


def print_corner_coordinate_table(
    corners_source: dict[str, np.ndarray],
    corners_ccf: dict[str, np.ndarray],
    source_spacing_xyz_mm: np.ndarray,
    atlas_spacing_xyz_mm: np.ndarray,
) -> None:
    display_names = {
        "bottom_left": "P1",
        "bottom_right": "P2",
        "top_left": "D1",
        "top_right": "D2",
    }
    print("\nCorner coordinate trace:")
    print(
        "Point  Type     Source XYZ vox        Source XYZ mm         "
        "CCF XYZ vox           Coronal 2D mm      Sagittal 2D mm     Axial 2D mm"
    )
    print("-" * 132)
    for key in ("bottom_left", "bottom_right", "top_left", "top_right"):
        name = display_names[key]
        point_type = "clicked" if key in {"bottom_left", "bottom_right"} else "derived"
        source_vox = corners_source[key]
        source_mm = source_vox * source_spacing_xyz_mm
        ccf_vox = corners_ccf[key]
        coronal = project_xyz_to_view_mm(ccf_vox, "coronal", atlas_spacing_xyz_mm)
        sagittal = project_xyz_to_view_mm(ccf_vox, "sagittal", atlas_spacing_xyz_mm)
        axial = project_xyz_to_view_mm(ccf_vox, "axial", atlas_spacing_xyz_mm)
        print(
            f"{name:<5}  {point_type:<7}  "
            f"({source_vox[0]:7.2f},{source_vox[1]:7.2f},{source_vox[2]:7.2f})  "
            f"({source_mm[0]:6.3f},{source_mm[1]:6.3f},{source_mm[2]:6.3f})  "
            f"({ccf_vox[0]:7.2f},{ccf_vox[1]:7.2f},{ccf_vox[2]:7.2f})  "
            f"({coronal[0]:6.3f},{coronal[1]:7.3f})  "
            f"({sagittal[0]:6.3f},{sagittal[1]:7.3f})  "
            f"({axial[0]:6.3f},{axial[1]:7.3f})"
        )


def print_axis_projection_guide() -> None:
    """Document the standard viewer projection convention for debugging."""
    print("\nAxis-ordering / projection guide:")
    print("  Stored corner convention: XYZ voxel coordinates; volume arrays are Z,Y,X.")
    print("  Coronal view:  slice index = Z/AP; plotted axes = X horizontal, Y vertical.")
    print("  Sagittal view: slice index = X/ML; plotted axes = Y horizontal, Z vertical.")
    print("  Axial view:    slice index = Y/DV; plotted axes = X horizontal, Z vertical.")
    print("  Quick checks for future axis bugs:")
    print("    - P1-P2 should mainly change along the plotted width axis in each view.")
    print("    - P1-D1 and P2-D2 should project as nearly parallel edges.")
    print("    - Sagittal/axial views should collapse to a narrow edge when the plane is nearly perpendicular.")
    print("    - If the outline appears in the right shape but wrong view, suspect XYZ/ZYX or AP/DV/ML swaps.")
    print("    - If the shape is mirrored, suspect a sign flip or origin-direction mismatch.")


def set_axes_equal_3d(axis) -> None:
    limits = np.array([
        axis.get_xlim3d(),
        axis.get_ylim3d(),
        axis.get_zlim3d(),
    ], dtype=float)
    centers = limits.mean(axis=1)
    radius = 0.5 * float((limits[:, 1] - limits[:, 0]).max())
    axis.set_xlim3d(centers[0] - radius, centers[0] + radius)
    axis.set_ylim3d(centers[1] - radius, centers[1] + radius)
    axis.set_zlim3d(centers[2] - radius, centers[2] + radius)


def save_corner_diagnostic_3d(
    output_path: Path,
    corners_source: dict[str, np.ndarray],
    corners_ccf: dict[str, np.ndarray],
    source_spacing_xyz_mm: np.ndarray,
    atlas_spacing_xyz_mm: np.ndarray,
) -> None:
    order = ("bottom_left", "bottom_right", "top_right", "top_left")
    labels = {
        "bottom_left": "P1",
        "bottom_right": "P2",
        "top_left": "D1",
        "top_right": "D2",
    }

    figure = plt.figure(figsize=(10, 5), dpi=160)
    for index, (title, corners, spacing) in enumerate((
        ("Source microCT space (mm)", corners_source, source_spacing_xyz_mm),
        ("Allen CCF space (mm)", corners_ccf, atlas_spacing_xyz_mm),
    ), start=1):
        axis = figure.add_subplot(1, 2, index, projection="3d")
        points = {key: corners[key] * spacing for key in order}
        loop = np.stack([points[key] for key in order + (order[0],)])
        axis.plot(loop[:, 0], loop[:, 1], loop[:, 2], color="#1f77b4", linewidth=2)
        for key in order:
            point = points[key]
            axis.scatter(point[0], point[1], point[2], s=48, color="#ffcc33",
                         edgecolor="black", depthshade=False)
            axis.text(point[0], point[1], point[2], f" {labels[key]}",
                      fontsize=9, color="black")
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("X mm")
        axis.set_ylabel("Y mm")
        axis.set_zlabel("Z mm")
        set_axes_equal_3d(axis)
        axis.view_init(elev=22, azim=-58)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def atlas_box_vertices_xyz(shape_zyx: tuple[int, int, int]) -> np.ndarray:
    nz, ny, nx = shape_zyx
    return np.array([
        (x, y, z)
        for x in (0.0, float(nx - 1))
        for y in (0.0, float(ny - 1))
        for z in (0.0, float(nz - 1))
    ])


def plane_box_intersection_uv(
    geometry: PlaneGeometry,
    shape_zyx: tuple[int, int, int],
) -> np.ndarray:
    """Intersect the infinite prism plane with the atlas bounding box."""
    vertices = atlas_box_vertices_xyz(shape_zyx)
    edges = (
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
    )
    distances = (vertices - geometry.origin_xyz) @ geometry.normal_unit_xyz
    intersections = []
    for first_index, second_index in edges:
        first = vertices[first_index]
        second = vertices[second_index]
        first_distance = float(distances[first_index])
        second_distance = float(distances[second_index])
        if abs(first_distance) <= 1e-7:
            intersections.append(first)
        if abs(second_distance) <= 1e-7:
            intersections.append(second)
        if first_distance * second_distance < 0:
            fraction = first_distance / (first_distance - second_distance)
            intersections.append(first + fraction * (second - first))
    if len(intersections) < 3:
        raise ValueError("The fitted plane does not intersect the annotation volume.")
    unique = np.unique(np.round(np.stack(intersections), 7), axis=0)
    return geometry.xyz_to_uv(unique)


def full_plane_grid(
    geometry: PlaneGeometry,
    annotation_shape: tuple[int, int, int],
    max_samples: int,
    padding_fraction: float,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    intersection_uv = plane_box_intersection_uv(geometry, annotation_shape)
    minimum = intersection_uv.min(axis=0)
    maximum = intersection_uv.max(axis=0)
    size = maximum - minimum
    padding = np.maximum(size * padding_fraction, 1.0)
    minimum -= padding
    maximum += padding
    size = maximum - minimum

    if max_samples < 64:
        raise ValueError("--max-plane-samples must be at least 64.")
    step = max(float(size.max()) / max_samples, 1.0)
    width_samples = max(2, int(np.ceil(size[0] / step)) + 1)
    height_samples = max(2, int(np.ceil(size[1] / step)) + 1)
    u_grid, v_grid = np.meshgrid(
        np.linspace(minimum[0], maximum[0], width_samples),
        np.linspace(minimum[1], maximum[1], height_samples),
    )
    plane_uv = np.stack((u_grid, v_grid), axis=-1)
    extent_uv = (
        float(minimum[0]), float(maximum[0]),
        float(minimum[1]), float(maximum[1]),
    )
    return geometry.uv_to_xyz(plane_uv), extent_uv


def sample_annotation(annotation: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    rounded = np.rint(points_xyz).astype(np.int64)
    x, y, z = rounded[..., 0], rounded[..., 1], rounded[..., 2]
    valid = (
        (z >= 0) & (z < annotation.shape[0])
        & (y >= 0) & (y < annotation.shape[1])
        & (x >= 0) & (x < annotation.shape[2])
    )
    labels = np.full(x.shape, OUT_OF_BOUNDS_ID, dtype=np.int64)
    labels[valid] = annotation[z[valid], y[valid], x[valid]].astype(np.int64)
    return labels


def sample_template(template: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    """Trilinearly sample the anatomical template on an arbitrary plane."""
    try:
        from scipy.ndimage import map_coordinates
    except ImportError:
        rounded = np.rint(points_xyz).astype(np.int64)
        x, y, z = rounded[..., 0], rounded[..., 1], rounded[..., 2]
        valid = (
            (z >= 0) & (z < template.shape[0])
            & (y >= 0) & (y < template.shape[1])
            & (x >= 0) & (x < template.shape[2])
        )
        sampled = np.zeros(x.shape, dtype=float)
        sampled[valid] = template[z[valid], y[valid], x[valid]]
        return sampled
    coordinates = np.stack(
        (points_xyz[..., 2], points_xyz[..., 1], points_xyz[..., 0])
    )
    return map_coordinates(
        template.astype(float, copy=False),
        coordinates,
        order=1,
        mode="constant",
        cval=0.0,
    )


def as_display_image(image: np.ndarray) -> np.ndarray:
    """Return an RGB/RGBA floating image in 0..1 for display and warping."""
    display = np.asarray(image)
    if display.ndim == 2:
        display = np.stack((display, display, display), axis=-1)
    if display.ndim != 3 or display.shape[2] not in (3, 4):
        raise ValueError("2p image must be grayscale, RGB, or RGBA.")
    if np.issubdtype(display.dtype, np.integer):
        info = np.iinfo(display.dtype)
        display = display.astype(float) / float(info.max)
    else:
        display = display.astype(float, copy=False)
        finite = display[np.isfinite(display)]
        if finite.size and finite.max() > 1.0:
            display = display / 255.0
    return np.clip(display, 0.0, 1.0)


def load_two_photon_image(image_path: Path) -> np.ndarray:
    """Load a 2p JPG/PNG-like image through Matplotlib's image reader."""
    if not image_path.exists():
        raise FileNotFoundError(f"2p image not found: {image_path}")
    return as_display_image(plt.imread(image_path))


def describe_two_photon_image(image: np.ndarray) -> str:
    if image.ndim == 2:
        color_mode = "grayscale"
    elif image.ndim == 3 and image.shape[2] == 3:
        color_mode = "RGB"
    elif image.ndim == 3 and image.shape[2] == 4:
        color_mode = "RGBA"
    else:
        color_mode = f"{image.ndim}D"
    finite = image[np.isfinite(image)]
    if finite.size:
        value_range = f"range={float(finite.min()):.3f}..{float(finite.max()):.3f}"
    else:
        value_range = "range=empty/non-finite"
    return f"shape={image.shape}, dtype={image.dtype}, mode={color_mode}, {value_range}"


def choose_image_file_interactively(
    prompt: str = "Select 2p JPG/PNG image",
) -> Path | None:
    """Open a local image picker when possible; None if canceled/unavailable.

    ``prompt`` titles the dialog so the unzoomed and zoomed images cannot be
    confused with one another.
    """
    if sys.platform == "darwin":
        try:
            completed = subprocess.run(
                [
                    "osascript",
                    "-e",
                    f'POSIX path of (choose file with prompt "{prompt}" '
                    'of type {"public.jpeg", "public.png", "public.tiff", "public.image"})',
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            selected = completed.stdout.strip()
            if completed.returncode == 0 and selected:
                return Path(selected).expanduser().resolve()
            return None
        except OSError as error:
            print(f"macOS file picker unavailable: {error}")
            return None

    try:
        from tkinter import Tk, filedialog
    except ImportError:
        Tk = None
        filedialog = None
    if Tk is not None and filedialog is not None:
        try:
            root = Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            selected = filedialog.askopenfilename(
                title=prompt,
                filetypes=(
                    ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff"),
                    ("All files", "*.*"),
                ),
            )
            root.destroy()
            if selected:
                return Path(selected).expanduser().resolve()
        except Exception as error:
            print(f"Tk file picker unavailable, falling back: {error}")
    return None


def ask_number_interactively(
    question: str,
    default: float,
) -> float | None:
    """Prompt for a positive number in a native dialog, else the terminal."""
    if sys.platform == "darwin":
        script = (
            f'text returned of (display dialog "{question}" '
            f'default answer "{default:g}" with title "2p Mapper")'
        )
        try:
            completed = subprocess.run(
                ["osascript", "-e", script],
                check=False, capture_output=True, text=True,
            )
        except OSError as error:
            print(f"macOS dialog unavailable: {error}")
        else:
            if completed.returncode != 0:
                return None
            typed = completed.stdout.strip()
            if not typed:
                return default
            try:
                value = float(typed)
            except ValueError:
                print(f"'{typed}' is not a number; using {default:g}.")
                return default
            if value <= 0.0 or not np.isfinite(value):
                print(f"Magnification must be positive; using {default:g}.")
                return default
            return value

    while True:
        print(f"(answer in this terminal) {question} [{default:g}]")
        typed = input("> ").strip()
        if not typed:
            return default
        try:
            value = float(typed)
        except ValueError:
            print("  Enter a number, for example 2 for 2x.")
            continue
        if value <= 0.0 or not np.isfinite(value):
            print("  Magnification must be a positive, finite number.")
            continue
        return value


def polygon_signed_area(points_xy: np.ndarray) -> float:
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def solve_projective_transform(
    source_xy: np.ndarray,
    target_uv: np.ndarray,
) -> np.ndarray:
    """Return H where target homogeneous coordinates are proportional to H*source."""
    source_xy = np.asarray(source_xy, dtype=float)
    target_uv = np.asarray(target_uv, dtype=float)
    if source_xy.shape != (4, 2) or target_uv.shape != (4, 2):
        raise ValueError("Projective mapping requires exactly four 2D points.")

    rows = []
    values = []
    for (x, y), (u, v) in zip(source_xy, target_uv):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        values.append(u)
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        values.append(v)
    matrix = np.asarray(rows, dtype=float)
    vector = np.asarray(values, dtype=float)
    params, *_ = np.linalg.lstsq(matrix, vector, rcond=None)
    homography = np.array([
        [params[0], params[1], params[2]],
        [params[3], params[4], params[5]],
        [params[6], params[7], 1.0],
    ], dtype=float)
    if abs(float(np.linalg.det(homography))) < 1e-12:
        raise ValueError("2p corner mapping is singular; reselect non-collinear corners.")
    return homography


def apply_projective_transform(points_xy: np.ndarray, homography: np.ndarray) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=float)
    homogeneous = np.concatenate(
        (points_xy, np.ones((*points_xy.shape[:-1], 1), dtype=float)),
        axis=-1,
    )
    mapped = homogeneous @ homography.T
    scale = mapped[..., 2:3]
    return mapped[..., :2] / np.where(np.abs(scale) > 1e-12, scale, np.nan)


def warp_image_with_homography(
    image: np.ndarray,
    homography: np.ndarray,
    extent_uv_mm: tuple[float, float, float, float],
    output_shape: tuple[int, int],
    opacity: float,
    clip_polygon_uv_mm: np.ndarray | None = None,
) -> np.ndarray:
    """Warp an image onto the in-plane grid using a precomputed homography.

    The homography maps source image pixels to prism-plane millimetres. It may
    be fitted from four corner pairs, as for the unzoomed 2p image, or composed
    from other transforms, as for the zoomed 2p image, which is why this takes
    the matrix itself rather than point correspondences.
    """
    image = as_display_image(image)
    inverse = np.linalg.inv(homography)
    rows, columns = output_shape
    u_values = np.linspace(extent_uv_mm[0], extent_uv_mm[1], columns)
    v_values = np.linspace(extent_uv_mm[2], extent_uv_mm[3], rows)
    u_grid, v_grid = np.meshgrid(u_values, v_values)
    destination = np.stack((u_grid, v_grid), axis=-1)
    source_xy = apply_projective_transform(destination, inverse)

    height, width = image.shape[:2]
    valid = (
        (source_xy[..., 0] >= 0.0) & (source_xy[..., 0] <= width - 1)
        & (source_xy[..., 1] >= 0.0) & (source_xy[..., 1] <= height - 1)
        & np.isfinite(source_xy).all(axis=-1)
    )
    if clip_polygon_uv_mm is not None:
        inside_target = MatplotlibPath(
            np.asarray(clip_polygon_uv_mm, dtype=float)
        ).contains_points(
            destination.reshape(-1, 2), radius=1e-9
        ).reshape(rows, columns)
        valid = valid & inside_target
    warped = np.zeros((rows, columns, 4), dtype=float)

    try:
        from scipy.ndimage import map_coordinates
    except ImportError:
        nearest_x = np.clip(np.rint(source_xy[..., 0]).astype(int), 0, width - 1)
        nearest_y = np.clip(np.rint(source_xy[..., 1]).astype(int), 0, height - 1)
        sampled = image[nearest_y, nearest_x, :3]
    else:
        coordinates = np.stack((source_xy[..., 1], source_xy[..., 0]))
        sampled_channels = [
            map_coordinates(
                image[..., channel], coordinates, order=1,
                mode="constant", cval=0.0,
            )
            for channel in range(3)
        ]
        sampled = np.stack(sampled_channels, axis=-1)

    warped[..., :3] = sampled[..., :3]
    warped[..., 3] = np.where(valid, float(np.clip(opacity, 0.0, 1.0)), 0.0)
    return warped


def warp_two_photon_to_plane(
    image: np.ndarray,
    source_points_xy: np.ndarray,
    target_points_uv_mm: np.ndarray,
    extent_uv_mm: tuple[float, float, float, float],
    output_shape: tuple[int, int],
    opacity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a projective transform from four corner pairs, then warp with it."""
    target_points_uv_mm = np.asarray(target_points_uv_mm, dtype=float)
    homography = solve_projective_transform(source_points_xy, target_points_uv_mm)
    inverse = np.linalg.inv(homography)
    warped = warp_image_with_homography(
        image, homography, extent_uv_mm, output_shape, opacity,
        clip_polygon_uv_mm=target_points_uv_mm[[0, 1, 3, 2]],
    )
    return warped, homography, inverse


def score_zoom_alignment(
    wide_image: np.ndarray,
    zoom_image: np.ndarray,
    magnification: float,
    translation_wide_px: tuple[float, float],
    search_px: int = 24,
    coarse_pixels: int = 96,
) -> dict[str, Any]:
    """Grade a manual zoom alignment against the image data itself.

    The zoomed image is a magnified view of part of the unzoomed image, so when
    it is scaled down and placed correctly the two should show the same thing.
    This shrinks the zoom to the size it occupies in the unzoomed image and
    measures how well the two match, using Pearson correlation: +1 is a perfect
    linear match, 0 is unrelated.

    The same measurement is repeated over a grid of nearby offsets. If some
    other offset scores clearly better than the one chosen by hand, the manual
    placement is probably a few pixels out. Returns the user's score, the best
    nearby offset and its score, and the shift between them.
    """
    wide = as_display_image(wide_image)[..., :3].mean(axis=-1)
    zoom = as_display_image(zoom_image)[..., :3].mean(axis=-1)
    wide_height, wide_width = wide.shape[:2]
    zoom_height, zoom_width = zoom.shape[:2]
    display_width, display_height = zoom_display_size_px(
        magnification, (wide_height, wide_width)
    )
    translate_x, translate_y = (float(v) for v in translation_wide_px)

    step = max(1.0, max(display_width, display_height) / float(coarse_pixels))
    offsets_u = np.arange(0.0, max(display_width - 1.0, 1.0), step)
    offsets_v = np.arange(0.0, max(display_height - 1.0, 1.0), step)
    if offsets_u.size < 4 or offsets_v.size < 4:
        return {"available": False, "reason": "zoom footprint too small to score"}

    # matching sample positions inside the zoom itself
    zoom_columns = np.clip(
        np.rint(offsets_u / max(display_width - 1.0, 1e-9) * (zoom_width - 1)),
        0, zoom_width - 1,
    ).astype(int)
    zoom_rows = np.clip(
        np.rint(offsets_v / max(display_height - 1.0, 1e-9) * (zoom_height - 1)),
        0, zoom_height - 1,
    ).astype(int)
    zoom_patch = zoom[np.ix_(zoom_rows, zoom_columns)]
    zoom_centred = zoom_patch - zoom_patch.mean()
    zoom_norm = float(np.sqrt((zoom_centred ** 2).sum()))
    if zoom_norm <= 0.0:
        return {"available": False, "reason": "zoomed image has no contrast"}

    def correlation(shift_x: float, shift_y: float) -> float | None:
        columns = np.rint(translate_x + shift_x + offsets_u).astype(int)
        rows = np.rint(translate_y + shift_y + offsets_v).astype(int)
        if (
            columns.min() < 0 or columns.max() > wide_width - 1
            or rows.min() < 0 or rows.max() > wide_height - 1
        ):
            return None
        patch = wide[np.ix_(rows, columns)]
        centred = patch - patch.mean()
        norm = float(np.sqrt((centred ** 2).sum()))
        if norm <= 0.0:
            return None
        return float((centred * zoom_centred).sum() / (norm * zoom_norm))

    chosen = correlation(0.0, 0.0)
    best_score, best_shift = chosen, (0.0, 0.0)
    for shift_y in range(-search_px, search_px + 1):
        for shift_x in range(-search_px, search_px + 1):
            value = correlation(float(shift_x), float(shift_y))
            if value is not None and (best_score is None or value > best_score):
                best_score, best_shift = value, (float(shift_x), float(shift_y))

    if chosen is None or best_score is None:
        return {"available": False, "reason": "zoom falls outside the unzoomed image"}
    distance = float(np.hypot(*best_shift))
    return {
        "available": True,
        "correlation_at_chosen_offset": chosen,
        "best_nearby_correlation": best_score,
        "best_nearby_shift_px": {"x": best_shift[0], "y": best_shift[1]},
        "shift_distance_px": distance,
        "search_radius_px": int(search_px),
        "samples": int(zoom_patch.size),
        "suspicious": bool(distance > 2.0 and best_score - chosen > 0.02),
    }


def plane_render_grid(
    footprint_uv_mm: np.ndarray,
    source_shape: tuple[int, int],
    render_scale: float = 1.0,
    max_pixels: int = MAX_TWO_PHOTON_RENDER_PX,
    padding_fraction: float = 0.01,
) -> tuple[tuple[float, float, float, float], tuple[int, int], float]:
    """Choose an output extent and grid size that preserve the source detail.

    ``footprint_uv_mm`` holds the prism-plane millimetre positions of the source
    image's four corners, ordered top-left, top-right, bottom-left,
    bottom-right. The returned grid spans their bounding box at a pixel pitch
    matched to the source image.

    This exists because the atlas sampling grid is deliberately floored at one
    annotation voxel (25 microns) per pixel, which is right for atlas labels
    and badly wrong for a photograph: a 512-pixel 2p image covering a 1.5 mm
    prism face would be crushed to roughly 60 pixels. The 2p layers are drawn
    with their own extent, so they can be sampled finely without changing the
    atlas arrays at all.

    Returns the extent in millimetres, the (rows, columns) grid shape, and the
    resulting pixel pitch in millimetres.
    """
    footprint = np.asarray(footprint_uv_mm, dtype=float)
    if footprint.shape != (4, 2) or not np.isfinite(footprint).all():
        raise ValueError("A render-grid footprint needs four finite corners.")
    top_left, top_right, bottom_left, bottom_right = footprint
    width_mm = 0.5 * (
        float(np.linalg.norm(top_right - top_left))
        + float(np.linalg.norm(bottom_right - bottom_left))
    )
    height_mm = 0.5 * (
        float(np.linalg.norm(bottom_left - top_left))
        + float(np.linalg.norm(bottom_right - top_right))
    )
    source_height, source_width = int(source_shape[0]), int(source_shape[1])

    pitches = []
    if source_width > 1 and width_mm > 0.0:
        pitches.append(width_mm / (source_width - 1))
    if source_height > 1 and height_mm > 0.0:
        pitches.append(height_mm / (source_height - 1))
    if not pitches:
        raise ValueError("Cannot size a render grid for a degenerate footprint.")
    pitch = min(pitches) / max(float(render_scale), 1e-6)

    minimum = footprint.min(axis=0)
    maximum = footprint.max(axis=0)
    padding = np.maximum((maximum - minimum) * padding_fraction, pitch)
    minimum = minimum - padding
    maximum = maximum + padding
    span = maximum - minimum

    columns = int(np.clip(np.ceil(span[0] / pitch) + 1, 2, max_pixels))
    rows = int(np.clip(np.ceil(span[1] / pitch) + 1, 2, max_pixels))
    # honour the cap by coarsening rather than cropping
    pitch = max(span[0] / max(columns - 1, 1), span[1] / max(rows - 1, 1))
    extent = (
        float(minimum[0]), float(maximum[0]),
        float(minimum[1]), float(maximum[1]),
    )
    return extent, (rows, columns), float(pitch)


def print_two_photon_mapping_diagnostics(
    source_points_xy: np.ndarray,
    target_points_uv_mm: np.ndarray,
    homography: np.ndarray,
) -> None:
    labels = ("D1/top-left", "D2/top-right", "P1/bottom-left", "P2/bottom-right")
    mapped = apply_projective_transform(source_points_xy, homography)
    source_order = source_points_xy[[0, 1, 3, 2]]
    target_order = target_points_uv_mm[[0, 1, 3, 2]]
    source_area = polygon_signed_area(source_order)
    target_area = polygon_signed_area(target_order)
    print("\n2p -> prism-plane mapping diagnostics:")
    print("  Corner order: 2p TL->D1, TR->D2, BL->P1, BR->P2")
    print("  Projective transform matrix H, mapping 2p pixels -> prism-plane mm:")
    for row in homography:
        print("    " + " ".join(f"{value: .8g}" for value in row))
    print(f"  Source orientation signed area: {source_area:.3f} px^2")
    print(f"  Target orientation signed area: {target_area:.6f} mm^2")
    if source_area * target_area < 0:
        print(
            "  WARNING: mapping includes a mirror flip. This may be correct if "
            "the microscope/camera mirrors the image, but verify biology before "
            "using cell locations."
        )
    print("  Selected source pixel -> mapped prism-plane mm -> target prism-plane mm")
    for label, source, actual, target in zip(
        labels, source_points_xy, mapped, target_points_uv_mm
    ):
        error = float(np.linalg.norm(actual - target))
        print(
            f"    {label:<16} "
            f"({source[0]:8.2f}, {source[1]:8.2f}) -> "
            f"({actual[0]:8.4f}, {actual[1]:8.4f}) vs "
            f"({target[0]:8.4f}, {target[1]:8.4f}); error={error:.6f} mm"
        )


def zoom_to_wide_matrix(
    magnification: float,
    translation_wide_px: tuple[float, float],
    zoom_shape: tuple[int, int],
    wide_shape: tuple[int, int],
) -> np.ndarray:
    """Return the 3x3 mapping zoomed 2p pixels onto unzoomed (wide) 2p pixels.

    The zoomed image is acquired at ``magnification`` times the wide image's
    magnification, so its field of view spans ``wide_width / magnification``
    wide pixels regardless of how many pixels the zoom itself was sampled at.
    Scale and translation only: both frames share a scan angle, so there is no
    rotation term.

    Coordinates are 0-based, origin at the top-left, y increasing downward,
    matching numpy and Matplotlib's image convention.
    """
    magnification = float(magnification)
    if not np.isfinite(magnification) or magnification <= 0.0:
        raise ValueError("2p magnification must be a positive, finite number.")
    zoom_height, zoom_width = int(zoom_shape[0]), int(zoom_shape[1])
    wide_height, wide_width = int(wide_shape[0]), int(wide_shape[1])
    if min(zoom_height, zoom_width, wide_height, wide_width) < 2:
        raise ValueError("2p images must be at least 2x2 pixels.")
    scale_x = (wide_width / magnification) / zoom_width
    scale_y = (wide_height / magnification) / zoom_height
    translate_x, translate_y = (float(value) for value in translation_wide_px)
    return np.array([
        [scale_x, 0.0, translate_x],
        [0.0, scale_y, translate_y],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def zoom_display_size_px(
    magnification: float,
    wide_shape: tuple[int, int],
) -> tuple[float, float]:
    """Return (width, height) that the scaled zoom occupies in wide pixels."""
    magnification = float(magnification)
    if not np.isfinite(magnification) or magnification <= 0.0:
        raise ValueError("2p magnification must be a positive, finite number.")
    wide_height, wide_width = int(wide_shape[0]), int(wide_shape[1])
    return wide_width / magnification, wide_height / magnification


def compose_zoom_to_plane(
    wide_to_plane: np.ndarray,
    zoom_to_wide: np.ndarray,
) -> np.ndarray:
    """Chain zoom to wide to prism plane into a single matrix.

    Matrix multiplication is transform chaining, so one matrix multiply
    replaces running a point through two separate steps.
    """
    return (
        np.asarray(wide_to_plane, dtype=float)
        @ np.asarray(zoom_to_wide, dtype=float)
    )


def write_two_photon_alignment(
    path: Path,
    wide_image_path: Path | None,
    zoom_image_path: Path | None,
    magnification: float,
    translation_wide_px: tuple[float, float],
    zoom_shape: tuple[int, int],
    wide_shape: tuple[int, int],
    zoom_to_wide: np.ndarray,
    wide_to_plane: np.ndarray | None = None,
    zoom_to_plane: np.ndarray | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist the zoom alignment so it never has to be redone by hand."""
    zoom_to_wide = np.asarray(zoom_to_wide, dtype=float)
    translate_x, translate_y = (float(value) for value in translation_wide_px)
    display_width, display_height = zoom_display_size_px(magnification, wide_shape)
    payload: dict[str, Any] = {
        "wide_image": str(wide_image_path) if wide_image_path else None,
        "zoom_image": str(zoom_image_path) if zoom_image_path else None,
        "magnification": float(magnification),
        "scale_factor_x": float(zoom_to_wide[0, 0]),
        "scale_factor_y": float(zoom_to_wide[1, 1]),
        "wide_shape_rows_columns": [int(wide_shape[0]), int(wide_shape[1])],
        "zoom_shape_rows_columns": [int(zoom_shape[0]), int(zoom_shape[1])],
        "zoom_display_size_wide_px": [display_width, display_height],
        "translation_wide_px_0based": {"x": translate_x, "y": translate_y},
        "translation_wide_px_1based": {
            "x": translate_x + 1.0,
            "y": translate_y + 1.0,
        },
        "zoom_to_wide_matrix": zoom_to_wide.tolist(),
        "coordinate_convention": (
            "Matrices operate on 0-based pixel coordinates with the origin at "
            "the top-left of the image and y increasing downward. Translation "
            "is the wide-image pixel coordinate of the scaled zoom's top-left "
            "corner. The *_1based fields repeat it in the 1..N convention used "
            "by ImageJ and MATLAB."
        ),
        "transform_chain": (
            "zoom px -> wide px (scale + translate) -> prism-plane mm "
            "(projective) -> Allen CCF XYZ voxels (PlaneGeometry.uv_to_xyz)"
        ),
    }
    if wide_to_plane is not None:
        payload["wide_to_plane_mm_matrix"] = np.asarray(wide_to_plane).tolist()
    if zoom_to_plane is not None:
        payload["zoom_to_plane_mm_matrix"] = np.asarray(zoom_to_plane).tolist()
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    return path


def annotation_id_at_point(annotation: np.ndarray, point_xyz: np.ndarray) -> int:
    """Match Pinpoint: round XYZ to the nearest annotation voxel."""
    x, y, z = np.rint(point_xyz).astype(np.int64)
    if (
        z < 0 or z >= annotation.shape[0]
        or y < 0 or y >= annotation.shape[1]
        or x < 0 or x >= annotation.shape[2]
    ):
        return OUT_OF_BOUNDS_ID
    return int(annotation[z, y, x])


def count_points_in_bounds(
    points_xyz: dict[str, np.ndarray],
    shape_zyx: tuple[int, int, int],
) -> int:
    """Count XYZ points that fall inside a Z/Y/X array volume."""
    nz, ny, nx = shape_zyx
    upper = np.array([nx - 1, ny - 1, nz - 1], dtype=float)
    return sum(
        int(np.all(point >= 0) and np.all(point <= upper))
        for point in points_xyz.values()
    )


def corner_region_rows(
    annotation: np.ndarray,
    corners_ccf: dict[str, np.ndarray],
    ontology: AllenOntology,
) -> list[dict[str, Any]]:
    display_names = {
        "bottom_left": "P1",
        "bottom_right": "P2",
        "top_left": "D1",
        "top_right": "D2",
    }
    rows = []
    for key in ("bottom_left", "bottom_right", "top_left", "top_right"):
        point = corners_ccf[key]
        region_id = annotation_id_at_point(annotation, point)
        info = ontology.get(region_id)
        rows.append({
            "corner": display_names[key],
            "corner_key": key,
            "x_ccf": float(point[0]),
            "y_ccf": float(point[1]),
            "z_ccf": float(point[2]),
            "region_id": region_id,
            "acronym": info.acronym,
            "name": info.name,
            "color_hex": info.color_hex,
        })
    return rows


def face_mask_for_grid(
    geometry: PlaneGeometry,
    plane_xyz: np.ndarray,
) -> np.ndarray:
    plane_uv = geometry.xyz_to_uv(plane_xyz)
    polygon_uv = np.stack([
        geometry.face_uv["bottom_left"],
        geometry.face_uv["bottom_right"],
        geometry.face_uv["top_right"],
        geometry.face_uv["top_left"],
    ])
    path = MatplotlibPath(np.vstack((polygon_uv, polygon_uv[0])), closed=True)
    return path.contains_points(
        plane_uv.reshape(-1, 2), radius=1e-9
    ).reshape(plane_uv.shape[:2])


def region_summary(labels: np.ndarray, ontology: AllenOntology) -> list[dict[str, Any]]:
    counts = Counter(int(value) for value in labels.ravel())
    total = int(labels.size)
    inside = total - counts.get(OUTSIDE_ID, 0) - counts.get(OUT_OF_BOUNDS_ID, 0)
    rows = []
    for region_id, count in counts.most_common():
        info = ontology.get(region_id)
        rows.append({
            "region_id": region_id,
            "acronym": info.acronym,
            "name": info.name,
            "color_hex": info.color_hex,
            "pixel_count": count,
            "percent_of_plane": 100.0 * count / total,
            "percent_of_brain_area": (
                100.0 * count / inside if inside > 0 and region_id > 0 else 0.0
            ),
        })
    return rows


def largest_component_center(mask: np.ndarray) -> tuple[float, float, int] | None:
    if not np.any(mask):
        return None
    visited = np.zeros(mask.shape, dtype=bool)
    best: list[tuple[int, int]] = []
    height, width = mask.shape
    for start_y, start_x in zip(*np.nonzero(mask)):
        if visited[start_y, start_x]:
            continue
        queue = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        component = []
        while queue:
            y, x = queue.popleft()
            component.append((y, x))
            for next_y, next_x in (
                (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)
            ):
                if (
                    0 <= next_y < height and 0 <= next_x < width
                    and mask[next_y, next_x] and not visited[next_y, next_x]
                ):
                    visited[next_y, next_x] = True
                    queue.append((next_y, next_x))
        if len(component) > len(best):
            best = component
    ys = np.fromiter((point[0] for point in best), dtype=float)
    xs = np.fromiter((point[1] for point in best), dtype=float)
    return float(xs.mean()), float(ys.mean()), len(best)


def render_plane_map(
    annotation: np.ndarray,
    labels: np.ndarray,
    template: np.ndarray | None,
    plane_template: np.ndarray | None,
    microct_image: np.ndarray | None,
    microct_path: Path | None,
    ontology: AllenOntology,
    corner_rows: list[dict[str, Any]],
    face_summary: list[dict[str, Any]],
    corners_source: dict[str, np.ndarray],
    display_corners_ccf: dict[str, np.ndarray],
    geometry: PlaneGeometry,
    extent_uv: tuple[float, float, float, float],
    atlas_spacing_mm: float,
    source_spacing_xyz_mm: np.ndarray,
    output_path: Path,
    show: bool,
    initial_view: str,
    zoom_context_mm: float,
    two_photon_image_path: Path | None = None,
    two_photon_opacity: float = TWO_PHOTON_OVERLAY_ALPHA,
    two_photon_zoom_image_path: Path | None = None,
    two_photon_magnification: float | None = None,
    two_photon_zoom_opacity: float = TWO_PHOTON_ZOOM_OVERLAY_ALPHA,
    two_photon_render_scale: float = 1.0,
) -> None:
    face_order = ("bottom_left", "bottom_right", "top_right", "top_left")
    corner_by_key = {row["corner_key"]: row for row in corner_rows}
    label_offsets = {
        "bottom_left": (-78, -28),
        "bottom_right": (18, -28),
        "top_left": (-78, 22),
        "top_right": (18, 22),
    }
    atlas_spacing_xyz_mm = np.array(
        (atlas_spacing_mm, atlas_spacing_mm, atlas_spacing_mm),
        dtype=float,
    )
    in_plane_extent = tuple(value * atlas_spacing_mm for value in extent_uv)
    in_plane_points = {
        key: geometry.face_uv[key] * atlas_spacing_mm for key in face_order
    }
    two_photon_target_uv_mm = np.stack([
        in_plane_points["top_left"],
        in_plane_points["top_right"],
        in_plane_points["bottom_left"],
        in_plane_points["bottom_right"],
    ])
    two_photon_output_path = output_path.with_name(
        f"{output_path.stem}_2p_overlay{output_path.suffix}"
    )
    two_photon_alignment_path = output_path.with_name(
        f"{output_path.stem}_2p_alignment.json"
    )
    two_photon_alignment_visual_path = output_path.with_name(
        f"{output_path.stem}_2p_alignment{output_path.suffix}"
    )

    center_xyz = geometry.origin_xyz
    nz, ny, nx = annotation.shape
    coronal_index = int(np.clip(round(center_xyz[2]), 0, nz - 1))
    sagittal_index = int(np.clip(round(center_xyz[0]), 0, nx - 1))
    axial_index = int(np.clip(round(center_xyz[1]), 0, ny - 1))
    coronal_points = {
        key: np.array((point[0], -point[1])) * atlas_spacing_mm
        for key, point in display_corners_ccf.items()
    }
    sagittal_points = {
        key: project_xyz_to_view_mm(point, "sagittal", atlas_spacing_xyz_mm)
        for key, point in display_corners_ccf.items()
    }
    axial_points = {
        key: project_xyz_to_view_mm(point, "axial", atlas_spacing_xyz_mm)
        for key, point in display_corners_ccf.items()
    }

    modes = {
        "in_plane": {
            "labels": labels,
            "template": plane_template,
            "extent": in_plane_extent,
            "origin": "lower",
            "points": in_plane_points,
            "title": "In-plane prism view",
            "slice_axis": None,
            "slice_index": None,
        },
        "coronal": {
            "labels": annotation[coronal_index, :, :],
            "template": (
                template[coronal_index, :, :] if template is not None else None
            ),
            "extent": (
                0.0, (nx - 1) * atlas_spacing_mm,
                -(ny - 1) * atlas_spacing_mm, 0.0,
            ),
            "origin": "upper",
            "points": coronal_points,
            "title": f"Coronal Allen slice (AP index {coronal_index})",
            "slice_axis": 2,
            "slice_index": coronal_index,
        },
        "sagittal": {
            "labels": annotation[:, :, sagittal_index].T,
            "template": (
                template[:, :, sagittal_index].T
                if template is not None else None
            ),
            "extent": (
                0.0, (nz - 1) * atlas_spacing_mm,
                -(ny - 1) * atlas_spacing_mm, 0.0,
            ),
            "origin": "upper",
            "points": sagittal_points,
            "title": f"Sagittal Allen slice (ML index {sagittal_index})",
            "slice_axis": 0,
            "slice_index": sagittal_index,
        },
        "axial": {
            "labels": annotation[:, axial_index, :].T,
            "template": (
                template[:, axial_index, :].T
                if template is not None else None
            ),
            "extent": (
                0.0, (nz - 1) * atlas_spacing_mm,
                -(nx - 1) * atlas_spacing_mm, 0.0,
            ),
            "origin": "upper",
            "points": axial_points,
            "title": f"Axial Allen slice (DV index {axial_index})",
            "slice_axis": 1,
            "slice_index": axial_index,
        },
    }
    normal_axis = int(np.argmax(np.abs(geometry.normal_unit_xyz)))
    nearest_mode = ("sagittal", "axial", "coronal")[normal_axis]
    microct_available = microct_image is not None
    microct_slices: dict[str, int] = {}
    if microct_available:
        source_center_xyz = np.mean(
            np.stack([corners_source[key] for key in face_order]), axis=0
        )
        microct_nz, microct_ny, microct_nx = microct_image.shape
        microct_slices = {
            "coronal": int(np.clip(round(source_center_xyz[2]), 0, microct_nz - 1)),
            "sagittal": int(np.clip(round(source_center_xyz[0]), 0, microct_nx - 1)),
            "axial": int(np.clip(round(source_center_xyz[1]), 0, microct_ny - 1)),
        }

    figure, axis = plt.subplots(figsize=(14.2, 8.2), facecolor=UI_BG)
    figure.subplots_adjust(left=0.006, right=0.771, top=0.936, bottom=0.078)
    microct_axis = figure.add_axes([0.487, 0.078, 0.24, 0.852])
    microct_axis.set_visible(False)
    microct_axis.set_facecolor(UI_BG)
    panel_axis = figure.add_axes([0.781, 0.078, 0.209, 0.852])
    panel_axis.set_facecolor(UI_PANEL)
    banner_axis = figure.add_axes([0.006, 0.942, 0.984, 0.046])
    banner_axis.set_facecolor(UI_BANNER)
    banner_axis.set_xticks([])
    banner_axis.set_yticks([])
    for _spine in banner_axis.spines.values():
        _spine.set_color(UI_ACCENT_VIEW)
        _spine.set_linewidth(1.1)
    banner_steps = banner_axis.text(
        0.007, 0.5, "", ha="left", va="center", color=UI_ACCENT_VIEW,
        fontsize=UI_FONT_SMALL, fontweight="bold",
        transform=banner_axis.transAxes,
    )
    banner_message = banner_axis.text(
        0.052, 0.5, "", ha="left", va="center", color=UI_TEXT,
        fontsize=UI_FONT_BODY - 0.6, transform=banner_axis.transAxes,
    )
    banner_title = banner_axis.text(
        0.994, 0.5, "", ha="right", va="center", color=UI_MUTED,
        fontsize=UI_FONT_SMALL, fontweight="bold",
        transform=banner_axis.transAxes,
    )
    state: dict[str, Any] = {
        "mode": "in_plane",
        "compare_microct": False,
        "show_atlas_colors": True,
        "show_labels": True,
        "show_prism_fill": True,
        "show_two_photon": True,
        "two_photon_image_path": two_photon_image_path,
        "two_photon_image": None,
        "two_photon_warped_rgba": None,
        "two_photon_source_points": None,
        "two_photon_homography": None,
        "two_photon_opacity": float(np.clip(two_photon_opacity, 0.0, 1.0)),
        "two_photon_selecting": False,
        "two_photon_selection_points": [],
        "two_photon_selection_bounds": None,
        "two_photon_pan": False,
        "two_photon_zoom_image_path": two_photon_zoom_image_path,
        "two_photon_zoom_image": None,
        "two_photon_magnification": (
            float(two_photon_magnification)
            if two_photon_magnification else DEFAULT_ZOOM_MAGNIFICATION
        ),
        "two_photon_zoom_translation": (0.0, 0.0),
        "two_photon_zoom_alpha": float(
            np.clip(two_photon_zoom_opacity, 0.0, 1.0)
        ),
        "two_photon_zoom_to_wide": None,
        "two_photon_zoom_homography": None,
        "two_photon_zoom_warped_rgba": None,
        "two_photon_warped_extent": None,
        "two_photon_zoom_warped_extent": None,
        "two_photon_render_scale": max(float(two_photon_render_scale), 0.05),
        "context_group": None,
        "show_region_panel": True,
        "show_two_photon_zoom": True,
        "two_photon_aligning": False,
        "two_photon_align_bounds": None,
        "two_photon_align_drag": None,
        "bounds": {},
        "microct_bounds": {},
        "microct_pan": False,
        "drag_start": None,
        "drag_limits": None,
        "microct_drag_start": None,
        "microct_drag_limits": None,
    }

    def color_image(mode_labels: np.ndarray) -> tuple[np.ndarray, ListedColormap]:
        region_ids = sorted(int(value) for value in np.unique(mode_labels))
        id_to_index = {
            region_id: index for index, region_id in enumerate(region_ids)
        }
        indices = np.vectorize(id_to_index.__getitem__)(mode_labels)
        color_map = ListedColormap(
            [ontology.get(region_id).rgb for region_id in region_ids]
        )
        return indices, color_map

    def calculate_bounds(
        mode_labels: np.ndarray,
        extent: tuple[float, float, float, float],
        polygon_mm: np.ndarray,
        origin: str,
    ) -> tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]:
        rows, columns = np.nonzero(mode_labels > 0)
        if rows.size:
            x_values = extent[0] + (
                columns / max(1, mode_labels.shape[1] - 1)
                * (extent[1] - extent[0])
            )
            if origin == "lower":
                y_values = extent[2] + (
                    rows / max(1, mode_labels.shape[0] - 1)
                    * (extent[3] - extent[2])
                )
            else:
                y_values = extent[3] + (
                    rows / max(1, mode_labels.shape[0] - 1)
                    * (extent[2] - extent[3])
                )
            content_x = np.concatenate((x_values, polygon_mm[:, 0]))
            content_y = np.concatenate((y_values, polygon_mm[:, 1]))
            span = max(float(np.ptp(content_x)), float(np.ptp(content_y)))
            padding = max(span * 0.055, atlas_spacing_mm * 4)
            full = (
                float(content_x.min() - padding),
                float(content_x.max() + padding),
                float(content_y.min() - padding),
                float(content_y.max() + padding),
            )
        else:
            full = extent
        center = polygon_mm.mean(axis=0)
        face_width = max(float(np.ptp(polygon_mm[:, 0])), atlas_spacing_mm * 4)
        face_height = max(float(np.ptp(polygon_mm[:, 1])), atlas_spacing_mm * 4)
        half_span = max(face_width, face_height) * 1.8
        half_span = max(half_span, zoom_context_mm / 2.0)
        close = (
            float(center[0] - half_span),
            float(center[0] + half_span),
            float(center[1] - half_span),
            float(center[1] + half_span),
        )
        return close, full

    def comparison_active() -> bool:
        return (
            bool(state["compare_microct"])
            and microct_available
            and state["mode"] in {"coronal", "sagittal", "axial"}
        )

    def clamp_bounds_to_extent(
        bounds: tuple[float, float, float, float],
        extent: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        x_min, x_max, y_min, y_max = [float(value) for value in bounds]
        ex_min, ex_max, ey_min, ey_max = [float(value) for value in extent]
        x_low, x_high = sorted((ex_min, ex_max))
        y_low, y_high = sorted((ey_min, ey_max))

        width = min(abs(x_max - x_min), x_high - x_low)
        height = min(abs(y_max - y_min), y_high - y_low)
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        x_center = float(np.clip(x_center, x_low + width / 2.0, x_high - width / 2.0))
        y_center = float(np.clip(y_center, y_low + height / 2.0, y_high - height / 2.0))
        return (
            x_center - width / 2.0,
            x_center + width / 2.0,
            y_center - height / 2.0,
            y_center + height / 2.0,
        )

    def apply_viewer_layout() -> None:
        """Position the axes for the current mode.

        The main view grows when the contextual button row is hidden and
        when the region panel is collapsed, so screen space follows what is
        actually on show.
        """
        bottom = 0.136 if state.get("context_group") else 0.078
        top = 0.936
        height = top - bottom
        comparing = comparison_active()
        panel_open = bool(state.get("show_region_panel", True)) or comparing
        if comparing:
            axis.set_position([0.022, bottom, 0.45, height])
            microct_axis.set_position([0.487, bottom, 0.24, height])
            microct_axis.set_visible(True)
            panel_axis.set_position([0.742, bottom, 0.248, height])
        else:
            main_width = 0.765 if panel_open else 0.984
            axis.set_position([0.006, bottom, main_width, height])
            panel_axis.set_position([0.781, bottom, 0.209, height])
            microct_axis.clear()
            microct_axis.set_visible(False)
        panel_axis.set_visible(panel_open)

    context_widgets: dict[str, list[Any]] = {
        "align": [], "corners": [], "uct": [], "layers": [],
    }

    def set_context_group(name: str | None) -> None:
        """Show one contextual button row at a time, or none at all."""
        state["context_group"] = name
        for group, widgets in context_widgets.items():
            visible = group == name
            for widget in widgets:
                widget.ax.set_visible(visible)
                # Invisible widgets still receive events unless deactivated,
                # which let stacked sliders fight over the mouse grab and
                # let hidden buttons fire on clicks inside the atlas view.
                widget.set_active(visible)
        apply_viewer_layout()
        figure.canvas.draw_idle()

    def microct_display(mode_name: str) -> dict[str, Any]:
        if microct_image is None:
            raise RuntimeError("No microCT image is available for comparison.")
        sx, sy, sz = [float(value) for value in source_spacing_xyz_mm]
        nz_m, ny_m, nx_m = microct_image.shape
        slice_index = microct_slices[mode_name]
        if mode_name == "coronal":
            image_slice = microct_image[slice_index, :, :]
            extent = (0.0, (nx_m - 1) * sx, -(ny_m - 1) * sy, 0.0)
            points = {
                key: project_xyz_to_view_mm(point, mode_name, source_spacing_xyz_mm)
                for key, point in corners_source.items()
            }
            slice_axis = 2
            max_slice = nz_m - 1
            label = "Coronal microCT"
        elif mode_name == "sagittal":
            image_slice = microct_image[:, :, slice_index].T
            extent = (0.0, (nz_m - 1) * sz, -(ny_m - 1) * sy, 0.0)
            points = {
                key: project_xyz_to_view_mm(point, mode_name, source_spacing_xyz_mm)
                for key, point in corners_source.items()
            }
            slice_axis = 0
            max_slice = nx_m - 1
            label = "Sagittal microCT"
        else:
            image_slice = microct_image[:, slice_index, :].T
            extent = (0.0, (nz_m - 1) * sz, -(nx_m - 1) * sx, 0.0)
            points = {
                key: project_xyz_to_view_mm(point, mode_name, source_spacing_xyz_mm)
                for key, point in corners_source.items()
            }
            slice_axis = 1
            max_slice = ny_m - 1
            label = "Axial microCT"
        intersection_xyz = face_slice_intersection_xyz(
            corners_source, face_order, slice_axis, slice_index
        )
        near_keys = near_slice_corner_keys(
            corners_source, face_order, slice_axis, slice_index
        )
        return {
            "slice": image_slice,
            "extent": extent,
            "points": points,
            "intersection_points": project_xyz_list_to_view_mm(
                intersection_xyz, mode_name, source_spacing_xyz_mm
            ),
            "near_corner_keys": near_keys,
            "slice_index": slice_index,
            "max_slice": max_slice,
            "title": label,
        }

    def draw_microct_comparison() -> None:
        if not comparison_active():
            return
        display = microct_display(state["mode"])
        image_slice = display["slice"]
        positive = image_slice[np.isfinite(image_slice) & (image_slice > 0)]
        if positive.size:
            low, high = np.percentile(positive, (1.0, 99.5))
        elif np.isfinite(image_slice).any():
            low, high = np.percentile(
                image_slice[np.isfinite(image_slice)], (1.0, 99.5)
            )
        else:
            low, high = 0.0, 1.0
        high = float(max(high, low + 1e-6))

        microct_axis.clear()
        microct_axis.set_facecolor("black")
        microct_axis.imshow(
            image_slice, cmap="gray", origin="upper", interpolation="bilinear",
            extent=display["extent"], aspect="equal",
            vmin=float(low), vmax=high,
        )
        polygon_mm = np.stack([display["points"][key] for key in face_order])
        if state["show_prism_fill"]:
            microct_axis.add_patch(Polygon(
                polygon_mm, closed=True, facecolor=PRISM_FILL_COLOR,
                edgecolor="none", linewidth=0.0, zorder=5,
            ))
        microct_axis.add_patch(Polygon(
            polygon_mm, closed=True, facecolor="none",
            edgecolor=PRISM_OVERLAY_COLOR, linewidth=2.4, zorder=6,
        ))
        visible_keys = face_order
        for key in visible_keys:
            position = display["points"][key]
            microct_axis.scatter(
                [position[0]], [position[1]], s=58,
                c=PRISM_MARKER_COLOR, edgecolors="white", linewidths=1.0,
                zorder=7, clip_on=False,
            )
        microct_axis.set_title(
            f'{display["title"]} | slice '
            f'{display["slice_index"]}/{display["max_slice"]}',
            color="white", fontsize=10.5, fontweight="bold", pad=7,
        )
        bounds = state["microct_bounds"].get(state["mode"], display["extent"])
        microct_axis.set_xlim(bounds[0], bounds[1])
        microct_axis.set_ylim(bounds[2], bounds[3])
        microct_axis.axis("off")
        microct_axis.set_aspect("equal", adjustable="box")

    def set_microct_view(bounds: tuple[float, float, float, float]) -> None:
        if not comparison_active():
            return
        display = microct_display(state["mode"])
        bounds = clamp_bounds_to_extent(bounds, display["extent"])
        state["microct_bounds"][state["mode"]] = bounds
        microct_axis.set_xlim(bounds[0], bounds[1])
        microct_axis.set_ylim(bounds[2], bounds[3])
        figure.canvas.draw_idle()

    def zoom_microct_view(scale: float) -> None:
        if not comparison_active():
            return
        display = microct_display(state["mode"])
        x_min, x_max = microct_axis.get_xlim()
        y_min, y_max = microct_axis.get_ylim()
        if not np.isfinite([x_min, x_max, y_min, y_max]).all():
            x_min, x_max, y_min, y_max = display["extent"]
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        full_width = abs(display["extent"][1] - display["extent"][0])
        full_height = abs(display["extent"][3] - display["extent"][2])
        half_width = np.clip(
            abs(x_max - x_min) * scale / 2.0,
            min(full_width / 2.0, max(source_spacing_xyz_mm[0] * 2.0, 1e-6)),
            full_width / 2.0,
        )
        half_height = np.clip(
            abs(y_max - y_min) * scale / 2.0,
            min(full_height / 2.0, max(source_spacing_xyz_mm[1] * 2.0, 1e-6)),
            full_height / 2.0,
        )
        set_microct_view((
            center_x - half_width,
            center_x + half_width,
            center_y - half_height,
            center_y + half_height,
        ))

    def reset_microct_view() -> None:
        if not comparison_active():
            return
        display = microct_display(state["mode"])
        set_microct_view(display["extent"])

    def pan_microct_view(delta_x: float, delta_y: float) -> None:
        if not comparison_active():
            return
        x_min, x_max = microct_axis.get_xlim()
        y_min, y_max = microct_axis.get_ylim()
        set_microct_view((
            x_min - delta_x,
            x_max - delta_x,
            y_min - delta_y,
            y_max - delta_y,
        ))

    def change_microct_slice(delta: int) -> None:
        if not comparison_active():
            return
        display = microct_display(state["mode"])
        microct_slices[state["mode"]] = int(np.clip(
            display["slice_index"] + delta, 0, display["max_slice"]
        ))
        draw_microct_comparison()
        figure.canvas.draw_idle()

    def set_status(
        message: str,
        step: int | None = None,
        tone: str = "info",
        flush: bool = False,
    ) -> None:
        """Put the current instruction on screen, not only in the terminal.

        ``flush`` forces a synchronous repaint, used before opening a modal
        file dialog so the instruction is readable behind it.
        """
        tones = {
            "info": UI_ACCENT_VIEW,
            "twop": TWO_PHOTON_BUTTON_COLOR,
            "zoom": TWO_PHOTON_ZOOM_BUTTON_COLOR,
        }
        colour = tones.get(tone, UI_ACCENT_VIEW)
        banner_steps.set_text(
            "" if step is None else "  ".join(
                "\u25cf" if index == step else "\u25cb"
                for index in (1, 2, 3)
            )
        )
        banner_steps.set_color(colour)
        banner_message.set_text(message)
        for spine in banner_axis.spines.values():
            spine.set_color(colour)
        figure.canvas.draw_idle()
        if flush:
            try:
                figure.canvas.draw()
                figure.canvas.flush_events()
            except Exception as error:
                print(f"WARNING: banner repaint failed: {error}")

    def fit_bounds_to_axes(
        bounds: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        """Widen bounds to the axes' own aspect ratio.

        With equal aspect, a view whose data is squarer than its box gets
        letterboxed with dead space either side. Expanding the short axis to
        match the box means the slice fills the window instead.
        """
        position = axis.get_position()
        figure_width, figure_height = figure.get_size_inches()
        box_width = position.width * figure_width
        box_height = position.height * figure_height
        half_x = (bounds[1] - bounds[0]) / 2.0
        half_y = (bounds[3] - bounds[2]) / 2.0
        if min(box_width, box_height) <= 0 or not (half_x and half_y):
            return bounds
        box_aspect = box_width / box_height
        if abs(half_x) / abs(half_y) < box_aspect:
            half_x = np.sign(half_x) * abs(half_y) * box_aspect
        else:
            half_y = np.sign(half_y) * abs(half_x) / box_aspect
        centre_x = (bounds[0] + bounds[1]) / 2.0
        centre_y = (bounds[2] + bounds[3]) / 2.0
        return (
            float(centre_x - half_x), float(centre_x + half_x),
            float(centre_y - half_y), float(centre_y + half_y),
        )

    def set_view_title(text: str, colour: str = UI_MUTED) -> None:
        """Show the current view name in the banner.

        An axes title would sit in the strip the banner occupies, so the two
        overlapped; keeping it in the banner also returns that height to the
        slice itself.
        """
        banner_title.set_text(text)
        banner_title.set_color(colour)
        figure.canvas.draw_idle()

    def set_view(bounds: tuple[float, float, float, float]) -> None:
        bounds = fit_bounds_to_axes(bounds)
        axis.set_xlim(bounds[0], bounds[1])
        axis.set_ylim(bounds[2], bounds[3])
        if state["two_photon_selecting"]:
            state["two_photon_selection_bounds"] = bounds
        figure.canvas.draw_idle()

    def refresh_live_view() -> None:
        """Force the interactive Matplotlib window to repaint after modal dialogs."""
        try:
            plt.figure(figure.number)
            figure.canvas.draw_idle()
            figure.canvas.draw()
            figure.canvas.flush_events()
            plt.pause(0.05)
        except Exception as error:
            print(f"WARNING: live viewer refresh failed: {error}")

    def two_photon_zoom_extent_px() -> tuple[float, float, float, float]:
        """Matplotlib imshow extent for the scaled zoom, in wide-image pixels."""
        translate_x, translate_y = state["two_photon_zoom_translation"]
        display_width, display_height = zoom_display_size_px(
            state["two_photon_magnification"],
            state["two_photon_image"].shape[:2],
        )
        return (
            translate_x, translate_x + display_width,
            translate_y + display_height, translate_y,
        )

    def current_zoom_to_wide() -> np.ndarray:
        return zoom_to_wide_matrix(
            state["two_photon_magnification"],
            state["two_photon_zoom_translation"],
            state["two_photon_zoom_image"].shape[:2],
            state["two_photon_image"].shape[:2],
        )

    def center_two_photon_zoom() -> None:
        wide_height, wide_width = state["two_photon_image"].shape[:2]
        display_width, display_height = zoom_display_size_px(
            state["two_photon_magnification"], (wide_height, wide_width)
        )
        state["two_photon_zoom_translation"] = (
            (wide_width - display_width) / 2.0,
            (wide_height - display_height) / 2.0,
        )

    def draw_two_photon_alignment_view(reset_view: bool = False) -> None:
        wide_image = state["two_photon_image"]
        zoom_image = state["two_photon_zoom_image"]
        if wide_image is None or zoom_image is None:
            print("2p alignment needs both an unzoomed and a zoomed image.")
            return
        height, width = wide_image.shape[:2]
        state["compare_microct"] = False
        apply_viewer_layout()
        axis.clear()
        axis.set_facecolor("black")
        axis.imshow(wide_image, origin="upper", extent=(0.0, width, height, 0.0))
        extent = two_photon_zoom_extent_px()
        alpha = float(np.clip(state["two_photon_zoom_alpha"], 0.0, 1.0))
        axis.imshow(
            zoom_image, origin="upper", extent=extent, alpha=alpha,
            zorder=5, interpolation="bilinear",
        )
        axis.add_patch(Rectangle(
            (extent[0], extent[3]),
            extent[1] - extent[0], extent[2] - extent[3],
            facecolor="none", edgecolor=TWO_PHOTON_ZOOM_OUTLINE_COLOR,
            linewidth=2.2, zorder=6,
        ))
        axis.set_aspect("equal", adjustable="box")
        axis.axis("off")
        if reset_view or state["two_photon_align_bounds"] is None:
            axis.set_xlim(0.0, width)
            axis.set_ylim(height, 0.0)
            state["two_photon_align_bounds"] = (
                0.0, float(width), float(height), 0.0
            )
        else:
            bounds = state["two_photon_align_bounds"]
            axis.set_xlim(bounds[0], bounds[1])
            axis.set_ylim(bounds[2], bounds[3])
        translate_x, translate_y = state["two_photon_zoom_translation"]
        axis.set_title("")
        set_view_title(
            f"{state['two_photon_magnification']:g}x  \u00b7  top-left "
            f"({translate_x:.1f}, {translate_y:.1f}) px  \u00b7  "
            f"alpha {alpha:.2f}",
            TWO_PHOTON_ZOOM_OUTLINE_COLOR,
        )
        figure.canvas.draw_idle()

    def nudge_two_photon_zoom(delta_x: float, delta_y: float) -> None:
        translate_x, translate_y = state["two_photon_zoom_translation"]
        state["two_photon_zoom_translation"] = (
            translate_x + float(delta_x), translate_y + float(delta_y),
        )
        draw_two_photon_alignment_view(reset_view=False)

    def reset_two_photon_alignment() -> None:
        center_two_photon_zoom()
        draw_two_photon_alignment_view(reset_view=True)

    def cancel_two_photon_alignment() -> None:
        state["two_photon_aligning"] = False
        state["two_photon_zoom_image"] = None
        state["two_photon_align_drag"] = None
        state["two_photon_align_bounds"] = None
        print("2p zoom alignment canceled; the zoomed image was discarded.")
        set_context_group(None)
        set_status("Ready. Map 2p starts the two-photon overlay.", None, "info")
        draw_mode(state["mode"])

    def save_two_photon_alignment_visual() -> None:
        try:
            figure.savefig(
                two_photon_alignment_visual_path, dpi=260,
                facecolor=figure.get_facecolor(),
                bbox_inches="tight", pad_inches=0.02,
            )
            print(
                "Saved 2p zoom alignment QC visual: "
                f"{two_photon_alignment_visual_path}"
            )
        except (OSError, ValueError) as error:
            print(f"WARNING: could not save 2p alignment visual: {error}")

    def start_two_photon_corner_selection() -> None:
        set_context_group("corners")
        set_status(
            "Step 3 of 3  \u2014  click the four corners of the UNZOOMED "
            "image along the black border: D1 top-left, D2 top-right, "
            "P1 bottom-left, P2 bottom-right", 3, "twop",
        )
        state["two_photon_selecting"] = True
        state["two_photon_selection_points"] = []
        state["two_photon_selection_bounds"] = None
        state["two_photon_pan"] = False
        state["drag_start"] = None
        state["drag_limits"] = None
        print(
            "\n2p corner selection is now active in the main atlas window. "
            "Click the corners of the UNZOOMED image in order: D1/top-left, "
            "D2/top-right, P1/bottom-left, P2/bottom-right, along the black "
            "border where the band of emptiness appears. The overlay is "
            "generated automatically after the fourth click."
        )
        draw_two_photon_selection_view(reset_view=True)
        refresh_live_view()

    def report_zoom_alignment_quality() -> dict[str, Any]:
        """Score the current alignment and say plainly whether it looks right."""
        try:
            quality = score_zoom_alignment(
                state["two_photon_image"],
                state["two_photon_zoom_image"],
                state["two_photon_magnification"],
                state["two_photon_zoom_translation"],
            )
        except (ValueError, IndexError) as error:
            print(f"WARNING: could not score the zoom alignment: {error}")
            return {"available": False, "reason": str(error)}
        if not quality.get("available"):
            print(f"Alignment check skipped: {quality.get('reason')}")
            set_view_title("alignment check unavailable", UI_MUTED)
            return quality
        chosen = quality["correlation_at_chosen_offset"]
        best = quality["best_nearby_correlation"]
        shift = quality["best_nearby_shift_px"]
        print(
            "\nAlignment check (image match, 1.0 is perfect):\n"
            f"  your offset:      r = {chosen:+.4f}\n"
            f"  best within {quality['search_radius_px']} px: "
            f"r = {best:+.4f} at dx={shift['x']:+.0f}, dy={shift['y']:+.0f}"
        )
        if quality["suspicious"]:
            print(
                "  WARNING: a nearby offset matches noticeably better. "
                "Consider nudging by that amount and checking again."
            )
            set_view_title(
                f"match r={chosen:+.3f}  \u00b7  better by "
                f"({shift['x']:+.0f}, {shift['y']:+.0f}) at r={best:+.3f}",
                TWO_PHOTON_BUTTON_COLOR,
            )
        else:
            print("  Looks consistent: no better offset found nearby.")
            set_view_title(
                f"match r={chosen:+.3f}  \u00b7  no better offset nearby",
                TWO_PHOTON_ZOOM_OUTLINE_COLOR,
            )
        return quality

    def complete_two_photon_alignment() -> None:
        if not state["two_photon_aligning"]:
            return
        zoom_to_wide = current_zoom_to_wide()
        state["two_photon_zoom_to_wide"] = zoom_to_wide
        translate_x, translate_y = state["two_photon_zoom_translation"]
        print(
            "\n2p zoom alignment recorded:\n"
            f"  magnification: {state['two_photon_magnification']:g}x\n"
            f"  scale factor:  {zoom_to_wide[0, 0]:.6f} (x), "
            f"{zoom_to_wide[1, 1]:.6f} (y)\n"
            f"  translation:   x={translate_x:.3f}, y={translate_y:.3f} "
            "unzoomed px, 0-based, top-left corner of the scaled zoom"
        )
        quality = report_zoom_alignment_quality()
        save_two_photon_alignment_visual()
        try:
            write_two_photon_alignment(
                two_photon_alignment_path,
                state["two_photon_image_path"],
                state["two_photon_zoom_image_path"],
                state["two_photon_magnification"],
                (translate_x, translate_y),
                state["two_photon_zoom_image"].shape[:2],
                state["two_photon_image"].shape[:2],
                zoom_to_wide,
                extra={"alignment_quality": quality},
            )
            print(f"Saved 2p alignment parameters: {two_photon_alignment_path}")
        except OSError as error:
            print(f"WARNING: could not save 2p alignment parameters: {error}")
        state["two_photon_aligning"] = False
        state["two_photon_align_drag"] = None
        state["two_photon_zoom_alpha"] = 1.0
        try:
            zoom_alpha_slider.set_val(1.0)
        except NameError:
            pass
        start_two_photon_corner_selection()

    def draw_two_photon_selection_view(reset_view: bool = False) -> None:
        image = state["two_photon_image"]
        if image is None:
            print("2p selection cannot start: no 2p image is loaded.")
            return
        image = np.asarray(image)
        height, width = image.shape[:2]
        points = list(state["two_photon_selection_points"])
        corner_steps = ("D1", "D2", "P1", "P2")
        corner_descriptions = (
            "top-left 2p FOV corner",
            "top-right 2p FOV corner",
            "bottom-left 2p FOV corner",
            "bottom-right 2p FOV corner",
        )

        state["compare_microct"] = False
        apply_viewer_layout()
        axis.clear()
        axis.set_facecolor("black")
        axis.imshow(image, origin="upper", extent=(0.0, width, height, 0.0))
        axis.set_aspect("equal", adjustable="box")
        axis.axis("off")
        if reset_view or state["two_photon_selection_bounds"] is None:
            axis.set_xlim(0.0, width)
            axis.set_ylim(height, 0.0)
            state["two_photon_selection_bounds"] = (0.0, float(width), float(height), 0.0)
        else:
            set_view(state["two_photon_selection_bounds"])

        if len(points) < len(corner_steps):
            next_index = len(points)
            status = (
                f"2p corner selection | Next: {corner_steps[next_index]} - "
                f"{corner_descriptions[next_index]}"
            )
        else:
            status = "2p corner selection complete | mapping to prism plane..."
        status += " | Pan on" if state["two_photon_pan"] else " | Pan off"
        axis.set_title("")
        set_view_title(status, TWO_PHOTON_BUTTON_COLOR)

        for index, (x_value, y_value) in enumerate(points):
            label = corner_steps[index]
            axis.scatter(
                [x_value], [y_value], s=155, c=TWO_PHOTON_MARKER_COLOR,
                marker="o", edgecolors="black", linewidths=1.5, zorder=5,
            )
            axis.scatter(
                [x_value], [y_value], s=175, c="white", marker="+",
                linewidths=2.4, zorder=6,
            )
            axis.scatter(
                [x_value], [y_value], s=260, facecolors="none",
                edgecolors="black", linewidths=1.1, zorder=7,
            )
            text = axis.annotate(
                label, (x_value, y_value), xytext=(9, -9),
                textcoords="offset points", color="white",
                fontsize=10.5, fontweight="bold", zorder=8,
            )
            text.set_path_effects([
                path_effects.withStroke(linewidth=3, foreground="black")
            ])
        figure.canvas.draw_idle()

    def reset_two_photon_selection() -> None:
        state["two_photon_selection_points"] = []
        state["two_photon_selection_bounds"] = None
        draw_two_photon_selection_view(reset_view=True)

    def cancel_two_photon_selection() -> None:
        state["two_photon_selecting"] = False
        state["two_photon_selection_points"] = []
        state["two_photon_selection_bounds"] = None
        state["two_photon_pan"] = False
        print("2p corner selection canceled.")
        set_context_group(None)
        set_status("Ready. Map 2p starts the two-photon overlay.", None, "info")
        draw_mode(state["mode"])

    def undo_two_photon_selection() -> None:
        points = state["two_photon_selection_points"]
        if points:
            removed = points.pop()
            print(f"Removed last 2p corner: x={removed[0]:.3f}, y={removed[1]:.3f} px")
        draw_two_photon_selection_view(reset_view=False)

    def record_two_photon_selection_point(x_value: float, y_value: float) -> None:
        if not state["two_photon_selecting"]:
            return
        points = state["two_photon_selection_points"]
        if len(points) >= 4:
            return
        label = ("D1", "D2", "P1", "P2")[len(points)]
        points.append((float(x_value), float(y_value)))
        print(f"Selected 2p {label}: x={x_value:.3f}, y={y_value:.3f} px")
        draw_two_photon_selection_view(reset_view=False)
        if len(points) == 4:
            complete_two_photon_mapping_from_points()

    def save_two_photon_overlay_visual() -> None:
        draw_mode("in_plane")
        figure.savefig(
            two_photon_output_path, dpi=260, facecolor=figure.get_facecolor(),
            bbox_inches="tight", pad_inches=0.02,
        )
        print(f"Saved 2p prism-plane overlay visual: {two_photon_output_path}")

    def complete_two_photon_mapping_from_points() -> None:
        two_photon_image = state["two_photon_image"]
        source_points = np.asarray(state["two_photon_selection_points"], dtype=float)
        if two_photon_image is None or source_points.shape != (4, 2):
            print("2p mapping not ready: expected exactly four selected 2p corners.")
            return

        print("Recorded all four 2p corners in order: D1, D2, P1, P2.")
        for label, point in zip(("D1", "D2", "P1", "P2"), source_points):
            print(f"  2p {label}: x={point[0]:.3f}, y={point[1]:.3f} px")
        render_scale = state["two_photon_render_scale"]
        face_polygon = two_photon_target_uv_mm[[0, 1, 3, 2]]
        try:
            homography = solve_projective_transform(
                source_points, two_photon_target_uv_mm
            )
            inverse = np.linalg.inv(homography)
            wide_extent, wide_shape, wide_pitch = plane_render_grid(
                two_photon_target_uv_mm,
                two_photon_image.shape[:2],
                render_scale,
            )
            warped_rgba = warp_image_with_homography(
                two_photon_image, homography, wide_extent, wide_shape,
                1.0, clip_polygon_uv_mm=face_polygon,
            )
        except (np.linalg.LinAlgError, ValueError) as error:
            print(f"Could not map 2p image onto prism plane: {error}")
            return
        print(
            f"Unzoomed 2p overlay rendered at {wide_shape[1]} x "
            f"{wide_shape[0]} px, {wide_pitch * 1000.0:.2f} um/px "
            f"(the atlas plane grid is {atlas_spacing_mm * 1000.0:.0f} "
            "um/px)."
        )

        state["two_photon_source_points"] = source_points
        state["two_photon_warped_rgba"] = warped_rgba
        state["two_photon_warped_extent"] = wide_extent
        state["two_photon_homography"] = homography
        state["two_photon_inverse_homography"] = inverse
        state["two_photon_zoom_warped_rgba"] = None
        state["two_photon_zoom_homography"] = None
        zoom_image = state["two_photon_zoom_image"]
        zoom_to_wide = state["two_photon_zoom_to_wide"]
        if zoom_image is not None and zoom_to_wide is not None:
            try:
                zoom_homography = compose_zoom_to_plane(
                    homography, zoom_to_wide
                )
                zoom_height, zoom_width = zoom_image.shape[:2]
                zoom_corners_px = np.array([
                    [0.0, 0.0], [zoom_width - 1.0, 0.0],
                    [0.0, zoom_height - 1.0],
                    [zoom_width - 1.0, zoom_height - 1.0],
                ], dtype=float)
                zoom_footprint = apply_projective_transform(
                    zoom_corners_px, zoom_homography
                )
                zoom_extent, zoom_shape, zoom_pitch = plane_render_grid(
                    zoom_footprint, zoom_image.shape[:2], render_scale,
                )
                zoom_warped = warp_image_with_homography(
                    zoom_image, zoom_homography, zoom_extent, zoom_shape,
                    1.0, clip_polygon_uv_mm=face_polygon,
                )
            except (np.linalg.LinAlgError, ValueError) as error:
                print(f"Could not map the zoomed 2p image: {error}")
            else:
                state["two_photon_zoom_warped_rgba"] = zoom_warped
                state["two_photon_zoom_warped_extent"] = zoom_extent
                print(
                    f"Zoomed 2p overlay rendered at {zoom_shape[1]} x "
                    f"{zoom_shape[0]} px, {zoom_pitch * 1000.0:.2f} um/px."
                )
                state["two_photon_zoom_homography"] = zoom_homography
                state["show_two_photon_zoom"] = True
                zoom_pixels = int(
                    np.count_nonzero(zoom_warped[..., 3] > 0.0)
                )
                print(
                    "Generated zoomed 2p overlay: visible pixels="
                    f"{zoom_pixels}."
                )
                if zoom_pixels == 0:
                    print(
                        "WARNING: the zoomed overlay is empty. Check that "
                        "the zoom was aligned inside the unzoomed image's "
                        "prism quadrilateral."
                    )
                try:
                    write_two_photon_alignment(
                        two_photon_alignment_path,
                        state["two_photon_image_path"],
                        state["two_photon_zoom_image_path"],
                        state["two_photon_magnification"],
                        state["two_photon_zoom_translation"],
                        zoom_image.shape[:2],
                        state["two_photon_image"].shape[:2],
                        zoom_to_wide,
                        wide_to_plane=homography,
                        zoom_to_plane=zoom_homography,
                        extra={
                            "wide_corner_points_px_0based":
                                source_points.tolist(),
                            "prism_face_corners_uv_mm":
                                two_photon_target_uv_mm.tolist(),
                            "atlas_spacing_mm": float(atlas_spacing_mm),
                        },
                    )
                    print(
                        "Updated 2p alignment parameters with both "
                        f"transforms: {two_photon_alignment_path}"
                    )
                except OSError as error:
                    print(
                        "WARNING: could not update alignment "
                        f"parameters: {error}"
                    )
        state["show_two_photon"] = True
        state["show_prism_fill"] = False
        state["two_photon_selecting"] = False
        state["two_photon_selection_bounds"] = None
        state["two_photon_pan"] = False
        try:
            sync_toggle_buttons()
            two_photon_pan_button.label.set_text("Pan off")
        except NameError:
            pass

        alpha_pixels = int(np.count_nonzero(warped_rgba[..., 3] > 0.0))
        if alpha_pixels == 0:
            print(
                "WARNING: generated 2p overlay has zero visible pixels. "
                "Likely causes: corner order produced an invalid quadrilateral, "
                "or the destination prism polygon fell outside the in-plane grid."
            )
        else:
            print(
                f"Generated 2p overlay: {warped_rgba.shape}, "
                f"visible pixels={alpha_pixels}, "
                f"display opacity={state['two_photon_opacity']:.2f}."
            )
        print_two_photon_mapping_diagnostics(
            source_points, two_photon_target_uv_mm, homography
        )
        set_context_group(None)
        set_status(
            "2p overlay mapped onto the prism plane. "
            "Layers shows the visibility toggles and opacity sliders.",
            None, "info",
        )
        save_two_photon_overlay_visual()
        refresh_live_view()

    def draw_mode(mode_name: str, use_full: bool = False) -> None:
        state["mode"] = mode_name
        apply_viewer_layout()
        mode = modes[mode_name]
        mode_labels = mode["labels"]
        extent = mode["extent"]
        points = mode["points"]
        polygon_mm = np.stack([points[key] for key in face_order])
        bounds_points_mm = polygon_mm
        show_atlas_colors = bool(state["show_atlas_colors"])
        two_photon_visible = (
            mode_name == "in_plane" and state["show_two_photon"]
            and state["two_photon_warped_rgba"] is not None
        )
        two_photon_zoom_visible = (
            mode_name == "in_plane" and state["show_two_photon"]
            and state["show_two_photon_zoom"]
            and state["two_photon_zoom_warped_rgba"] is not None
        )

        axis.clear()
        axis.set_facecolor("black")
        mode_template = mode["template"]
        if mode_template is not None:
            positive = mode_template[
                np.isfinite(mode_template) & (mode_template > 0)
            ]
            if positive.size:
                low, high = np.percentile(positive, (1.0, 99.5))
            else:
                low, high = 0.0, 1.0
            axis.imshow(
                mode_template, origin=mode["origin"],
                interpolation="bilinear", cmap="gray", extent=extent,
                aspect="equal", vmin=float(low),
                vmax=float(max(high, low + 1e-6)),
            )
        if show_atlas_colors:
            indices, color_map = color_image(mode_labels)
            axis.imshow(
                indices, origin=mode["origin"], interpolation="nearest",
                cmap=color_map, extent=extent, aspect="equal",
                alpha=(
                    np.where(mode_labels > 0, 0.40, 0.0)
                    if mode_template is not None else 1.0
                ),
            )
            boundary = np.zeros(mode_labels.shape, dtype=bool)
            boundary[1:, :] |= mode_labels[1:, :] != mode_labels[:-1, :]
            boundary[:, 1:] |= mode_labels[:, 1:] != mode_labels[:, :-1]
            boundary_rgba = np.zeros((*mode_labels.shape, 4), dtype=float)
            boundary_rgba[boundary] = (0.0, 0.0, 0.0, 0.75)
            axis.imshow(
                boundary_rgba, origin=mode["origin"], interpolation="nearest",
                extent=extent, aspect="equal",
            )
        elif mode_template is None:
            axis.imshow(
                mode_labels > 0, origin=mode["origin"], interpolation="nearest",
                cmap="gray", extent=extent, aspect="equal", alpha=0.65,
            )
        if state["show_prism_fill"]:
            axis.add_patch(Polygon(
                polygon_mm, closed=True, facecolor=PRISM_FILL_COLOR,
                edgecolor="none", linewidth=0.0, zorder=5,
            ))
        if two_photon_visible:
            overlay_rgba = np.array(state["two_photon_warped_rgba"], copy=True)
            overlay_rgba[..., 3] *= float(state["two_photon_opacity"])
            axis.imshow(
                overlay_rgba, origin="lower", interpolation="bilinear",
                extent=state["two_photon_warped_extent"],
                aspect="equal", zorder=6,
            )
        if two_photon_zoom_visible:
            zoom_rgba = np.array(
                state["two_photon_zoom_warped_rgba"], copy=True
            )
            zoom_rgba[..., 3] *= float(state["two_photon_zoom_alpha"])
            axis.imshow(
                zoom_rgba, origin="lower", interpolation="bilinear",
                extent=state["two_photon_zoom_warped_extent"],
                aspect="equal", zorder=6.5,
            )
        axis.add_patch(Polygon(
            polygon_mm, closed=True, facecolor="none",
            edgecolor=PRISM_OVERLAY_COLOR, linewidth=3.8, zorder=7,
        ))
        visible_keys = face_order

        for key in visible_keys:
            row = corner_by_key[key]
            position_mm = points[key]
            axis.scatter(
                [position_mm[0]], [position_mm[1]], s=96,
                c=PRISM_MARKER_COLOR, edgecolors="white", linewidths=1.4,
                zorder=8, clip_on=False,
            )
            if state["show_labels"]:
                label = axis.annotate(
                    f'{row["corner"]}: ID {row["region_id"]}',
                    position_mm, xytext=label_offsets[key],
                    textcoords="offset points", color="white",
                    fontsize=8.5, fontweight="bold", zorder=9,
                    arrowprops={
                        "arrowstyle": "-", "color": PRISM_OVERLAY_COLOR, "linewidth": 1.0,
                    },
                )
                label.set_path_effects([
                    path_effects.withStroke(linewidth=3, foreground="black")
                ])
        axis.set_title("")
        set_view_title(mode["title"])
        axis.set_aspect("equal", adjustable="box")
        axis.axis("off")
        close_bounds, full_bounds = calculate_bounds(
            mode_labels, extent, bounds_points_mm, mode["origin"]
        )
        state["bounds"][mode_name] = {
            "close": close_bounds,
            "full": full_bounds,
        }
        set_view(full_bounds if use_full else close_bounds)
        try:
            refresh_view_buttons()
        except NameError:
            pass
        draw_microct_comparison()

    def zoom_view(scale: float, center: tuple[float, float] | None = None) -> None:
        x_min, x_max = axis.get_xlim()
        y_start, y_end = axis.get_ylim()
        center_x = (x_min + x_max) / 2.0
        center_y = (y_start + y_end) / 2.0
        if center is not None:
            center_x, center_y = center
        half_width = max((x_max - x_min) * scale / 2.0, atlas_spacing_mm * 2)
        half_height = max(abs(y_end - y_start) * scale / 2.0, atlas_spacing_mm * 2)
        if y_start > y_end:
            y_bounds = (center_y + half_height, center_y - half_height)
        else:
            y_bounds = (center_y - half_height, center_y + half_height)
        set_view((
            center_x - half_width,
            center_x + half_width,
            y_bounds[0],
            y_bounds[1],
        ))

    def pan_view(x_fraction: float, y_fraction: float) -> None:
        x_min, x_max = axis.get_xlim()
        y_min, y_max = axis.get_ylim()
        x_shift = (x_max - x_min) * x_fraction
        y_shift = (y_max - y_min) * y_fraction
        set_view((
            x_min + x_shift, x_max + x_shift,
            y_min + y_shift, y_max + y_shift,
        ))

    def draw_region_panel() -> None:
        """List every Allen region that falls inside the outlined prism ROI.

        Drawn once on its own axis; draw_mode only clears the main image axis,
        so this side panel persists across all four standard views.
        """
        panel_axis.clear()
        panel_axis.set_facecolor("black")
        panel_axis.set_xlim(0.0, 1.0)
        panel_axis.set_ylim(0.0, 1.0)
        panel_axis.axis("off")
        panel_axis.add_patch(Rectangle(
            (0.01, 0.01), 0.98, 0.98, fill=False,
            edgecolor="#666666", linewidth=1.2,
        ))
        roi_regions = [row for row in face_summary if row["region_id"] > 0]
        panel_axis.text(
            0.5, 0.975, "Regions within prism ROI", ha="center", va="top",
            color="white", fontsize=11.5, fontweight="bold",
        )
        panel_axis.text(
            0.5, 0.94,
            f"{len(roi_regions)} region(s) · share of imaging-face area",
            ha="center", va="top", color="#bbbbbb", fontsize=8,
        )
        if not roi_regions:
            panel_axis.text(
                0.5, 0.5, "No annotated brain regions\nwithin the ROI.",
                ha="center", va="center", color="#999999", fontsize=9,
            )
            return
        top = 0.90
        row_height = 0.034
        available = max(1, int((top - 0.03) / row_height))
        shown = roi_regions[:available]
        for index, row in enumerate(shown):
            y = top - index * row_height
            panel_axis.add_patch(Rectangle(
                (0.035, y - 0.022), 0.055, 0.024,
                facecolor=ontology.get(row["region_id"]).rgb,
                edgecolor="white", linewidth=0.5,
            ))
            panel_axis.text(
                0.11, y, row["acronym"], ha="left", va="top",
                color="white", fontsize=8.6, fontweight="bold",
            )
            panel_axis.text(
                0.34, y - 0.001, f'ID {row["region_id"]}',
                ha="left", va="top", color="#cccccc", fontsize=7.6,
            )
            panel_axis.text(
                0.97, y, f'{row["percent_of_plane"]:.1f}%', ha="right", va="top",
                color="#ffe45c", fontsize=8.4, fontweight="bold",
            )
        if len(roi_regions) > available:
            panel_axis.text(
                0.5, top - available * row_height,
                f"+{len(roi_regions) - available} more not shown",
                ha="center", va="top", color="#999999", fontsize=7.6,
                fontstyle="italic",
            )

    def choose_two_photon_path() -> Path | None:
        existing = state.get("two_photon_image_path")
        if existing is not None and state.get("two_photon_image") is None:
            return Path(existing)
        set_status(
            "Step 1 of 3  \u2014  choose the UNZOOMED (wide) 2p image",
            1, "twop", flush=True,
        )
        print("Select the UNZOOMED (wide) 2p image...")
        selected = choose_image_file_interactively(
            "Select the UNZOOMED (wide) 2p image")
        if selected is not None:
            return selected
        if sys.platform == "darwin":
            print("2p mapping canceled: no image selected.")
            return None
        typed = input("Path to 2p JPG/PNG image: ").strip()
        if not typed:
            print("2p mapping canceled: no image path entered.")
            return None
        return Path(typed).expanduser().resolve()

    def choose_two_photon_zoom_path() -> Path | None:
        existing = state.get("two_photon_zoom_image_path")
        if existing is not None and state.get("two_photon_zoom_image") is None:
            return Path(existing)
        set_status(
            "Step 2 of 3  \u2014  choose the ZOOMED 2p image  "
            "(cancel to skip and use the unzoomed image alone)",
            2, "zoom", flush=True,
        )
        print(
            "Select the ZOOMED 2p image... "
            "(cancel to continue with the unzoomed image only)"
        )
        selected = choose_image_file_interactively(
            "Select the ZOOMED 2p image")
        if selected is None:
            print("No zoomed image selected; continuing with the "
                  "unzoomed image only.")
            return None
        return selected

    def prompt_two_photon_magnification() -> float | None:
        current = state.get("two_photon_magnification")
        default = float(current) if current else DEFAULT_ZOOM_MAGNIFICATION
        set_status(
            "Step 2 of 3  \u2014  enter the magnification of the zoomed "
            "image", 2, "zoom", flush=True,
        )
        return ask_number_interactively(
            "Enter magnification of zoomed image", default)

    def begin_two_photon_mapping() -> None:
        image_path = choose_two_photon_path()
        if image_path is None:
            return
        try:
            wide_image = load_two_photon_image(image_path)
        except (OSError, ValueError, FileNotFoundError) as error:
            print(f"Could not load 2p image: {error}")
            return
        print(
            f"Loaded unzoomed (wide) 2p image: {image_path}\n"
            f"  {describe_two_photon_image(wide_image)}"
        )
        state["two_photon_image_path"] = image_path
        state["two_photon_image"] = wide_image
        state["two_photon_zoom_warped_rgba"] = None
        state["two_photon_zoom_homography"] = None
        state["two_photon_zoom_to_wide"] = None
        state["two_photon_aligning"] = False
        state["two_photon_selecting"] = False

        zoom_path = choose_two_photon_zoom_path()
        if zoom_path is None:
            state["two_photon_zoom_image"] = None
            start_two_photon_corner_selection()
            return
        try:
            zoom_image = load_two_photon_image(zoom_path)
        except (OSError, ValueError, FileNotFoundError) as error:
            print(f"Could not load zoomed 2p image: {error}")
            print("Continuing with the unzoomed image only.")
            state["two_photon_zoom_image"] = None
            start_two_photon_corner_selection()
            return
        print(
            f"Loaded zoomed 2p image: {zoom_path}\n"
            f"  {describe_two_photon_image(zoom_image)}"
        )
        magnification = prompt_two_photon_magnification()
        if magnification is None:
            state["two_photon_zoom_image"] = None
            start_two_photon_corner_selection()
            return

        state["two_photon_zoom_image_path"] = zoom_path
        state["two_photon_zoom_image"] = zoom_image
        state["two_photon_magnification"] = float(magnification)
        state["two_photon_zoom_alpha"] = TWO_PHOTON_ZOOM_ALIGN_ALPHA
        state["two_photon_aligning"] = True
        state["two_photon_align_bounds"] = None
        state["two_photon_align_drag"] = None
        state["drag_start"] = None
        state["drag_limits"] = None
        center_two_photon_zoom()
        set_context_group("align")
        set_status(
            "Step 2 of 3  \u2014  drag the zoomed image onto the unzoomed "
            "image, then press Confirm", 2, "zoom",
        )
        try:
            zoom_alpha_slider.set_val(TWO_PHOTON_ZOOM_ALIGN_ALPHA)
        except NameError:
            pass
        display_width, display_height = zoom_display_size_px(
            magnification, wide_image.shape[:2]
        )
        print(
            f"\nZoom alignment active. At {magnification:g}x the zoomed image "
            f"covers {display_width:.1f} x {display_height:.1f} unzoomed "
            "pixels.\n"
            "  Drag it into place, or nudge with the arrow keys "
            "(Shift for 10 px steps).\n"
            "  The 'zoom alpha' slider fades it so you can see both layers.\n"
            "  Enter or 'Zoom Done' confirms, R re-centers, Escape cancels."
        )
        draw_two_photon_alignment_view(reset_view=True)
        refresh_live_view()

    draw_region_panel()
    draw_mode("in_plane", use_full=initial_view == "full")
    figure.savefig(
        output_path, dpi=260, facecolor=figure.get_facecolor(),
        bbox_inches="tight", pad_inches=0.02,
    )
    print(f"Saved plane visual: {output_path}")

    full_output_path = output_path.with_name(
        f"{output_path.stem}_full{output_path.suffix}"
    )
    set_view(state["bounds"]["in_plane"]["full"])
    figure.savefig(
        full_output_path, dpi=260, facecolor=figure.get_facecolor(),
        bbox_inches="tight", pad_inches=0.02,
    )
    print(f"Saved full-slice visual: {full_output_path}")

    for mode_name in ("coronal", "sagittal", "axial"):
        standard_output_path = output_path.with_name(
            f"{output_path.stem}_{mode_name}{output_path.suffix}"
        )
        draw_mode(mode_name, use_full=False)
        figure.savefig(
            standard_output_path, dpi=260, facecolor=figure.get_facecolor(),
            bbox_inches="tight", pad_inches=0.02,
        )
        print(f"Saved {mode_name} atlas slice: {standard_output_path}")
    draw_mode("in_plane", use_full=initial_view == "full")

    if show:
        def switch_mode(mode_name: str) -> None:
            if state["two_photon_selecting"]:
                state["two_photon_selecting"] = False
                state["two_photon_selection_points"] = []
                state["two_photon_selection_bounds"] = None
                state["two_photon_pan"] = False
                print("2p corner selection canceled by view switch.")
            set_context_group(None)
            set_status(
                "Ready. Map 2p starts the two-photon overlay.", None, "info")
            draw_mode(mode_name)

        def show_full_current_view() -> None:
            if state["two_photon_selecting"]:
                image = state["two_photon_image"]
                if image is not None:
                    height, width = image.shape[:2]
                    set_view((0.0, float(width), float(height), 0.0))
                return
            set_view(state["bounds"][state["mode"]]["full"])

        def reset_current_view() -> None:
            if state["two_photon_selecting"]:
                reset_two_photon_selection()
                return
            set_view(state["bounds"][state["mode"]]["close"])

        TOOLBAR_Y, TOOLBAR_H = 0.018, 0.048
        CONTEXT_Y, CONTEXT_H = 0.082, 0.042
        SLIDER_Y, SLIDER_H = 0.092, 0.020

        buttons: list[Any] = []

        def register(widget: Any, group: str | None = None) -> Any:
            buttons.append(widget)
            if group is not None:
                context_widgets[group].append(widget)
                widget.ax.set_visible(False)
                widget.set_active(False)
            return widget

        # --- primary toolbar: only what is always relevant ---------------
        view_buttons: dict[str, Button] = {}
        for text, mode_name, left, width in (
            ("In-plane", "in_plane", 0.014, 0.080),
            ("Coronal", "coronal", 0.101, 0.072),
            ("Sagittal", "sagittal", 0.180, 0.074),
            ("Axial", "axial", 0.261, 0.060),
        ):
            button = make_button(
                figure, (left, TOOLBAR_Y, width, TOOLBAR_H), text,
                kind="active" if mode_name == state["mode"] else "default",
            )
            button.on_clicked(lambda _event, name=mode_name: switch_mode(name))
            view_buttons[mode_name] = register(button)

        def sync_toggle_buttons() -> None:
            """Colour the toggles by state, so the label can stay constant."""
            style_button(
                color_button,
                "active" if state["show_atlas_colors"] else "default")
            style_button(
                label_button,
                "active" if state["show_labels"] else "default")
            style_button(
                fill_button,
                "active" if state["show_prism_fill"] else "default")
            style_button(
                two_photon_toggle_button,
                "twop" if state["show_two_photon"] else "default")
            style_button(
                zoom_toggle_button,
                "zoom" if state["show_two_photon_zoom"] else "default")
            figure.canvas.draw_idle()

        def refresh_view_buttons() -> None:
            for mode_name, view_button in view_buttons.items():
                style_button(
                    view_button,
                    "active" if state["mode"] == mode_name else "default",
                )

        for text, left, width, callback in (
            ("\u2212", 0.339, 0.036, lambda _event: zoom_view(1.25)),
            ("+", 0.382, 0.036, lambda _event: zoom_view(0.80)),
            ("Fit", 0.425, 0.048, lambda _event: show_full_current_view()),
        ):
            button = make_button(figure, (left, TOOLBAR_Y, width, TOOLBAR_H), text)
            button.on_clicked(callback)
            register(button)

        color_button = make_button(
            figure, (0.485, TOOLBAR_Y, 0.064, TOOLBAR_H), "Colors")

        def on_color_toggle(_event: Any) -> None:
            state["show_atlas_colors"] = not state["show_atlas_colors"]
            sync_toggle_buttons()
            if state["two_photon_selecting"] or state["two_photon_aligning"]:
                figure.canvas.draw_idle()
                return
            draw_mode(state["mode"])

        color_button.on_clicked(on_color_toggle)
        register(color_button)

        label_button = make_button(
            figure, (0.556, TOOLBAR_Y, 0.064, TOOLBAR_H), "Labels")

        def on_label_toggle(_event: Any) -> None:
            state["show_labels"] = not state["show_labels"]
            sync_toggle_buttons()
            if state["two_photon_selecting"] or state["two_photon_aligning"]:
                figure.canvas.draw_idle()
                return
            draw_mode(state["mode"])

        label_button.on_clicked(on_label_toggle)
        register(label_button)

        layers_button = make_button(
            figure, (0.627, TOOLBAR_Y, 0.062, TOOLBAR_H), "Layers")

        def on_layers_toggle(_event: Any) -> None:
            if state["two_photon_aligning"] or state["two_photon_selecting"]:
                print("Finish or cancel the current 2p step first.")
                return
            set_context_group(
                None if state["context_group"] == "layers" else "layers")

        layers_button.on_clicked(on_layers_toggle)
        register(layers_button)

        two_photon_button = make_button(
            figure, (0.755, TOOLBAR_Y, 0.080, TOOLBAR_H), "Map 2p",
            kind="twop")
        two_photon_button.on_clicked(lambda _event: begin_two_photon_mapping())
        register(two_photon_button)

        help_button = make_button(
            figure, (0.842, TOOLBAR_Y, 0.030, TOOLBAR_H), "?")
        help_button.on_clicked(lambda _event: print(CONTROLS_HELP))
        register(help_button)

        panel_button = make_button(
            figure, (0.879, TOOLBAR_Y, 0.052, TOOLBAR_H), "Panel")

        def on_panel_toggle(_event: Any) -> None:
            state["show_region_panel"] = not state["show_region_panel"]
            panel_button.label.set_text(
                "Panel" if state["show_region_panel"] else "Panel +")
            apply_viewer_layout()
            figure.canvas.draw_idle()

        panel_button.on_clicked(on_panel_toggle)
        register(panel_button)

        done_button = make_button(
            figure, (0.938, TOOLBAR_Y, 0.050, TOOLBAR_H), "Done")
        done_button.label.set_fontweight("bold")

        def on_done(_event: Any) -> None:
            if state["two_photon_aligning"]:
                print("Confirm or cancel the zoom alignment first.")
                return
            if state["two_photon_selecting"]:
                print("Finish or cancel 2p corner selection first.")
                return
            print("Finishing: closing the atlas mapper window.")
            plt.close(figure)

        done_button.on_clicked(on_done)
        register(done_button)

        # --- contextual row: alignment -----------------------------------
        confirm_button = make_button(
            figure, (0.014, CONTEXT_Y, 0.092, CONTEXT_H), "Confirm",
            kind="zoom")
        confirm_button.on_clicked(
            lambda _event: complete_two_photon_alignment()
            if state["two_photon_aligning"] else None)
        register(confirm_button, "align")

        recentre_button = make_button(
            figure, (0.113, CONTEXT_Y, 0.088, CONTEXT_H), "Re-centre")
        recentre_button.on_clicked(
            lambda _event: reset_two_photon_alignment()
            if state["two_photon_aligning"] else None)
        register(recentre_button, "align")

        magnification_button = make_button(
            figure, (0.208, CONTEXT_Y, 0.092, CONTEXT_H),
            f"Mag {state['two_photon_magnification']:g}x")

        def on_change_magnification(_event: Any) -> None:
            """Re-scale the zoom about its centre so it does not jump."""
            if not state["two_photon_aligning"]:
                return
            current = float(state["two_photon_magnification"])
            value = ask_number_interactively(
                "Enter magnification of zoomed image", current)
            if value is None or value <= 0.0 or value == current:
                return
            wide_shape = state["two_photon_image"].shape[:2]
            old_width, old_height = zoom_display_size_px(current, wide_shape)
            new_width, new_height = zoom_display_size_px(value, wide_shape)
            translate_x, translate_y = state["two_photon_zoom_translation"]
            state["two_photon_zoom_translation"] = (
                translate_x + (old_width - new_width) / 2.0,
                translate_y + (old_height - new_height) / 2.0,
            )
            state["two_photon_magnification"] = float(value)
            magnification_button.label.set_text(f"Mag {value:g}x")
            draw_two_photon_alignment_view(reset_view=False)

        magnification_button.on_clicked(on_change_magnification)
        register(magnification_button, "align")

        check_button = make_button(
            figure, (0.307, CONTEXT_Y, 0.070, CONTEXT_H), "Check")
        check_button.on_clicked(
            lambda _event: report_zoom_alignment_quality()
            if state["two_photon_aligning"] else None)
        register(check_button, "align")

        align_alpha_slider = make_slider(
            figure, (0.420, SLIDER_Y, 0.170, SLIDER_H), "zoom alpha",
            state["two_photon_zoom_alpha"], TWO_PHOTON_ZOOM_BUTTON_COLOR)

        # --- contextual row: corner selection ----------------------------
        two_photon_undo_button = make_button(
            figure, (0.014, CONTEXT_Y, 0.078, CONTEXT_H), "Undo", kind="twop")
        two_photon_undo_button.on_clicked(
            lambda _event: undo_two_photon_selection()
            if state["two_photon_selecting"] else None)
        register(two_photon_undo_button, "corners")

        two_photon_pan_button = make_button(
            figure, (0.099, CONTEXT_Y, 0.078, CONTEXT_H), "Pan off",
            kind="twop")

        def toggle_two_photon_pan() -> None:
            if not state["two_photon_selecting"]:
                print("2p pan is available during 2p corner selection.")
                return
            state["two_photon_pan"] = not state["two_photon_pan"]
            state["drag_start"] = None
            state["drag_limits"] = None
            two_photon_pan_button.label.set_text(
                "Pan on" if state["two_photon_pan"] else "Pan off")
            draw_two_photon_selection_view(reset_view=False)

        two_photon_pan_button.on_clicked(lambda _event: toggle_two_photon_pan())
        register(two_photon_pan_button, "corners")

        # --- contextual row: layers --------------------------------------
        fill_button = make_button(
            figure, (0.014, CONTEXT_Y, 0.060, CONTEXT_H), "Fill")

        def on_fill_toggle(_event: Any) -> None:
            state["show_prism_fill"] = not state["show_prism_fill"]
            sync_toggle_buttons()
            draw_mode(state["mode"])

        fill_button.on_clicked(on_fill_toggle)
        register(fill_button, "layers")

        two_photon_toggle_button = make_button(
            figure, (0.081, CONTEXT_Y, 0.070, CONTEXT_H), "2p", kind="twop")

        def on_two_photon_toggle(_event: Any) -> None:
            state["show_two_photon"] = not state["show_two_photon"]
            sync_toggle_buttons()
            draw_mode(state["mode"])

        two_photon_toggle_button.on_clicked(on_two_photon_toggle)
        register(two_photon_toggle_button, "layers")

        zoom_toggle_button = make_button(
            figure, (0.158, CONTEXT_Y, 0.078, CONTEXT_H), "Zoom",
            kind="zoom")

        def on_zoom_toggle(_event: Any) -> None:
            state["show_two_photon_zoom"] = not state["show_two_photon_zoom"]
            sync_toggle_buttons()
            draw_mode(state["mode"])

        zoom_toggle_button.on_clicked(on_zoom_toggle)
        register(zoom_toggle_button, "layers")

        opacity_slider = make_slider(
            figure, (0.330, SLIDER_Y, 0.140, SLIDER_H), "2p opacity",
            state["two_photon_opacity"], TWO_PHOTON_BUTTON_COLOR)

        def on_opacity_change(value: float) -> None:
            state["two_photon_opacity"] = float(value)
            if state["two_photon_selecting"] or state["two_photon_aligning"]:
                return
            if state["two_photon_warped_rgba"] is not None:
                draw_mode(state["mode"])

        opacity_slider.on_changed(on_opacity_change)
        register(opacity_slider, "layers")

        zoom_alpha_slider = make_slider(
            figure, (0.560, SLIDER_Y, 0.140, SLIDER_H), "zoom alpha",
            state["two_photon_zoom_alpha"], TWO_PHOTON_ZOOM_BUTTON_COLOR)

        def on_zoom_alpha_change(value: float) -> None:
            state["two_photon_zoom_alpha"] = float(value)
            if state["two_photon_aligning"]:
                align_alpha_slider.eventson = False
                align_alpha_slider.set_val(value)
                align_alpha_slider.eventson = True
                draw_two_photon_alignment_view(reset_view=False)
                return
            if state["two_photon_selecting"]:
                return
            if state["two_photon_zoom_warped_rgba"] is not None:
                draw_mode(state["mode"])

        zoom_alpha_slider.on_changed(on_zoom_alpha_change)
        register(zoom_alpha_slider, "layers")
        align_alpha_slider.on_changed(on_zoom_alpha_change)
        register(align_alpha_slider, "align")

        # --- contextual row: microCT comparison --------------------------
        compare_button = None
        microct_pan_button = None
        if microct_available:
            compare_button = make_button(
                figure, (0.696, TOOLBAR_Y, 0.052, TOOLBAR_H), "uCT")

            def on_compare(_event: Any) -> None:
                if state["two_photon_selecting"] or state["two_photon_aligning"]:
                    print("Finish or cancel the current 2p step first.")
                    return
                state["compare_microct"] = not state["compare_microct"]
                style_button(
                    compare_button,
                    "active" if state["compare_microct"] else "default")
                set_context_group("uct" if state["compare_microct"] else None)
                if state["compare_microct"] and state["mode"] == "in_plane":
                    draw_mode(nearest_mode)
                    return
                draw_mode(state["mode"])

            compare_button.on_clicked(on_compare)
            register(compare_button)

            def ensure_microct_comparison() -> bool:
                if not state["compare_microct"]:
                    state["compare_microct"] = True
                    style_button(compare_button, "active")
                    set_context_group("uct")
                    draw_mode(
                        nearest_mode if state["mode"] == "in_plane"
                        else state["mode"])
                return comparison_active()

            for text, left, width, callback in (
                ("uCT +", 0.014, 0.060,
                 lambda _event: ensure_microct_comparison() and zoom_microct_view(0.80)),
                ("uCT \u2212", 0.081, 0.060,
                 lambda _event: ensure_microct_comparison() and zoom_microct_view(1.25)),
                ("uCT 1x", 0.148, 0.064,
                 lambda _event: ensure_microct_comparison() and reset_microct_view()),
            ):
                button = make_button(figure, (left, CONTEXT_Y, width, CONTEXT_H), text)
                button.on_clicked(callback)
                register(button, "uct")

            microct_pan_button = make_button(
                figure, (0.219, CONTEXT_Y, 0.082, CONTEXT_H), "uCT Pan off")

            def on_microct_pan(_event: Any) -> None:
                if ensure_microct_comparison():
                    state["microct_pan"] = not state["microct_pan"]
                    microct_pan_button.label.set_text(
                        "uCT Pan on" if state["microct_pan"] else "uCT Pan off")
                    state["microct_drag_start"] = None
                    state["microct_drag_limits"] = None
                    figure.canvas.draw_idle()

            microct_pan_button.on_clicked(on_microct_pan)
            register(microct_pan_button, "uct")
        else:
            def ensure_microct_comparison() -> bool:
                return False

            print("MicroCT comparison disabled: no source microCT image was found.")

        sync_toggle_buttons()
        set_context_group(None)
        set_status(
            "Ready. Map 2p starts the two-photon overlay.  "
            "Done or Q finishes.", None, "info")

        def on_scroll(event: Any) -> None:
            if event.inaxes is microct_axis and comparison_active():
                change_microct_slice(1 if event.button == "up" else -1)
                return
            if event.inaxes is not axis:
                return
            center = None
            if event.xdata is not None and event.ydata is not None:
                center = (float(event.xdata), float(event.ydata))
            zoom_view(0.80 if event.button == "up" else 1.25, center)

        def on_press(event: Any) -> None:
            if state["two_photon_aligning"]:
                if (
                    event.inaxes is axis and event.button == 1
                    and event.xdata is not None and event.ydata is not None
                ):
                    translate_x, translate_y = state[
                        "two_photon_zoom_translation"
                    ]
                    state["two_photon_align_drag"] = (
                        float(event.xdata), float(event.ydata),
                        translate_x, translate_y,
                    )
                return
            if state["two_photon_selecting"]:
                if event.inaxes is axis and event.button == 1:
                    if event.xdata is not None and event.ydata is not None:
                        if state["two_photon_pan"]:
                            state["drag_start"] = (event.xdata, event.ydata)
                            state["drag_limits"] = (axis.get_xlim(), axis.get_ylim())
                        else:
                            record_two_photon_selection_point(event.xdata, event.ydata)
                return
            if (
                event.inaxes is microct_axis and event.button == 1
                and comparison_active() and state["microct_pan"]
                and event.xdata is not None and event.ydata is not None
            ):
                state["microct_drag_start"] = (event.xdata, event.ydata)
                state["microct_drag_limits"] = (
                    microct_axis.get_xlim(), microct_axis.get_ylim()
                )
                return
            if event.inaxes is not axis or event.button != 1:
                return
            state["drag_start"] = (event.xdata, event.ydata)
            state["drag_limits"] = (axis.get_xlim(), axis.get_ylim())

        def on_motion(event: Any) -> None:
            if state["two_photon_aligning"]:
                drag = state["two_photon_align_drag"]
                if (
                    drag is None or event.inaxes is not axis
                    or event.xdata is None or event.ydata is None
                ):
                    return
                press_x, press_y, origin_x, origin_y = drag
                state["two_photon_zoom_translation"] = (
                    origin_x + (float(event.xdata) - press_x),
                    origin_y + (float(event.ydata) - press_y),
                )
                draw_two_photon_alignment_view(reset_view=False)
                return
            if state["two_photon_selecting"]:
                if (
                    not state["two_photon_pan"]
                    or state["drag_start"] is None
                    or event.inaxes is not axis
                    or event.xdata is None or event.ydata is None
                ):
                    return
                start_x, start_y = state["drag_start"]
                x_limits, y_limits = state["drag_limits"]
                delta_x = event.xdata - start_x
                delta_y = event.ydata - start_y
                set_view((
                    x_limits[0] - delta_x, x_limits[1] - delta_x,
                    y_limits[0] - delta_y, y_limits[1] - delta_y,
                ))
                return
            if (
                state["microct_drag_start"] is not None
                and event.inaxes is microct_axis
                and event.xdata is not None and event.ydata is not None
            ):
                start_x, start_y = state["microct_drag_start"]
                x_limits, y_limits = state["microct_drag_limits"]
                microct_axis.set_xlim(*x_limits)
                microct_axis.set_ylim(*y_limits)
                pan_microct_view(event.xdata - start_x, event.ydata - start_y)
                return
            if (
                state["drag_start"] is None or event.inaxes is not axis
                or event.xdata is None or event.ydata is None
            ):
                return
            start_x, start_y = state["drag_start"]
            x_limits, y_limits = state["drag_limits"]
            delta_x = event.xdata - start_x
            delta_y = event.ydata - start_y
            set_view((
                x_limits[0] - delta_x, x_limits[1] - delta_x,
                y_limits[0] - delta_y, y_limits[1] - delta_y,
            ))

        def on_release(_event: Any) -> None:
            state["two_photon_align_drag"] = None
            state["drag_start"] = None
            state["drag_limits"] = None
            state["microct_drag_start"] = None
            state["microct_drag_limits"] = None

        def on_close(_event: Any) -> None:
            state["two_photon_aligning"] = False
            state["two_photon_align_drag"] = None
            state["two_photon_selecting"] = False
            state["two_photon_pan"] = False
            state["drag_start"] = None
            state["drag_limits"] = None
            state["microct_drag_start"] = None
            state["microct_drag_limits"] = None
            print("Atlas mapper visualization closed cleanly.")

        def on_key(event: Any) -> None:
            if state["two_photon_aligning"]:
                nudges = {
                    "left": (-1.0, 0.0), "right": (1.0, 0.0),
                    "up": (0.0, -1.0), "down": (0.0, 1.0),
                }
                coarse = {
                    "shift+left": (-10.0, 0.0), "shift+right": (10.0, 0.0),
                    "shift+up": (0.0, -10.0), "shift+down": (0.0, 10.0),
                }
                if event.key in nudges:
                    nudge_two_photon_zoom(*nudges[event.key])
                elif event.key in coarse:
                    nudge_two_photon_zoom(*coarse[event.key])
                elif event.key in ("enter", "return"):
                    complete_two_photon_alignment()
                elif event.key in ("r", "home"):
                    reset_two_photon_alignment()
                elif event.key == "escape":
                    cancel_two_photon_alignment()
                return
            if event.key in ("escape", "q"):
                if state["two_photon_selecting"]:
                    cancel_two_photon_selection()
                else:
                    print("Closing atlas mapper visualization.")
                    plt.close(figure)
            elif event.key in ("z", "backspace", "delete") and state["two_photon_selecting"]:
                undo_two_photon_selection()
            elif event.key == "p" and state["two_photon_selecting"]:
                toggle_two_photon_pan()
            elif event.key in ("+", "="):
                zoom_view(0.80)
            elif event.key in ("-", "_"):
                zoom_view(1.25)
            elif event.key == "0":
                show_full_current_view()
            elif event.key in ("r", "home"):
                reset_current_view()
            elif event.key == "i":
                switch_mode("in_plane")
            elif event.key == "c":
                switch_mode("coronal")
            elif event.key == "s":
                switch_mode("sagittal")
            elif event.key == "a":
                switch_mode("axial")
            elif event.key == "g":
                state["show_atlas_colors"] = not state["show_atlas_colors"]
                sync_toggle_buttons()
                if state["two_photon_selecting"]:
                    draw_two_photon_selection_view(reset_view=False)
                    return
                draw_mode(state["mode"])
            elif event.key == "l":
                state["show_labels"] = not state["show_labels"]
                sync_toggle_buttons()
                if state["two_photon_selecting"]:
                    draw_two_photon_selection_view(reset_view=False)
                    return
                draw_mode(state["mode"])
            elif event.key == "f":
                state["show_prism_fill"] = not state["show_prism_fill"]
                sync_toggle_buttons()
                if state["two_photon_selecting"]:
                    draw_two_photon_selection_view(reset_view=False)
                    return
                draw_mode(state["mode"])
            elif event.key == "t":
                if state["two_photon_selecting"]:
                    print("Finish or cancel 2p corner selection before toggling the overlay.")
                    return
                state["show_two_photon"] = not state["show_two_photon"]
                sync_toggle_buttons()
                draw_mode(state["mode"])
            elif event.key == "m" and microct_available:
                if state["two_photon_selecting"]:
                    print("Finish or cancel 2p corner selection before comparing uCT.")
                    return
                state["compare_microct"] = not state["compare_microct"]
                if compare_button is not None:
                    compare_button.label.set_text(
                        "Atlas only" if state["compare_microct"]
                        else "Compare uCT"
                    )
                if state["compare_microct"] and state["mode"] == "in_plane":
                    draw_mode(nearest_mode)
                else:
                    draw_mode(state["mode"])
            elif event.key == "p" and microct_available:
                if ensure_microct_comparison() and microct_pan_button is not None:
                    state["microct_pan"] = not state["microct_pan"]
                    microct_pan_button.label.set_text(
                        "uCT Pan on" if state["microct_pan"] else "uCT Pan off"
                    )
                    state["microct_drag_start"] = None
                    state["microct_drag_limits"] = None
            elif event.key == "left":
                pan_view(0.15, 0.0)
            elif event.key == "right":
                pan_view(-0.15, 0.0)
            elif event.key == "up":
                pan_view(0.0, -0.15)
            elif event.key == "down":
                pan_view(0.0, 0.15)

        figure.canvas.mpl_connect("scroll_event", on_scroll)
        figure.canvas.mpl_connect("button_press_event", on_press)
        figure.canvas.mpl_connect("motion_notify_event", on_motion)
        figure.canvas.mpl_connect("button_release_event", on_release)
        figure.canvas.mpl_connect("key_press_event", on_key)
        figure.canvas.mpl_connect("close_event", on_close)
        figure._microprism_buttons = buttons
        print(CONTROLS_HELP)
        plt.show()
    else:
        plt.close(figure)


def write_outputs(
    output_dir: Path,
    corners_source: dict[str, np.ndarray],
    corners_ccf: dict[str, np.ndarray],
    rectified_corners_ccf: dict[str, np.ndarray],
    labels: np.ndarray,
    face_mask: np.ndarray,
    full_plane_summary: list[dict[str, Any]],
    face_summary: list[dict[str, Any]],
    corner_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "microprism_atlas_regions.csv"
    json_path = output_dir / "microprism_atlas_regions.json"
    labels_path = output_dir / "microprism_atlas_plane_labels.npy"
    mask_path = output_dir / "microprism_atlas_face_mask.npy"

    fieldnames = [
        "scope", "region_id", "acronym", "name", "color_hex",
        "pixel_count", "percent_of_plane", "percent_of_brain_area",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for scope, rows in (
            ("full_atlas_plane", full_plane_summary),
            ("prism_face", face_summary),
        ):
            for row in rows:
                writer.writerow({"scope": scope, **row})

    payload = {
        **metadata,
        "source_corners_xyz_voxel": {
            key: value.tolist() for key, value in corners_source.items()
        },
        "ccf_corners_xyz_voxel": {
            key: value.tolist() for key, value in corners_ccf.items()
        },
        "rectified_face_corners_xyz_voxel": {
            key: value.tolist() for key, value in rectified_corners_ccf.items()
        },
        "corner_regions": corner_rows,
        "full_plane_region_summary": full_plane_summary,
        "prism_face_region_summary": face_summary,
    }
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    np.save(labels_path, labels)
    np.save(mask_path, face_mask)
    return summary_path, json_path, labels_path, mask_path


def print_summary(
    corner_rows: list[dict[str, Any]],
    face_summary: list[dict[str, Any]],
) -> None:
    print("\n" + "=" * 84)
    print("REGIONS AT THE FOUR MICROPRISM FACE CORNERS")
    print("=" * 84)
    print(f"{'Corner':<16}{'CCF XYZ voxel':<28}{'Acronym':<14}Region ID")
    print("-" * 84)
    for row in corner_rows:
        coordinate = (
            f'({row["x_ccf"]:.1f}, {row["y_ccf"]:.1f}, {row["z_ccf"]:.1f})'
        )
        print(
            f'{row["corner"]:<16}{coordinate:<28}'
            f'{row["acronym"]:<14}{row["region_id"]}'
        )
    print("\nREGION COVERAGE WITHIN THE OUTLINED PRISM FACE")
    print(f"{'Acronym':<14}{'% face':>10}  Region ID")
    print("-" * 84)
    for row in face_summary:
        if row["percent_of_plane"] >= 0.05:
            print(
                f'{row["acronym"]:<14}{row["percent_of_plane"]:>9.2f}%  '
                f'{row["region_id"]}'
            )
    print("=" * 84)


def print_validation_note(microct_path: Path | None, corners_in_bounds: int) -> None:
    print("\n" + "=" * 84)
    print("VALIDATION CHECKLIST")
    print("=" * 84)
    if microct_path is not None:
        print(f"Compare the atlas plane against the source microCT: {microct_path}")
        print(
            "Use Compare uCT, uCT +/-/1x, uCT Pan, and microCT scrolling to "
            "check that the prism face sits where it should in each standard view."
        )
    else:
        print(
            "No source microCT was loaded for comparison. Re-run with --microct "
            "or confirm that microprism_corners.json contains a valid source_image."
        )
    print(f"Transformed prism corners inside Allen bounds: {corners_in_bounds}/4")
    print("If the atlas plane looks wrong, likely things to re-check:")
    print("  1. Landmark registration quality and landmark pair ordering.")
    print("  2. Whether the transform is stale or from a different microCT image.")
    print("  3. MicroCT voxel spacing, especially 0.072 mm vs 0.025 mm atlas spacing.")
    print("  4. Transform direction: fixed-to-moving should be inverted for mapping to CCF.")
    print("  5. Axis orientation and whether corners were picked on the aligned/corrected TIFF.")
    print("  6. Prism corner selections and entered prism width/length dimensions.")
    print("Suggested retry path: rerun midline alignment if orientation looks off, rerun")
    print("landmark_registration.py if the transform looks stale/bad, then rerun")
    print("microprism_corner_tracker.py and microprism_atlas_mapper.py.")
    print("=" * 84)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the complete Allen CCF plane corresponding to a tracked "
            "microprism face and identify all four corner regions."
        )
    )
    parser.add_argument("project_path", nargs="?")
    parser.add_argument("--corners", help="Path to microprism_corners.json.")
    parser.add_argument("--annotation", help="Allen annotation NRRD, TIFF, or NPY.")
    parser.add_argument(
        "--template",
        help="Allen average-template NRRD, TIFF, or NPY for anatomical detail.",
    )
    parser.add_argument("--ontology", help="Allen structure_tree.json.")
    parser.add_argument(
        "--no-template",
        action="store_true",
        help="Show annotation colors only and do not download the Allen template.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not automatically download missing shared Allen CCF resources.",
    )
    parser.add_argument("--transform", help="SimpleITK registration transform.")
    parser.add_argument(
        "--microct",
        help=(
            "MicroCT TIFF/NRRD/NPY to show beside atlas standard views. "
            "Defaults to the source_image recorded by microprism_corner_tracker."
        ),
    )
    parser.add_argument(
        "--two-photon-image",
        help=(
            "Optional unzoomed (wide) JPG/PNG 2p image to map onto the "
            "in-plane prism view. If omitted, the 2p Mapper button "
            "prompts for a path."
        ),
    )
    parser.add_argument(
        "--two-photon-zoom-image",
        help=(
            "Optional zoomed 2p JPG/PNG to scale and align on top of the "
            "unzoomed image before both are mapped onto the prism plane."
        ),
    )
    parser.add_argument(
        "--two-photon-magnification",
        type=float,
        default=DEFAULT_ZOOM_MAGNIFICATION,
        help=(
            "Magnification of the zoomed 2p image relative to the "
            "unzoomed one. At 2 the zoom covers half the unzoomed field "
            "of view in each direction."
        ),
    )
    parser.add_argument(
        "--two-photon-render-scale",
        type=float,
        default=1.0,
        help=(
            "Sampling density for the mapped 2p layers, relative to the "
            "source images. 1.0 keeps their native resolution; 2.0 "
            "oversamples. Capped at "
            f"{MAX_TWO_PHOTON_RENDER_PX} pixels per side."
        ),
    )
    parser.add_argument(
        "--two-photon-zoom-opacity",
        type=float,
        default=TWO_PHOTON_ZOOM_OVERLAY_ALPHA,
        help="Initial opacity for the mapped zoom layer, from 0 to 1.",
    )
    parser.add_argument(
        "--two-photon-opacity",
        type=float,
        default=TWO_PHOTON_OVERLAY_ALPHA,
        help="Initial opacity for the mapped 2p texture overlay, from 0 to 1.",
    )
    parser.add_argument("--output", help="Output directory. Defaults to project/outputs.")
    parser.add_argument(
        "--coordinate-space",
        choices=("auto", "registered", "aligned"),
        default="auto",
    )
    parser.add_argument(
        "--saved-transform-maps",
        choices=("auto", "fixed-to-moving", "moving-to-fixed"),
        default="auto",
        help=(
            "Direction of the saved transform. Auto tests both directions "
            "against the annotation bounds."
        ),
    )
    parser.add_argument(
        "--source-spacing", type=float,
        help="Fallback source voxel size in mm. Tracker output is used by default.",
    )
    parser.add_argument(
        "--source-spacing-x", type=float,
        help="Source voxel size along X in mm. Overrides tracker/default X spacing.",
    )
    parser.add_argument(
        "--source-spacing-y", type=float,
        help="Source voxel size along Y in mm. Overrides tracker/default Y spacing.",
    )
    parser.add_argument(
        "--source-spacing-z", type=float,
        help="Source voxel size along Z in mm. Overrides tracker/default Z spacing.",
    )
    parser.add_argument(
        "--atlas-spacing", type=float, default=DEFAULT_SPACING_MM,
        help="Fallback Allen atlas voxel size in mm. annotation_25 uses 0.025.",
    )
    parser.add_argument(
        "--atlas-spacing-x", type=float,
        help="Allen atlas voxel size along X in mm. Overrides --atlas-spacing for X.",
    )
    parser.add_argument(
        "--atlas-spacing-y", type=float,
        help="Allen atlas voxel size along Y in mm. Overrides --atlas-spacing for Y.",
    )
    parser.add_argument(
        "--atlas-spacing-z", type=float,
        help="Allen atlas voxel size along Z in mm. Overrides --atlas-spacing for Z.",
    )
    parser.add_argument(
        "--max-plane-samples", type=int, default=1400,
        help="Maximum pixels along the longest axis of the full atlas plane.",
    )
    parser.add_argument(
        "--plane-padding", type=float, default=0.02,
        help="Fractional padding around the atlas/plane intersection.",
    )
    parser.add_argument(
        "--initial-view",
        choices=("prism", "full"),
        default="prism",
        help="Open centered on the prism face or on the complete atlas slice.",
    )
    parser.add_argument(
        "--zoom-context-mm",
        type=float,
        default=1.0,
        help="Minimum width and height of the initial prism-centered view.",
    )
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = resolve_project_path(args.project_path)
    output_dir = (
        Path(args.output).expanduser().resolve()
        if args.output else project / "outputs"
    )
    corners_path = resolve_input(
        args.corners, project, ("microprism_corners.json",),
        "microprism tracker output",
    )
    annotation_path = resolve_input(
        args.annotation, project,
        ("annotation_25.nrrd", "annotation_25.tif", "annotation_25.npy"),
        "Allen CCF annotation", required=False,
    )
    ontology_path = resolve_input(
        args.ontology, project, ("structure_tree.json",),
        "Allen structure tree", required=False,
    )
    template_path = resolve_input(
        args.template, project,
        ("average_template_25.nrrd", "average_template_25.tif",
         "average_template_25.npy"),
        "Allen average template", required=False,
    )
    if annotation_path is None or ontology_path is None:
        shared_annotation, shared_ontology = ensure_allen_ccf_resources(
            allow_download=not args.no_download
        )
        annotation_path = annotation_path or shared_annotation
        ontology_path = ontology_path or shared_ontology
    if template_path is None and not args.no_template:
        template_path = ensure_allen_template(
            allow_download=not args.no_download
        )

    print("=" * 84)
    print("MICROPRISM ATLAS PLANE MAPPER")
    print("=" * 84)
    print(f"Tracker corners: {corners_path}")
    print(f"CCF annotation: {annotation_path}")
    print(f"CCF template: {template_path or 'disabled'}")
    print(f"Structure tree: {ontology_path}")

    corners_source, tracker_data = load_face_corners(corners_path)
    print("Using imaging-face corners: P1, P2, dL, dR (derived top points).")
    annotation = load_annotation(annotation_path)
    template = load_annotation(template_path) if template_path else None
    if template is not None and template.shape != annotation.shape:
        raise ValueError(
            "Allen template and annotation shapes must match: "
            f"{template.shape} != {annotation.shape}"
        )
    microct_path = None
    microct_image = None
    if not args.no_show:
        microct_path = resolve_microct_for_comparison(
            args.microct, tracker_data, project
        )
        if microct_path is not None:
            print(f"MicroCT comparison image: {microct_path}")
            microct_image = load_volume(microct_path, "MicroCT comparison image")
        else:
                print("MicroCT comparison image: not found")
    ontology = AllenOntology(ontology_path)
    two_photon_image_path = None
    if args.two_photon_image:
        candidate = Path(args.two_photon_image).expanduser()
        if not candidate.is_absolute():
            project_candidate = project / candidate
            candidate = project_candidate if project_candidate.exists() else candidate
        two_photon_image_path = candidate.resolve()
    two_photon_zoom_image_path = None
    if args.two_photon_zoom_image:
        candidate = Path(args.two_photon_zoom_image).expanduser()
        if not candidate.is_absolute():
            project_candidate = project / candidate
            candidate = (
                project_candidate if project_candidate.exists() else candidate
            )
        two_photon_zoom_image_path = candidate.resolve()
    coordinate_space = infer_coordinate_space(
        args.coordinate_space, tracker_data, annotation.shape
    )
    transform_path = resolve_input(
        args.transform, project,
        ("transform_refined.tfm", "transform_landmark.tfm"),
        "registration transform", required=False,
    )
    registration_metrics = load_registration_metrics(transform_path)
    source_spacing_xyz = spacing_xyz(
        args.source_spacing,
        args.source_spacing_x,
        args.source_spacing_y,
        args.source_spacing_z,
        tracker_spacing_xyz(tracker_data),
    )
    atlas_spacing_xyz = spacing_xyz(
        args.atlas_spacing,
        args.atlas_spacing_x,
        args.atlas_spacing_y,
        args.atlas_spacing_z,
        DEFAULT_SPACING_XYZ_MM,
    )
    report_face_geometry(
        "Source microCT JSON", corners_source, source_spacing_xyz
    )
    saved_transform_maps = args.saved_transform_maps
    if saved_transform_maps == "auto":
        metadata_direction = metadata_transform_direction(registration_metrics)
        if metadata_direction is not None:
            saved_transform_maps = metadata_direction
            print(
                "Using transform direction from registration metrics: "
                f"{saved_transform_maps}."
            )
    warn_transform_provenance(
        transform_path, project, registration_metrics, source_spacing_xyz
    )
    atlas_spacing_for_display = float(np.mean(atlas_spacing_xyz))
    corners_ccf, selected_transform_direction = transform_corners_to_ccf(
        corners_source,
        coordinate_space,
        transform_path,
        source_spacing_xyz,
        atlas_spacing_xyz,
        saved_transform_maps,
        annotation.shape,
    )
    corners_in_bounds = count_points_in_bounds(corners_ccf, annotation.shape)
    if corners_in_bounds == 0:
        print(
            "\nWARNING: all transformed prism corners are outside the Allen "
            "annotation volume. The displayed plane is unlikely to be meaningful."
        )
        print(
            "Likely causes: stale/bad landmark transform, wrong transform "
            "direction, wrong source spacing, or tracking corners from a "
            "different image than the registration transform."
        )
    print_corner_coordinate_table(
        corners_source, corners_ccf, source_spacing_xyz, atlas_spacing_xyz
    )
    print_axis_projection_guide()

    prism_width_mm = float(tracker_data.get("prism_width_mm", 0.0))
    prism_length_mm = float(tracker_data.get("prism_length_mm", 0.0))
    target_face_ratio = (
        prism_length_mm / prism_width_mm
        if prism_width_mm > 0 and prism_length_mm > 0
        else None
    )
    geometry = build_plane_geometry(corners_ccf, target_face_ratio)
    rectified_corners_ccf = rectified_face_corners_xyz(geometry)
    measured_geometry = with_face_uv(
        geometry, measured_face_uv_from_corners(geometry, corners_ccf)
    )
    report_face_geometry(
        "Transformed CCF", corners_ccf, atlas_spacing_xyz
    )
    report_face_geometry(
        "Rectified display CCF", rectified_corners_ccf, atlas_spacing_xyz
    )
    plane_xyz, extent_uv = full_plane_grid(
        geometry, annotation.shape,
        args.max_plane_samples, args.plane_padding,
    )
    labels = sample_annotation(annotation, plane_xyz)
    plane_template = (
        sample_template(template, plane_xyz) if template is not None else None
    )
    face_mask = face_mask_for_grid(measured_geometry, plane_xyz)
    full_plane_summary = region_summary(labels, ontology)
    face_summary = region_summary(labels[face_mask], ontology)
    corner_rows = corner_region_rows(
        annotation, corners_ccf, ontology
    )
    diagnostic_3d_path = output_dir / "microprism_corner_geometry_3d.png"
    save_corner_diagnostic_3d(
        diagnostic_3d_path,
        corners_source,
        corners_ccf,
        source_spacing_xyz,
        atlas_spacing_xyz,
    )

    metadata = {
        "tracker_output": str(corners_path),
        "annotation": str(annotation_path),
        "template": str(template_path) if template_path else None,
        "ontology": str(ontology_path),
        "coordinate_space": coordinate_space,
        "transform": str(transform_path) if transform_path else None,
        "saved_transform_maps": saved_transform_maps,
        "selected_transform_direction": selected_transform_direction,
        "source_spacing_mm": float(np.mean(source_spacing_xyz)),
        "source_spacing_mm_xyz": source_spacing_xyz.tolist(),
        "atlas_spacing_mm": atlas_spacing_for_display,
        "atlas_spacing_mm_xyz": atlas_spacing_xyz.tolist(),
        "annotation_shape_zyx": list(annotation.shape),
        "plane_grid_shape_rows_columns": list(labels.shape),
        "plane_extent_uv_voxels": list(extent_uv),
        "plane_origin_xyz_voxel": geometry.origin_xyz.tolist(),
        "plane_width_unit_xyz": geometry.width_unit_xyz.tolist(),
        "plane_length_unit_xyz": geometry.length_unit_xyz.tolist(),
        "plane_normal_unit_xyz": geometry.normal_unit_xyz.tolist(),
        "corner_plane_fit_error_voxels": geometry.fit_error_voxels,
        "display_face_length_to_width_ratio": target_face_ratio,
        "display_corner_source": (
            "actual transformed P1/P2/D1/D2 points; fitted plane is used only "
            "as the 2D sampling coordinate system"
        ),
        "view_projection_convention": {
            "stored_corner_order": "XYZ voxel coordinates; volume arrays are Z,Y,X",
            "coronal": {
                "slice_index": "Z / AP",
                "plot_x": "X / ML",
                "plot_y": "Y / DV, displayed with negative sign so dorsal is up",
            },
            "sagittal": {
                "slice_index": "X / ML",
                "plot_x": "Z / AP",
                "plot_y": "Y / DV, displayed with negative sign so dorsal is up",
            },
            "axial": {
                "slice_index": "Y / DV",
                "plot_x": "Z / AP",
                "plot_y": "X / ML, displayed with negative sign",
            },
        },
        "plane_method": (
            "Fit perpendicular width/length axes to all four face corners; "
            "intersect the infinite fitted plane with the annotation-volume "
            "box; sample the complete intersection; and use the actual "
            "transformed P1/P2/D1/D2 quadrilateral as the prism ROI."
        ),
    }
    summary_path, json_path, labels_path, mask_path = write_outputs(
        output_dir, corners_source, corners_ccf, rectified_corners_ccf,
        labels, face_mask,
        full_plane_summary, face_summary, corner_rows, metadata,
    )
    visual_path = output_dir / "microprism_atlas_plane.png"
    render_plane_map(
        annotation, labels, template, plane_template, microct_image, microct_path,
        ontology, corner_rows, face_summary, corners_source,
        corners_ccf, measured_geometry, extent_uv,
        atlas_spacing_for_display, source_spacing_xyz, visual_path, not args.no_show,
        args.initial_view, max(float(args.zoom_context_mm), 0.1),
        two_photon_image_path,
        float(np.clip(args.two_photon_opacity, 0.0, 1.0)),
        two_photon_zoom_image_path,
        float(args.two_photon_magnification),
        float(np.clip(args.two_photon_zoom_opacity, 0.0, 1.0)),
        max(float(args.two_photon_render_scale), 0.05),
    )
    print_summary(corner_rows, face_summary)
    print(f"CSV summary: {summary_path}")
    print(f"JSON details: {json_path}")
    print(f"Full-plane labels: {labels_path}")
    print(f"Prism face mask: {mask_path}")
    print(f"3D corner geometry diagnostic: {diagnostic_3d_path}")
    print_validation_note(microct_path, corners_in_bounds)


if __name__ == "__main__":
    main()
