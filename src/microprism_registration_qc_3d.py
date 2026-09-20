"""
3D validation viewer for microprism-to-Allen registration.

This follows the same practical pattern as the Sakata-style
fiber_visualizer_3d.py:

1. Load a registered volume that already lives in Allen/CCF space.
2. Display orthogonal image slices as surfaces inside a 3D coordinate system.
3. Draw the tracked object on top of those surfaces.
4. Add sliders/toggles so the user can rotate the 3D scene and inspect slices.

For the original multifiber pipeline, the tracked objects are fibers. For the
microprism pipeline, the tracked object is the microprism imaging face.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from allen_ccf_resources import (
    ensure_allen_ccf_resources,
    ensure_allen_template,
)
from microprism_atlas_mapper import (
    DEFAULT_SPACING_MM,
    DEFAULT_SPACING_XYZ_MM,
    build_plane_geometry,
    count_points_in_bounds,
    infer_coordinate_space,
    load_annotation,
    load_face_corners,
    load_registration_metrics,
    load_volume,
    metadata_transform_direction,
    rectified_face_corners_xyz,
    resolve_input,
    spacing_xyz,
    tracker_spacing_xyz,
    transform_corners_to_ccf,
    warn_transform_provenance,
)
from project_paths import resolve_project_path


FACE_KEYS = ("bottom_left", "bottom_right", "top_right", "top_left")
FACE_LABELS = {
    "bottom_left": "P1",
    "bottom_right": "P2",
    "top_left": "dL",
    "top_right": "dR",
}


def robust_normalize(values: np.ndarray) -> np.ndarray:
    """Normalize image data while ignoring very bright CT artifacts."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=float)
    low, high = np.percentile(finite, (1, 99.5))
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros(values.shape, dtype=float)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def atlas_microct_facecolors(
    atlas_slice: np.ndarray,
    microct_slice: np.ndarray | None,
    microct_alpha: float,
) -> np.ndarray:
    """Return RGB colors for a 3D slice surface.

    Allen template is grayscale. Registered microCT is added in green so that
    mismatches are visible while both volumes occupy the same 3D plane.
    """
    atlas = robust_normalize(atlas_slice)
    rgb = np.dstack((atlas, atlas, atlas))
    if microct_slice is None:
        return rgb
    micro = robust_normalize(microct_slice)
    rgb[..., 0] = np.clip(rgb[..., 0] * (1.0 - microct_alpha * micro), 0.0, 1.0)
    rgb[..., 1] = np.clip(rgb[..., 1] + microct_alpha * micro, 0.0, 1.0)
    rgb[..., 2] = np.clip(rgb[..., 2] * (1.0 - microct_alpha * micro), 0.0, 1.0)
    return rgb


def downsample_surface(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    colors: np.ndarray,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stride = max(int(stride), 1)
    return (
        x_grid[::stride, ::stride],
        y_grid[::stride, ::stride],
        z_grid[::stride, ::stride],
        colors[::stride, ::stride, :],
    )


def clipped_index(value: float, maximum: int) -> int:
    return int(np.clip(round(float(value)), 0, maximum - 1))


def load_landmarks(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError):
        return {}


def landmarks_zyx_to_xyz(values: Any) -> np.ndarray:
    points = np.asarray(values, dtype=float)
    if points.size == 0:
        return np.empty((0, 3), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Landmarks must be stored as an Nx3 array.")
    return points[:, [2, 1, 0]]


def transform_points_to_ccf(
    points_xyz: np.ndarray,
    transform_path: Path | None,
    source_spacing_xyz_mm: np.ndarray,
    atlas_spacing_xyz_mm: np.ndarray,
    selected_transform_direction: str,
) -> np.ndarray:
    """Transform aligned microCT XYZ voxel points into Allen CCF XYZ voxels."""
    if points_xyz.size == 0:
        return points_xyz.reshape(0, 3)
    if selected_transform_direction == "registered":
        return points_xyz.copy()
    if transform_path is None:
        return np.empty((0, 3), dtype=float)
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError("Landmark transformation requires SimpleITK.") from exc

    transform = sitk.ReadTransform(str(transform_path))
    if selected_transform_direction == "inverse":
        transform = transform.GetInverse()
    elif selected_transform_direction not in {"direct", "inverse"}:
        try:
            transform = transform.GetInverse()
        except RuntimeError:
            pass

    transformed = []
    for point in points_xyz:
        source_mm = tuple((point * source_spacing_xyz_mm).tolist())
        ccf_mm = np.asarray(transform.TransformPoint(source_mm), dtype=float)
        transformed.append(ccf_mm / atlas_spacing_xyz_mm)
    return np.asarray(transformed, dtype=float)


class MicroprismRegistration3DViewer:
    """Interactive 3D scene for registration validation."""

    def __init__(
        self,
        atlas_template: np.ndarray,
        registered_microct: np.ndarray | None,
        rectified_corners_xyz: dict[str, np.ndarray],
        fixed_landmarks_xyz: np.ndarray,
        transformed_moving_landmarks_xyz: np.ndarray,
        metrics: dict[str, Any],
        atlas_spacing_xyz_mm: np.ndarray,
        selected_transform_direction: str,
        microct_alpha: float,
        surface_stride: int,
    ) -> None:
        self.atlas_template = atlas_template
        self.registered_microct = registered_microct
        self.nz, self.ny, self.nx = atlas_template.shape
        self.rectified_corners_xyz = rectified_corners_xyz
        self.fixed_landmarks_xyz = fixed_landmarks_xyz
        self.transformed_moving_landmarks_xyz = transformed_moving_landmarks_xyz
        self.metrics = metrics
        self.atlas_spacing_xyz_mm = atlas_spacing_xyz_mm
        self.selected_transform_direction = selected_transform_direction
        self.microct_alpha = float(np.clip(microct_alpha, 0.0, 1.0))
        self.surface_stride = max(1, int(surface_stride))

        face = np.stack([self.rectified_corners_xyz[key] for key in FACE_KEYS])
        center = face.mean(axis=0)
        self.slice_x = clipped_index(center[0], self.nx)
        self.slice_y = clipped_index(center[1], self.ny)
        self.slice_z = clipped_index(center[2], self.nz)

        self.show_coronal = True
        self.show_sagittal = False
        self.show_axial = False
        self.show_microct = registered_microct is not None
        self.show_landmarks = True
        self.zoom_level = 1.0
        self.fig = None
        self.ax_3d = None
        self.ax_info = None
        self.surfaces = []
        self.buttons: list[Any] = []

    def start(self, save_path: Path | None, show: bool) -> None:
        print("\n" + "=" * 84)
        print("STARTING 3D MICROPRISM REGISTRATION VALIDATION")
        print("=" * 84)
        print("Controls:")
        print("  - Mouse drag: rotate 3D view")
        print("  - Mouse wheel over 3D scene: zoom in/out")
        print("  - Sliders: move coronal/sagittal/axial slice planes")
        print("  - Checkboxes: show/hide slice planes, microCT overlay, landmarks")
        print("  - View buttons: jump to top/side/front/3D camera views")
        print("=" * 84 + "\n")

        self.fig = plt.figure(figsize=(18, 10))
        self.fig.subplots_adjust(left=0.04, right=0.98, top=0.94, bottom=0.10)
        self.ax_3d = self.fig.add_axes([0.03, 0.18, 0.66, 0.74], projection="3d")
        self.ax_info = self.fig.add_axes([0.72, 0.38, 0.25, 0.50])
        self.ax_info.axis("off")

        self._make_controls()
        self._draw_scene()
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)

        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.fig.savefig(save_path, dpi=180, bbox_inches="tight")
            print(f"Saved 3D QC screenshot: {save_path}")
        if show:
            plt.show()
        else:
            plt.close(self.fig)

    def _make_controls(self) -> None:
        ax_slider_z = self.fig.add_axes([0.10, 0.105, 0.55, 0.025])
        ax_slider_x = self.fig.add_axes([0.10, 0.065, 0.55, 0.025])
        ax_slider_y = self.fig.add_axes([0.10, 0.025, 0.55, 0.025])
        self.slider_z = Slider(ax_slider_z, "Coronal Z", 0, self.nz - 1, valinit=self.slice_z, valstep=1)
        self.slider_x = Slider(ax_slider_x, "Sagittal X", 0, self.nx - 1, valinit=self.slice_x, valstep=1)
        self.slider_y = Slider(ax_slider_y, "Axial Y", 0, self.ny - 1, valinit=self.slice_y, valstep=1)
        self.slider_z.on_changed(lambda value: self._set_slice("z", value))
        self.slider_x.on_changed(lambda value: self._set_slice("x", value))
        self.slider_y.on_changed(lambda value: self._set_slice("y", value))

        ax_checks = self.fig.add_axes([0.72, 0.19, 0.22, 0.16])
        checks = CheckButtons(
            ax_checks,
            ("Coronal", "Sagittal", "Axial", "microCT", "Landmarks"),
            (
                self.show_coronal,
                self.show_sagittal,
                self.show_axial,
                self.show_microct,
                self.show_landmarks,
            ),
        )

        def on_check(label: str) -> None:
            if label == "Coronal":
                self.show_coronal = not self.show_coronal
            elif label == "Sagittal":
                self.show_sagittal = not self.show_sagittal
            elif label == "Axial":
                self.show_axial = not self.show_axial
            elif label == "microCT":
                self.show_microct = not self.show_microct
            elif label == "Landmarks":
                self.show_landmarks = not self.show_landmarks
            self._draw_scene()

        checks.on_clicked(on_check)
        self.buttons.append(checks)

        specs = [
            ([0.72, 0.105, 0.055, 0.045], "Top", lambda _: self._view(90, -90)),
            ([0.785, 0.105, 0.055, 0.045], "Side", lambda _: self._view(0, 0)),
            ([0.85, 0.105, 0.055, 0.045], "Front", lambda _: self._view(0, -90)),
            ([0.915, 0.105, 0.055, 0.045], "3D", lambda _: self._view(22, -55)),
            ([0.72, 0.035, 0.065, 0.045], "Zoom +", lambda _: self._change_zoom(1.25)),
            ([0.795, 0.035, 0.065, 0.045], "Zoom -", lambda _: self._change_zoom(1 / 1.25)),
            ([0.87, 0.035, 0.08, 0.045], "Reset", lambda _: self._reset_zoom()),
        ]
        for bounds, label, callback in specs:
            button = Button(self.fig.add_axes(bounds), label)
            button.on_clicked(callback)
            self.buttons.append(button)

    def _set_slice(self, axis: str, value: float) -> None:
        if axis == "z":
            self.slice_z = int(value)
        elif axis == "x":
            self.slice_x = int(value)
        elif axis == "y":
            self.slice_y = int(value)
        self._draw_scene()

    def _on_scroll(self, event) -> None:
        if event.inaxes != self.ax_3d:
            return
        if event.button == "up":
            self.zoom_level = min(self.zoom_level * 1.15, 8.0)
        elif event.button == "down":
            self.zoom_level = max(self.zoom_level / 1.15, 0.5)
        self._apply_axes()
        self.fig.canvas.draw_idle()

    def _change_zoom(self, factor: float) -> None:
        self.zoom_level = float(np.clip(self.zoom_level * factor, 0.5, 8.0))
        self._apply_axes()
        self._draw_info()
        self.fig.canvas.draw_idle()

    def _view(self, elev: float, azim: float) -> None:
        self.ax_3d.view_init(elev=elev, azim=azim)
        self.fig.canvas.draw_idle()

    def _reset_zoom(self) -> None:
        self.zoom_level = 1.0
        self._apply_axes()
        self.fig.canvas.draw_idle()

    def _draw_scene(self) -> None:
        elev = self.ax_3d.elev if self.ax_3d else 22
        azim = self.ax_3d.azim if self.ax_3d else -55
        self.ax_3d.clear()
        self.surfaces.clear()
        if self.show_coronal:
            self._draw_coronal_surface(self.slice_z)
        if self.show_sagittal:
            self._draw_sagittal_surface(self.slice_x)
        if self.show_axial:
            self._draw_axial_surface(self.slice_y)
        self._draw_prism_face()
        self._draw_landmarks()
        self._draw_atlas_box()
        self._apply_axes()
        self.ax_3d.view_init(elev=elev, azim=azim)
        self.ax_3d.set_title(
            "3D registration validation: Allen template + registered microCT + microprism plane",
            fontsize=12,
            fontweight="bold",
            pad=18,
        )
        self._draw_info()
        self.fig.canvas.draw_idle()

    def _draw_coronal_surface(self, z: int) -> None:
        atlas_slice = self.atlas_template[z, :, :]
        micro = self.registered_microct[z, :, :] if self._can_show_microct() else None
        colors = atlas_microct_facecolors(atlas_slice, micro, self.microct_alpha)
        xx, yy = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        zz = np.full_like(xx, z, dtype=float)
        xx, yy, zz, colors = downsample_surface(xx, yy, zz, colors, self.surface_stride)
        self.ax_3d.plot_surface(xx, yy, zz, facecolors=colors, shade=False, alpha=0.58, linewidth=0)

    def _draw_sagittal_surface(self, x: int) -> None:
        atlas_slice = self.atlas_template[:, :, x]
        micro = self.registered_microct[:, :, x] if self._can_show_microct() else None
        colors = atlas_microct_facecolors(atlas_slice, micro, self.microct_alpha)
        yy, zz = np.meshgrid(np.arange(self.ny), np.arange(self.nz))
        xx = np.full_like(yy, x, dtype=float)
        xx, yy, zz, colors = downsample_surface(xx, yy, zz, colors, self.surface_stride)
        self.ax_3d.plot_surface(xx, yy, zz, facecolors=colors, shade=False, alpha=0.58, linewidth=0)

    def _draw_axial_surface(self, y: int) -> None:
        atlas_slice = self.atlas_template[:, y, :]
        micro = self.registered_microct[:, y, :] if self._can_show_microct() else None
        colors = atlas_microct_facecolors(atlas_slice, micro, self.microct_alpha)
        xx, zz = np.meshgrid(np.arange(self.nx), np.arange(self.nz))
        yy = np.full_like(xx, y, dtype=float)
        xx, yy, zz, colors = downsample_surface(xx, yy, zz, colors, self.surface_stride)
        self.ax_3d.plot_surface(xx, yy, zz, facecolors=colors, shade=False, alpha=0.58, linewidth=0)

    def _can_show_microct(self) -> bool:
        return (
            self.show_microct
            and self.registered_microct is not None
            and self.registered_microct.shape == self.atlas_template.shape
        )

    def _draw_prism_face(self) -> None:
        face = np.stack([self.rectified_corners_xyz[key] for key in FACE_KEYS])
        polygon = Poly3DCollection(
            [face],
            facecolors=(1.0, 0.86, 0.1, 0.30),
            edgecolors="#ffd84d",
            linewidths=3.0,
        )
        self.ax_3d.add_collection3d(polygon)
        closed = np.vstack((face, face[0]))
        self.ax_3d.plot(closed[:, 0], closed[:, 1], closed[:, 2], color="#ffd84d", linewidth=3)
        self.ax_3d.scatter(face[:, 0], face[:, 1], face[:, 2], color="#ffd84d", edgecolor="black", s=70)
        for key, point in self.rectified_corners_xyz.items():
            self.ax_3d.text(point[0], point[1], point[2], FACE_LABELS[key], color="black", fontsize=10)

    def _draw_landmarks(self) -> None:
        if not self.show_landmarks:
            return
        if self.fixed_landmarks_xyz.size:
            self.ax_3d.scatter(
                self.fixed_landmarks_xyz[:, 0],
                self.fixed_landmarks_xyz[:, 1],
                self.fixed_landmarks_xyz[:, 2],
                c="#ff4dff",
                s=24,
                marker="o",
                alpha=0.65,
                label="Allen landmarks",
            )
        if self.transformed_moving_landmarks_xyz.size:
            self.ax_3d.scatter(
                self.transformed_moving_landmarks_xyz[:, 0],
                self.transformed_moving_landmarks_xyz[:, 1],
                self.transformed_moving_landmarks_xyz[:, 2],
                c="#00e676",
                s=35,
                marker="+",
                alpha=0.9,
                label="microCT landmarks -> Allen",
            )
        if self.fixed_landmarks_xyz.size or self.transformed_moving_landmarks_xyz.size:
            self.ax_3d.legend(loc="upper left", fontsize=8)

    def _draw_atlas_box(self) -> None:
        corners = np.array([
            (x, y, z)
            for x in (0, self.nx - 1)
            for y in (0, self.ny - 1)
            for z in (0, self.nz - 1)
        ], dtype=float)
        edges = (
            (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
            (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
        )
        for first_index, second_index in edges:
            first = corners[first_index]
            second = corners[second_index]
            self.ax_3d.plot(
                [first[0], second[0]],
                [first[1], second[1]],
                [first[2], second[2]],
                color="0.35",
                linewidth=0.8,
                alpha=0.35,
            )

    def _apply_axes(self) -> None:
        center = np.array([self.nx / 2.0, self.ny / 2.0, self.nz / 2.0])
        ranges = np.array([self.nx, self.ny, self.nz], dtype=float) / (2.0 * self.zoom_level)
        self.ax_3d.set_xlim(center[0] - ranges[0], center[0] + ranges[0])
        self.ax_3d.set_ylim(center[1] - ranges[1], center[1] + ranges[1])
        self.ax_3d.set_zlim(center[2] - ranges[2], center[2] + ranges[2])
        self.ax_3d.set_xlabel("X")
        self.ax_3d.set_ylabel("Y")
        self.ax_3d.set_zlabel("Z")
        self.ax_3d.set_box_aspect((self.nx, self.ny, self.nz))

    def _draw_info(self) -> None:
        self.ax_info.clear()
        self.ax_info.axis("off")
        face = np.stack([self.rectified_corners_xyz[key] for key in FACE_KEYS])
        width_vox = np.linalg.norm(
            self.rectified_corners_xyz["bottom_right"]
            - self.rectified_corners_xyz["bottom_left"]
        )
        length_vox = np.linalg.norm(
            self.rectified_corners_xyz["top_left"]
            - self.rectified_corners_xyz["bottom_left"]
        )
        spacing_mean = float(np.mean(self.atlas_spacing_xyz_mm))
        text = [
            "3D validation",
            "",
            f"Transform used: {self.selected_transform_direction}",
            f"Atlas shape ZYX: {(self.nz, self.ny, self.nx)}",
            f"Atlas spacing: {self.atlas_spacing_xyz_mm.tolist()} mm",
            f"microCT spacing: {self.metrics.get('moving_spacing_um', 'unknown')} um",
            f"Landmark pairs: {self.metrics.get('num_landmarks', 'unknown')}",
            f"Mean error: {self.metrics.get('mean_error_mm', 'unknown')} mm",
            f"Max error: {self.metrics.get('max_error_mm', 'unknown')} mm",
            "",
            f"Plane center XYZ: {np.round(face.mean(axis=0), 1).tolist()}",
            f"Plane width: {width_vox * spacing_mean:.3f} mm",
            f"Plane length: {length_vox * spacing_mean:.3f} mm",
            f"Zoom: {self.zoom_level:.1f}x",
            "",
            "Scene guide:",
            "gray = Allen template",
            "green = registered microCT",
            "yellow = imaging face",
            "P1/P2/dL/dR = prism face corners",
            "magenta/green + = landmark QC",
        ]
        self.ax_info.text(
            0.02, 0.98, "\n".join(str(line) for line in text),
            transform=self.ax_info.transAxes,
            va="top",
            ha="left",
            fontsize=9.5,
            family="monospace",
            bbox={"facecolor": "#f6f6f6", "edgecolor": "0.65", "pad": 10},
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show a Sakata-style 3D validation scene for microprism registration."
        )
    )
    parser.add_argument("project_path", nargs="?")
    parser.add_argument("--corners", help="Path to microprism_corners.json.")
    parser.add_argument("--template", help="Allen average-template NRRD, TIFF, or NPY.")
    parser.add_argument("--annotation", help="Allen annotation NRRD, TIFF, or NPY.")
    parser.add_argument("--transform", help="SimpleITK registration transform.")
    parser.add_argument(
        "--registered-microct",
        help="Registered microCT TIFF/NRRD/NPY. Defaults to data/processed/microct_registered.tif.",
    )
    parser.add_argument("--landmarks", help="Path to landmark_registration landmarks.json.")
    parser.add_argument("--output", help="Output PNG path.")
    parser.add_argument(
        "--coordinate-space",
        choices=("auto", "registered", "aligned"),
        default="auto",
        help="Coordinate space of microprism_corners.json.",
    )
    parser.add_argument(
        "--saved-transform-maps",
        choices=("auto", "fixed-to-moving", "moving-to-fixed"),
        default="auto",
        help="Direction of the saved transform.",
    )
    parser.add_argument("--source-spacing", type=float, help="Fallback source voxel size in mm.")
    parser.add_argument("--source-spacing-x", type=float)
    parser.add_argument("--source-spacing-y", type=float)
    parser.add_argument("--source-spacing-z", type=float)
    parser.add_argument(
        "--atlas-spacing", type=float, default=DEFAULT_SPACING_MM,
        help="Allen atlas voxel size in mm. annotation_25 uses 0.025.",
    )
    parser.add_argument("--atlas-spacing-x", type=float)
    parser.add_argument("--atlas-spacing-y", type=float)
    parser.add_argument("--atlas-spacing-z", type=float)
    parser.add_argument("--microct-alpha", type=float, default=0.55)
    parser.add_argument(
        "--surface-stride",
        type=int,
        default=2,
        help=(
            "Downsample factor for 3D surface planes. Use 1 for the least "
            "pixelated view; larger values are faster."
        ),
    )
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = resolve_project_path(args.project_path)

    corners_path = resolve_input(
        args.corners, project, ("microprism_corners.json",),
        "microprism tracker output",
    )
    annotation_path = resolve_input(
        args.annotation, project,
        ("annotation_25.nrrd", "annotation_25.tif", "annotation_25.npy"),
        "Allen CCF annotation", required=False,
    )
    template_path = resolve_input(
        args.template, project,
        ("average_template_25.nrrd", "average_template_25.tif", "average_template_25.npy"),
        "Allen average template", required=False,
    )
    if annotation_path is None:
        annotation_path, _ = ensure_allen_ccf_resources(
            allow_download=not args.no_download
        )
    if template_path is None:
        template_path = ensure_allen_template(allow_download=not args.no_download)

    transform_path = resolve_input(
        args.transform, project,
        ("transform_refined.tfm", "transform_landmark.tfm"),
        "registration transform", required=False,
    )
    registered_microct_path = resolve_input(
        args.registered_microct, project,
        ("microct_registered.tif", "microct_registered.tiff", "registered_microct.tif"),
        "registered microCT", required=False,
    )
    landmarks_path = resolve_input(
        args.landmarks, project, ("landmarks.json",),
        "landmark pairs", required=False,
    )

    print("=" * 84)
    print("MICROPRISM 3D REGISTRATION VALIDATION")
    print("=" * 84)
    print(f"Tracker corners: {corners_path}")
    print(f"Allen annotation: {annotation_path}")
    print(f"Allen template: {template_path}")
    print(f"Transform: {transform_path or 'not found'}")
    print(f"Registered microCT: {registered_microct_path or 'not found'}")
    print(f"Landmarks: {landmarks_path or 'not found'}")

    corners_source, tracker_data = load_face_corners(corners_path)
    annotation = load_annotation(annotation_path)
    atlas_template = load_annotation(template_path)
    if atlas_template.shape != annotation.shape:
        raise ValueError(
            "Allen template and annotation shapes must match: "
            f"{atlas_template.shape} != {annotation.shape}"
        )

    registered_microct = None
    if registered_microct_path is not None:
        registered_microct = load_volume(registered_microct_path, "registered microCT")
        if registered_microct.shape != atlas_template.shape:
            print(
                "\nWARNING: registered microCT shape does not match Allen template. "
                "The 3D scene will show Allen slices and prism geometry, but it "
                "will not blend microCT onto the surfaces."
            )
            print(f"  registered microCT shape: {registered_microct.shape}")
            print(f"  Allen template shape:    {atlas_template.shape}")

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
    saved_transform_maps = args.saved_transform_maps
    if saved_transform_maps == "auto":
        metadata_direction = metadata_transform_direction(registration_metrics)
        if metadata_direction is not None:
            saved_transform_maps = metadata_direction
            print(f"Using transform direction from registration metrics: {saved_transform_maps}.")

    warn_transform_provenance(
        transform_path, project, registration_metrics, source_spacing_xyz
    )

    coordinate_space = infer_coordinate_space(
        args.coordinate_space, tracker_data, annotation.shape
    )
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
    print(f"Prism corners in Allen bounds: {corners_in_bounds}/4")
    if corners_in_bounds == 0:
        print(
            "\nWARNING: all prism corners landed outside Allen space. "
            "This usually means the registration transform, spacing, axis "
            "mapping, or tracked source image is wrong."
        )

    prism_width_mm = float(tracker_data.get("prism_width_mm", 0.0))
    prism_length_mm = float(tracker_data.get("prism_length_mm", 0.0))
    target_face_ratio = (
        prism_length_mm / prism_width_mm
        if prism_width_mm > 0 and prism_length_mm > 0
        else None
    )
    geometry = build_plane_geometry(corners_ccf, target_face_ratio)
    rectified_corners = rectified_face_corners_xyz(geometry)

    landmark_data = load_landmarks(landmarks_path)
    fixed_landmarks_xyz = landmarks_zyx_to_xyz(landmark_data.get("fixed_landmarks", []))
    moving_landmarks_xyz = landmarks_zyx_to_xyz(landmark_data.get("moving_landmarks", []))
    transformed_moving_landmarks_xyz = transform_points_to_ccf(
        moving_landmarks_xyz,
        transform_path,
        source_spacing_xyz,
        atlas_spacing_xyz,
        selected_transform_direction,
    )
    if fixed_landmarks_xyz.size and transformed_moving_landmarks_xyz.size:
        errors_mm = np.linalg.norm(
            (transformed_moving_landmarks_xyz - fixed_landmarks_xyz)
            * atlas_spacing_xyz,
            axis=1,
        )
        print(
            "Landmark overlay check: "
            f"mean={errors_mm.mean():.3f} mm, max={errors_mm.max():.3f} mm"
        )

    print("\nValidation checklist:")
    print("  1. Rotate the scene and confirm the green microCT anatomy sits inside the gray Allen template.")
    print("  2. Check whether magenta Allen landmarks and green transformed microCT landmarks overlap.")
    print("  3. Confirm the yellow microprism plane is inside the brain and tilted plausibly.")
    print("  4. Use Top/Side/Front/3D buttons to inspect axis orientation.")
    print("  5. If the scene looks wrong, re-check spacing, transform direction, stale files, and landmark pairing.")

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output else project / "outputs" / "microprism_registration_qc_3d.png"
    )
    viewer = MicroprismRegistration3DViewer(
        atlas_template=atlas_template,
        registered_microct=registered_microct,
        rectified_corners_xyz=rectified_corners,
        fixed_landmarks_xyz=fixed_landmarks_xyz,
        transformed_moving_landmarks_xyz=transformed_moving_landmarks_xyz,
        metrics=registration_metrics,
        atlas_spacing_xyz_mm=atlas_spacing_xyz,
        selected_transform_direction=selected_transform_direction,
        microct_alpha=args.microct_alpha,
        surface_stride=args.surface_stride,
    )
    viewer.start(save_path=output_path, show=not args.no_show)


if __name__ == "__main__":
    main()
