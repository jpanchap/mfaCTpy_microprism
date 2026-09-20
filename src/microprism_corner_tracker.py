"""
microprism_corner_tracker.py  (v10 - commented + padded UI)
─────────────────────────────────────────────────────────────────────────────
Interactive tool for marking microprism mirror-face corners in microCT brain
volumes. Matches the postdoc's UI style (light wheat/cream background, black
text, standard matplotlib colors) from midline_alignment.py.

3D DIAGRAM
──────────
Right-angle prism: tall rectangular body with bottom-right corner cut at 45°
(pentagon cross-section), extruded into depth. The hypotenuse parallelogram
is the mirror face. Matches the hand-drawn schematic exactly.

WORKFLOW
────────
  Phase 1 — Coronal:  click P1 → P2 → P3 → P4
  Phase 2 — Sagittal: click P1 → P2 → P3 → P4
  Phase 3 — Axial:    click P1 → P2 → P3 → P4

  Final coordinate = average of 3 orientation clicks per point.
  P1/P2 are shared with the imaging plane. D1/D2 are derived from P3/P4 by
  solving the side cross-section as a right triangle.

OUTPUT
──────
  outputs/microprism_corners.json   — full results with per-orientation data
  outputs/microprism_corners.csv    — voxel + mm coordinates, one row per point

Keyboard: C/S/A switch views · arrows/scroll navigate · +/- zoom · O overlays
          U undo · Enter save

UI update: larger brain image panel, added bottom padding, and comments/docstrings.

TODO: Update PRISM_EDGE_MM once confirmed by postdoc.
Dependencies: pip install numpy tifffile matplotlib
"""

from __future__ import annotations

import argparse
from project_paths import resolve_project_path
import csv
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tifffile
from matplotlib.widgets import Button, Slider

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
# Physical prism size and voxel spacing used for mm conversions in output files.
DEFAULT_PRISM_WIDTH_MM = 0.5
DEFAULT_PRISM_LENGTH_MM = 0.5
DEFAULT_SPACING = 0.025
DEFAULT_SPACING_XYZ = (DEFAULT_SPACING, DEFAULT_SPACING, DEFAULT_SPACING)
GUIDE_IMAGE_PATH = Path(__file__).resolve().parent / "assets" / "microprism_guide.png"

# Four points the user clicks in each of the three orientations. The color is
# used consistently in the image markers, guide markers, legend, and outputs.
POINT_DEFS = [
    {"id": "P1", "label": "Point 1", "desc": "bottom-left shared corner",  "color": "#0077ff"},
    {"id": "P2", "label": "Point 2", "desc": "bottom-right shared corner", "color": "#0077ff"},
    {"id": "P3", "label": "Point 3", "desc": "top-left prism corner",      "color": "#0077ff"},
    {"id": "P4", "label": "Point 4", "desc": "top-right prism corner",     "color": "#0077ff"},
]

# Orientation lookup tables:
#   ORIENT_KEYS   = internal identifiers used by the code
#   ORIENT_NAMES  = labels shown in the UI
#   ORIENT_COLORS = colors used in titles/progress labels
ORIENT_KEYS   = ["coronal",   "sagittal",  "axial"]
ORIENT_NAMES  = {"coronal": "Coronal (Z)", "sagittal": "Sagittal (X)", "axial": "Axial (Y)"}
ORIENT_COLORS = {"coronal": "#2980b9",     "sagittal": "#8e44ad",      "axial": "#d35400"}

# A click is stored as (display_column, display_row, slice_index). Because each
# anatomical view displays different axes, _click_to_xyz() converts these raw
# click triples into standard x/y/z voxel coordinates before saving.
Click = Tuple[int, int, int]
XYZ   = Tuple[float, float, float]


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE SELECTION
# ─────────────────────────────────────────────────────────────────────────────
def find_best_microct_image(project_path: Path,
                            explicit_image: Optional[Path] = None) -> Path:
    """Choose the TIFF stack to open.

    If --image is provided, use it directly. Otherwise, look inside the project
    folder and prefer corrected/processed TIFFs before falling back to raw TIFFs.
    """
    if explicit_image is not None:
        p = explicit_image.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        return p

    project_path  = project_path.expanduser().resolve()
    data_dir      = project_path / "data"
    processed_dir = data_dir / "processed"

    preferred = [
        processed_dir / "microct_aligned_corrected.tif",
        processed_dir / "microct_axis_corrected.tif",
        processed_dir / "microct_aligned_fullres.tif",
        data_dir / "microct_aligned_corrected.tif",
        data_dir / "microct_registered.tif",
        data_dir / "microct_aligned.tif",
    ]
    for c in preferred:
        if c.exists():
            return c

    fallback = (sorted(processed_dir.glob("*.tif")) +
                sorted(processed_dir.glob("*.tiff")) +
                sorted(data_dir.glob("*.tif")) +
                sorted(data_dir.glob("*.tiff")))
    if fallback:
        return fallback[0]

    raise FileNotFoundError(
        f"No TIFF found in {processed_dir} or {data_dir}. "
        "Use --image /path/to/file.tif to override."
    )


def prompt_float(label: str, default: float) -> float:
    """Prompt for a positive float, falling back to the default on blank input."""
    while True:
        try:
            raw = input(f"{label} [{default:g} mm]: ").strip()
        except EOFError:
            return float(default)
        if raw == "":
            return float(default)
        try:
            value = float(raw)
        except ValueError:
            print("  Please enter a number.")
            continue
        if value <= 0:
            print("  Please enter a positive number.")
            continue
        return value


def rational_to_float(value) -> float:
    """Convert TIFF rational tag values into plain floats."""
    try:
        numerator, denominator = value
        return float(numerator) / float(denominator)
    except TypeError:
        return float(value)


def spacing_xyz_from_values_um(values, source: str) -> Optional[Tuple[float, float, float]]:
    """Return X/Y/Z spacing in mm from X/Y/Z micrometer values."""
    if not values:
        return None
    try:
        vals = tuple(float(v) for v in values)
    except (TypeError, ValueError):
        return None
    if len(vals) != 3 or any(v <= 0 for v in vals):
        return None
    print(
        f"✓ Inferred voxel spacing from {source}: "
        f"X={vals[0]:g}, Y={vals[1]:g}, Z={vals[2]:g} µm"
    )
    return tuple(v / 1000.0 for v in vals)


def infer_spacing_xyz_mm(image_path: Path) -> Optional[Tuple[float, float, float]]:
    """Infer X/Y/Z voxel spacing from sidecar or TIFF/ImageJ metadata."""
    sidecar_path = image_path.with_suffix(image_path.suffix + ".metadata.json")
    if sidecar_path.exists():
        with open(sidecar_path, "r") as f:
            metadata = json.load(f)

        if metadata.get("spacing_um_xyz"):
            spacing = spacing_xyz_from_values_um(
                metadata["spacing_um_xyz"], "metadata sidecar")
            if spacing is not None:
                return spacing
        if metadata.get("spacing_um_zyx"):
            zyx = metadata["spacing_um_zyx"]
            spacing = spacing_xyz_from_values_um(
                [zyx[2], zyx[1], zyx[0]], "metadata sidecar")
            if spacing is not None:
                return spacing
        if metadata.get("spacing_mm_xyz"):
            values = tuple(float(v) for v in metadata["spacing_mm_xyz"])
            if len(values) == 3 and all(v > 0 for v in values):
                print(
                    "✓ Inferred voxel spacing from metadata sidecar: "
                    f"X={values[0]:g}, Y={values[1]:g}, Z={values[2]:g} mm"
                )
                return values
        if metadata.get("spacing_mm"):
            value = float(metadata["spacing_mm"])
            if value > 0:
                print(f"✓ Inferred isotropic voxel spacing from metadata sidecar: {value:g} mm")
                return (value, value, value)

    try:
        with tifffile.TiffFile(str(image_path)) as tif:
            imagej_metadata = tif.imagej_metadata or {}
            unit = str(imagej_metadata.get("unit", "")).lower()
            spacing = imagej_metadata.get("spacing")
            x_resolution = tif.pages[0].tags.get("XResolution")
            y_resolution = tif.pages[0].tags.get("YResolution")
            resolution_unit = tif.pages[0].tags.get("ResolutionUnit")
    except Exception:
        return None

    if spacing is not None and unit in {"um", "µm", "micron", "microns"}:
        value = float(spacing) / 1000.0
        print(f"✓ Inferred isotropic voxel spacing from ImageJ TIFF metadata: {float(spacing):g} µm")
        return (value, value, value)

    if x_resolution and y_resolution and resolution_unit:
        unit_name = str(resolution_unit.value).upper()
        if unit_name in {"CENTIMETER", "INCH"}:
            x_per_unit = rational_to_float(x_resolution.value)
            y_per_unit = rational_to_float(y_resolution.value)
            unit_um = 10000.0 if unit_name == "CENTIMETER" else 25400.0
            values_um = [unit_um / x_per_unit, unit_um / y_per_unit]
            if values_um[0] > 0 and values_um[1] > 0:
                z_um = values_um[0] if abs(values_um[0] - values_um[1]) < 1e-6 else None
                if z_um is not None:
                    return spacing_xyz_from_values_um(
                        [values_um[0], values_um[1], z_um], "TIFF resolution tags")

    return None


def prompt_spacing_xyz_mm(default_um: float = 72.0) -> Tuple[float, float, float]:
    """Ask the user for isotropic voxel spacing in micrometers."""
    while True:
        try:
            raw = input(
                "\nCould not determine microCT voxel spacing automatically.\n"
                f"Enter microCT voxel spacing in micrometers, e.g. 72 [{default_um:g}]: "
            ).strip()
        except EOFError:
            raw = ""
        if raw == "":
            value_um = float(default_um)
        else:
            try:
                value_um = float(raw)
            except ValueError:
                print("  Please enter a number, for example 72.")
                continue
        if value_um <= 0:
            print("  Spacing must be positive.")
            continue
        value_mm = value_um / 1000.0
        return (value_mm, value_mm, value_mm)


def spacing_xyz_from_args(args, image_path: Path) -> Tuple[float, float, float]:
    """Return spacing in SimpleITK/coordinate order: (x_mm, y_mm, z_mm)."""
    inferred = None
    if args.spacing is None or args.spacing_x is None or args.spacing_y is None or args.spacing_z is None:
        inferred = infer_spacing_xyz_mm(image_path)
    if inferred is None and args.spacing is None:
        inferred = prompt_spacing_xyz_mm()

    if args.spacing is not None:
        base_values = (float(args.spacing), float(args.spacing), float(args.spacing))
    else:
        base_values = inferred

    x = args.spacing_x if args.spacing_x is not None else base_values[0]
    y = args.spacing_y if args.spacing_y is not None else base_values[1]
    z = args.spacing_z if args.spacing_z is not None else base_values[2]
    values = (float(x), float(y), float(z))
    if any(value <= 0 for value in values):
        raise ValueError("All spacing values must be positive.")
    return values


# ─────────────────────────────────────────────────────────────────────────────
# TRACKER
# ─────────────────────────────────────────────────────────────────────────────
class MicroprismCornerTracker:

    def __init__(
        self,
        microct_image_path: str | Path,
        output_dir: str | Path | None = None,
        spacing_mm: float = DEFAULT_SPACING,
        spacing_xyz_mm: Tuple[float, float, float] | None = None,
        prism_width_mm: float = DEFAULT_PRISM_WIDTH_MM,
        prism_length_mm: float = DEFAULT_PRISM_LENGTH_MM,
        maximize: bool = True,
    ) -> None:
        """Load image data and initialize all state used by the interactive UI."""
        print("=" * 72)
        print("MICROPRISM CORNER TRACKER  v10")
        print("=" * 72)

        self.image_path = Path(microct_image_path).expanduser().resolve()
        self.spacing    = float(spacing_mm)
        self.spacing_xyz = np.array(
            spacing_xyz_mm or (self.spacing, self.spacing, self.spacing),
            dtype=float,
        )
        self.prism_width_mm = float(prism_width_mm)
        self.prism_length_mm = float(prism_length_mm)
        self.maximize   = bool(maximize)

        print(f"\nLoading: {self.image_path.name}")
        self.image = tifffile.imread(str(self.image_path))
        if self.image.ndim == 4 and self.image.shape[0] == 1:
            self.image = self.image[0]
        if self.image.ndim != 3:
            raise ValueError(f"Expected 3D TIFF. Got shape: {self.image.shape}")

        self.nz, self.ny, self.nx = self.image.shape
        print(f"  Shape (Z, Y, X): {self.image.shape}")
        print(f"  Dtype: {self.image.dtype}")
        print(
            "  Spacing (X,Y,Z): "
            f"{self.spacing_xyz[0]:g}, {self.spacing_xyz[1]:g}, "
            f"{self.spacing_xyz[2]:g} mm"
        )
        print(f"  Prism width W: {self.prism_width_mm:g} mm")
        print(f"  Prism length l: {self.prism_length_mm:g} mm")
        print("  Reminder: after each view, compare P1-P4 voxel coordinates.")
        print("            Matching corners should be relatively consistent across views.")

        # Normalize contrast for display only. Coordinates are always saved in
        # the original voxel space, not in this normalized image.
        img_f = self.image.astype(np.float32, copy=False)
        lo, hi = np.percentile(img_f, [1, 99])
        if hi <= lo:
            lo, hi = float(img_f.min()), float(img_f.max())
        self.img_disp = np.clip((img_f - lo) / (hi - lo + 1e-8), 0, 1)

        # Determine output folder if the user did not pass --output.
        if output_dir is None:
            p = self.image_path.parent
            if p.name == "processed" and p.parent.name == "data":
                output_dir = p.parent.parent / "outputs"
            elif p.name == "data":
                output_dir = p.parent / "outputs"
            else:
                output_dir = p / "outputs"
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Output: {self.output_dir}")

        # 3 orientation phases × 4 points = 12 total clicks. Each row stores
        # one orientation; each column stores P1, P2, P3, or P4.
        self.clicks: List[List[Optional[Click]]] = [[None]*4 for _ in range(3)]
        self.current_orient = 0
        self.current_point  = 0

        self.current_view  = "coronal"
        self.slice_indices = {
            "coronal":  self.nz // 2,
            "sagittal": self.nx // 2,
            "axial":    self.ny // 2,
        }
        self.current_slice = self.slice_indices[self.current_view]

        self._save_warned     = False
        self._updating_slider = False
        self.show_overlays    = False
        self.pan_mode         = False
        self.zoom_factor      = 1.0
        self.zoom_center: Optional[Tuple[float, float]] = None
        self._pan_anchor = None
        self.flash_marker: Optional[Tuple[str, int, int, int, str]] = None
        self._flash_timer = None

        # Matplotlib handles are assigned in _build_ui(). Keeping them as
        # attributes lets event handlers update the same figure/axes.
        self.fig    = None
        self.ax     = None
        self.preview_axes: Dict[str, plt.Axes] = {}
        self.ax_guide = None
        self.ax_list = None
        self.guide_image = None
        self.slider = None
        self.buttons: Dict[str, Button] = {}

    # ─────────────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch the GUI window after printing terminal instructions."""
        self._print_instructions()
        self._build_ui()
        plt.show()

    # ─────────────────────────────────────────────────────────────────────
    # UI BUILD — matches postdoc / midline_alignment.py style
    # Light background, wheat info box, standard matplotlib colors
    # ─────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Build the Matplotlib UI: brain image, guide, slider, buttons, hints.

        Changes made here:
        - Brain image panel is larger.
        - Bottom controls are moved upward so labels are not clipped.
        - The keyboard-hint box is moved above the figure edge for padding.
        """
        # Larger figure gives the brain image room while keeping controls readable.
        self.fig = plt.figure(figsize=(23, 11.5))
        self.fig.patch.set_facecolor("white")

        # Main image axes.
        # [left, bottom, width, height] are figure-relative coordinates.
        self.ax = self.fig.add_axes([0.220, 0.205, 0.510, 0.745])
        self.ax.set_facecolor("black")

        self.preview_axes = {
            "coronal":  self.fig.add_axes([0.735, 0.555, 0.255, 0.370]),
            "sagittal": self.fig.add_axes([0.735, 0.555, 0.255, 0.370]),
            "axial":    self.fig.add_axes([0.735, 0.135, 0.255, 0.370]),
        }
        for ax_prev in self.preview_axes.values():
            ax_prev.set_facecolor("black")

        self.ax_list = self.fig.add_axes([0.020, 0.565, 0.205, 0.360])
        self.ax_list.set_axis_off()

        # Visual guide — smaller and tucked under the coordinate table.
        self.ax_guide = self.fig.add_axes([0.065, 0.185, 0.095, 0.315])
        self.ax_guide.set_facecolor("#f7f4e8")  # wheat/cream tone
        if GUIDE_IMAGE_PATH.exists():
            self.guide_image = mpimg.imread(GUIDE_IMAGE_PATH)

        # Slider — same width as the larger brain panel, with padding above the
        # button rows.
        ax_sl = self.fig.add_axes([0.220, 0.145, 0.510, 0.022])
        self.slider = Slider(
            ax_sl, "Coronal slice", 0, self.nz - 1,
            valinit=self.current_slice, valstep=1
        )
        self.slider.on_changed(self._on_slider)

        # View buttons — row 1, moved upward and slightly widened to prevent
        # text clipping.
        view_specs = [
            ("coronal",  "Coronal",  [0.220, 0.088, 0.105, 0.042]),
            ("sagittal", "Sagittal", [0.338, 0.088, 0.105, 0.042]),
            ("axial",    "Axial",    [0.456, 0.088, 0.105, 0.042]),
        ]
        for key, label, pos in view_specs:
            ax_b = self.fig.add_axes(pos)
            btn  = Button(
                ax_b, label,
                color=ORIENT_COLORS[key],
                hovercolor=ORIENT_COLORS[key],
            )
            btn.label.set_color("white")
            btn.label.set_fontweight("bold")
            btn.on_clicked(lambda e, v=key: self._switch_view(v))
            self.buttons[key] = btn

        # Action and inspection buttons.
        self.buttons["undo"] = Button(
            self.fig.add_axes([0.220, 0.038, 0.075, 0.035]), "Undo")
        self.buttons["overlay"] = Button(
            self.fig.add_axes([0.303, 0.038, 0.105, 0.035]), "Show overlays")
        self.buttons["pan"] = Button(
            self.fig.add_axes([0.416, 0.038, 0.075, 0.035]), "Pan off")
        self.buttons["zoom_in"] = Button(
            self.fig.add_axes([0.499, 0.038, 0.060, 0.035]), "Zoom +")
        self.buttons["zoom_out"] = Button(
            self.fig.add_axes([0.567, 0.038, 0.060, 0.035]), "Zoom -")
        self.buttons["zoom_reset"] = Button(
            self.fig.add_axes([0.635, 0.038, 0.060, 0.035]), "Zoom 1x")
        self.buttons["save"] = Button(
            self.fig.add_axes([0.055, 0.095, 0.135, 0.035]), "Save & Quit")
        self.buttons["quit"] = Button(
            self.fig.add_axes([0.055, 0.045, 0.135, 0.035]), "Quit (no save)")

        self.buttons["undo"].on_clicked(self._on_undo)
        self.buttons["overlay"].on_clicked(self._on_toggle_overlays)
        self.buttons["pan"].on_clicked(self._on_toggle_pan)
        self.buttons["zoom_in"].on_clicked(lambda _e: self._change_zoom(1.5))
        self.buttons["zoom_out"].on_clicked(lambda _e: self._change_zoom(1/1.5))
        self.buttons["zoom_reset"].on_clicked(lambda _e: self._reset_zoom())
        self.buttons["save"].on_clicked(self._on_save)
        self.buttons["quit"].on_clicked(lambda e: plt.close(self.fig))

        # Event hookups: all mouse/keyboard input goes through these handlers.
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("scroll_event",       self._on_scroll)
        self.fig.canvas.mpl_connect("key_press_event",    self._on_key)

        self._update_display()
        self._update_diagram()
        self._try_maximize()

    def _try_maximize(self) -> None:
        """Best-effort maximize for common Matplotlib backends."""
        if not self.maximize:
            return
        try:
            mgr = plt.get_current_fig_manager()
            if hasattr(mgr, "window"):
                if hasattr(mgr.window, "showMaximized"):
                    mgr.window.showMaximized()
                elif hasattr(mgr.window, "state"):
                    mgr.window.state("zoomed")
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────
    # MAIN IMAGE DISPLAY
    # ─────────────────────────────────────────────────────────────────────

    def _update_display(self) -> None:
        """Redraw the left brain image panel for the current view and slice."""
        view   = self.current_view
        sl     = self.current_slice
        o_idx  = ORIENT_KEYS.index(view)
        placed = sum(c is not None for c in self.clicks[o_idx])
        max_sl = self._max_slice(view)
        active = ORIENT_KEYS[self.current_orient] if self.current_orient < 3 else None

        # Draw the active 2D slice. The display slice changes depending on
        # whether the user is in coronal, sagittal, or axial mode.
        self.ax.imshow(self._get_slice_2d(view, sl),
                       cmap="gray", vmin=0, vmax=1,
                       interpolation="nearest", aspect="equal")
        self.ax.axis("off")

        # Title — matches midline's 4-line title style
        if self.current_orient >= 3:
            status = "All points placed — click Save & Quit"
            tcol   = "green"
        elif view != active:
            status = (f"Active phase: {ORIENT_NAMES[active]} — "
                      f"click that button to switch")
            tcol   = "red"
        else:
            pdef   = POINT_DEFS[self.current_point]
            status = f"Next: {pdef['id']} — {pdef['desc']}"
            tcol   = ORIENT_COLORS[view]

        self._render_view_panel(
            self.ax, view, sl,
            is_main=True,
            title=None,
            apply_zoom=True,
        )
        self.ax.set_title(
            f"{ORIENT_NAMES[view]}  |  Slice {sl}/{max_sl}  |  "
            f"{placed}/4 placed  |  Zoom {self.zoom_factor:.1f}x  |  "
            f"{'Pan on' if self.pan_mode else 'Pan off'}\n{status}",
            fontsize=11, color=tcol, pad=6
        )

        if self.flash_marker is not None:
            f_view, fa, fb, fs, f_pid = self.flash_marker
            if f_view == view and fs == sl:
                self.ax.plot(fa, fb, "+", color="#00ffff", markersize=22,
                             markeredgewidth=3.0, zorder=8)
                self.ax.add_patch(plt.Circle(
                    (fa, fb), 10, fill=False, edgecolor="#00ffff",
                    linewidth=2.2, zorder=8))
                self.ax.text(
                    fa+12, fb-12, f_pid, color="#00ffff", fontsize=10,
                    fontweight="bold", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="black", edgecolor="#00ffff", alpha=0.65))

        preview_slots = [
            [0.735, 0.555, 0.255, 0.370],
            [0.735, 0.135, 0.255, 0.370],
        ]
        visible_previews = [v for v in ORIENT_KEYS if v != view]
        for slot, preview_view in zip(preview_slots, visible_previews):
            ax_prev = self.preview_axes[preview_view]
            ax_prev.set_position(slot)
            ax_prev.set_visible(True)
            preview_sl = self.slice_indices[preview_view]
            self._render_view_panel(
                ax_prev, preview_view, preview_sl,
                is_main=False,
                title=f"{ORIENT_NAMES[preview_view]} | Slice {preview_sl}",
                apply_zoom=False,
            )

        for preview_view, ax_prev in self.preview_axes.items():
            if preview_view == view:
                ax_prev.set_visible(False)

        self._update_point_list()

        # Sync slider — matches midline change_to_view
        self.slider.valmin = 0
        self.slider.valmax = max_sl
        self.slider.ax.set_xlim(0, max_sl)
        self.slider.label.set_text(f"{view.capitalize()} slice")
        self._set_slider_val(sl)

        self.fig.canvas.draw_idle()

    def _render_view_panel(
        self,
        ax,
        view: str,
        sl: int,
        is_main: bool,
        title: Optional[str],
        apply_zoom: bool,
    ) -> None:
        """Draw one orientation panel."""
        ax.clear()
        ax.set_facecolor("black")
        ax.imshow(self._get_slice_2d(view, sl),
                  cmap="gray", vmin=0, vmax=1,
                  interpolation="nearest", aspect="equal")
        ax.axis("off")
        if self.show_overlays:
            self._draw_cross_view_overlays(ax, view, sl, compact=not is_main)
        self._draw_current_view_points(ax, view, sl, compact=not is_main)
        if title:
            ax.set_title(title, fontsize=9, color=ORIENT_COLORS[view], pad=3)
        if apply_zoom:
            self._apply_zoom(ax, view)
        else:
            width, height = self._display_shape(view)
            ax.set_xlim(-0.5, width - 0.5)
            ax.set_ylim(height - 0.5, -0.5)

    def _draw_current_view_points(self, ax, view: str, sl: int, compact: bool = False) -> None:
        """Draw saved points from the active view and highlight the exact pixel."""
        o_idx = ORIENT_KEYS.index(view)
        for p_idx, click in enumerate(self.clicks[o_idx]):
            if click is None:
                continue
            a, b, s = click
            if s != sl:
                continue
            pdef = POINT_DEFS[p_idx]
            self._draw_pixel_marker(
                ax,
                a, b,
                label=pdef["id"],
                color="#00ffff",
                edgecolor="black",
                alpha=1.0,
                zorder=7,
                compact=compact,
            )

    def _draw_cross_view_overlays(self, ax, view: str, sl: int, compact: bool = False) -> None:
        """Project points from other orientations into the currently shown view."""
        target_o = ORIENT_KEYS.index(view)
        for source_o, source_view in enumerate(ORIENT_KEYS):
            if source_o == target_o:
                continue
            source_points = self.clicks[source_o]
            if not any(c is not None for c in source_points):
                continue
            for p_idx, click in enumerate(source_points):
                if click is None:
                    continue
                xyz = self._click_to_xyz(source_o, click)
                a, b, target_slice = self._xyz_to_view_display(view, xyz)
                if not self._display_point_in_bounds(view, a, b):
                    continue
                on_slice = int(round(target_slice)) == int(sl)
                color = ORIENT_COLORS[source_view]
                alpha = 0.90 if on_slice else 0.38
                ax.plot(
                    a, b, marker="x", markersize=9 if compact else 13,
                    markeredgewidth=2.2, color=color, alpha=alpha, zorder=5)
                ax.add_patch(plt.Circle(
                    (a, b), 5 if compact else 7, fill=False, linestyle="--",
                    edgecolor=color, linewidth=1.6, alpha=alpha, zorder=5))
                if not compact:
                    ax.text(
                        a + 9, b + 9,
                        f"{source_view[:3]} {POINT_DEFS[p_idx]['id']} s={target_slice:.0f}",
                        color=color, fontsize=7.5, alpha=alpha, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.12",
                                  facecolor="white", edgecolor="none", alpha=0.62))

    def _draw_pixel_marker(
        self,
        ax,
        a: float,
        b: float,
        label: str,
        color: str,
        edgecolor: str,
        alpha: float,
        zorder: int,
        compact: bool = False,
    ) -> None:
        """Draw a visible marker whose square is exactly one image pixel."""
        ax.add_patch(plt.Rectangle(
            (a - 0.5, b - 0.5), 1.0, 1.0,
            fill=False, edgecolor="#ffff00", linewidth=2.0,
            alpha=alpha, zorder=zorder + 2))
        ax.plot(
            a, b, "+", color=color, markersize=10 if compact else 16,
            markeredgewidth=2.4, alpha=alpha, zorder=zorder + 1)
        ax.add_patch(plt.Circle(
            (a, b), 5 if compact else 8, fill=False, edgecolor=color,
            linewidth=1.8, alpha=alpha, zorder=zorder))
        if not compact:
            ax.text(
                a + 10, b - 10, label, color=color, fontsize=8.5,
                fontweight="bold", alpha=alpha, zorder=zorder + 3,
                bbox=dict(boxstyle="round,pad=0.12",
                          facecolor=edgecolor, edgecolor=color, alpha=0.72))

    def _apply_zoom(self, ax, view: str) -> None:
        """Crop the visible panel around the zoom center without changing data."""
        width, height = self._display_shape(view)
        if self.zoom_factor <= 1.001:
            ax.set_xlim(-0.5, width - 0.5)
            ax.set_ylim(height - 0.5, -0.5)
            return
        if self.zoom_center is None:
            cx = (width - 1) / 2.0
            cy = (height - 1) / 2.0
        else:
            cx, cy = self.zoom_center
        half_w = width / (2.0 * self.zoom_factor)
        half_h = height / (2.0 * self.zoom_factor)
        xmin = float(np.clip(cx - half_w, -0.5, max(-0.5, width - 2 * half_w - 0.5)))
        xmax = xmin + 2 * half_w
        ymin = float(np.clip(cy - half_h, -0.5, max(-0.5, height - 2 * half_h - 0.5)))
        ymax = ymin + 2 * half_h
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)

    def _update_point_list(self) -> None:
        """Show saved clicks in a side list instead of drawing them over the scan."""
        if self.ax_list is None:
            return
        self.ax_list.clear()
        self.ax_list.set_axis_off()
        self.ax_list.add_patch(plt.Rectangle(
            (0.0, 0.0), 1.0, 1.0,
            transform=self.ax_list.transAxes,
            fill=False, edgecolor="#444", linewidth=1.2))

        self.ax_list.text(0.05, 0.95, "Voxel coordinates", transform=self.ax_list.transAxes,
                          ha="left", va="top", fontsize=12.5, fontweight="bold",
                          color="#333")
        self.ax_list.text(
            0.05, 0.85,
            "Check P1-P4 values are\nrelatively similar across views.",
            transform=self.ax_list.transAxes,
            ha="left", va="top", fontsize=9.4, color="#555")
        lines = ["View Pt      X      Y      Z"]
        for row in self._ordered_click_rows(include_empty=True):
            view = row["view"][:3].capitalize()
            pid = row["point_id"]
            if row["x_voxel"] == "":
                lines.append(f"{view:<4} {pid:<2}      -      -      -")
            else:
                lines.append(
                    f"{view:<4} {pid:<2} {row['x_voxel']:>6.1f}"
                    f" {row['y_voxel']:>6.1f} {row['z_voxel']:>6.1f}")
        placed = sum(c is not None for row in self.clicks for c in row)
        lines.append(f"Total {placed}/12")

        self.ax_list.text(0.05, 0.64, "\n".join(lines),
                          transform=self.ax_list.transAxes,
                          ha="left", va="top", fontsize=9.0, color="#333",
                          family="monospace", linespacing=1.03)

    # ─────────────────────────────────────────────────────────────────────
    # GUIDE DIAGRAM
    # ─────────────────────────────────────────────────────────────────────

    def _update_diagram(self) -> None:
        """Redraw the right-side prism guide and its point progress labels."""
        ax = self.ax_guide
        ax.clear()
        ax.set_facecolor("#f7f4e8")
        ax.set_axis_off()
        ax.set_title("Microprism Guide", fontsize=10, pad=4, color="#333")

        if self.guide_image is not None:
            guide_h, guide_w = self.guide_image.shape[:2]
            ax.imshow(self.guide_image, extent=(0, guide_w, guide_h, 0),
                      aspect="equal")
            ax.set_xlim(0, guide_w)
            ax.set_ylim(guide_h, 0)
        else:
            guide_w, guide_h = 441, 803
            ax.set_xlim(0, guide_w)
            ax.set_ylim(guide_h, 0)
            ax.text(guide_w / 2, guide_h / 2, "Guide image not found", ha="center", va="center",
                    fontsize=10, color="#444")

        # Pixel positions from the source 441 x 803 guide image. These sit
        # on the actual line intersections of the prism drawing.
        blue_pos = {
            "P1": (20, 802),
            "P2": (300, 802),
            "P3": (150, 417),
            "P4": (429, 417),
        }

        next_pid = (POINT_DEFS[self.current_point]["id"]
                    if self.current_orient < 3 else None)

        for i, pdef in enumerate(POINT_DEFS):
            pos    = blue_pos[pdef["id"]]
            n_done = sum(self.clicks[o][i] is not None for o in range(3))
            is_next = (pdef["id"] == next_pid)
            all3    = (n_done == 3)
            clicked_this_phase = (
                self.current_orient < 3
                and self.clicks[self.current_orient][i] is not None
            )

            col  = "#ff8c00" if is_next else pdef["color"]
            sz   = 135      if is_next else (70     if all3 else 110)
            alp  = 1.0      if is_next else (0.45   if all3 else 0.9)
            ec   = "black"  if is_next else ("gray"  if all3 else "white")
            lw   = 2.5      if is_next else (1.0     if all3 else 1.5)

            if clicked_this_phase:
                ax.scatter(*pos, c="#ffdf4d", s=255, alpha=0.95, zorder=9,
                           edgecolors="black", linewidths=1.4, clip_on=False)
            ax.scatter(*pos, c=col, s=sz, alpha=alp, zorder=10,
                       edgecolors=ec, linewidths=lw, clip_on=False)
            tx = pos[0] - 58 if pos[0] > 360 else pos[0] + 12
            ty = pos[1] - 28 if pos[1] > 720 else pos[1] - 16
            ax.text(tx, ty, f"{pdef['id']} ({n_done}/3)",
                    color="#ff8c00" if is_next else pdef["color"],
                    fontsize=8, fontweight="bold", alpha=alp,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                              edgecolor="none", alpha=0.7),
                    clip_on=False)

        self.fig.canvas.draw_idle()

    # ─────────────────────────────────────────────────────────────────────
    # VIEW SWITCHING — matches midline change_to_view exactly
    # ─────────────────────────────────────────────────────────────────────

    def _switch_view(self, view: str) -> None:
        """Switch views and restore the last slice used in the selected view."""
        self.slice_indices[self.current_view] = self.current_slice
        self.current_view  = view
        self.current_slice = self.slice_indices[view]
        self.zoom_center = None
        self._pan_anchor = None
        max_sl = self._max_slice(view)

        if self.slider is None:
            return

        self.slider.valmin = 0
        self.slider.valmax = max_sl
        self.slider.ax.set_xlim(0, max_sl)
        self.slider.label.set_text(f"{view.capitalize()} slice")
        self._set_slider_val(self.current_slice)

        print(f"\n  Switched to {ORIENT_NAMES[view]}")
        self._update_display()
        self._update_diagram()

    # ─────────────────────────────────────────────────────────────────────
    # EVENTS
    # ─────────────────────────────────────────────────────────────────────

    def _on_click(self, event) -> None:
        """Record the next point when the user clicks inside the brain image."""
        for preview_view, ax_prev in self.preview_axes.items():
            if event.inaxes is ax_prev and ax_prev.get_visible():
                self._switch_view(preview_view)
                return
        if event.inaxes is not self.ax:
            return
        if event.button != 1 or event.xdata is None:
            return
        if self.pan_mode:
            self._pan_anchor = (
                float(event.xdata), float(event.ydata),
                tuple(self.ax.get_xlim()), tuple(self.ax.get_ylim()),
            )
            return
        if self.current_orient >= 3:
            print("  All points placed — click Save & Quit.")
            return

        active = ORIENT_KEYS[self.current_orient]
        if self.current_view != active:
            print(f"  Switch to {ORIENT_NAMES[active]} to place next point.")
            return

        a, b = int(round(event.xdata)), int(round(event.ydata))
        a, b = self._clamp(self.current_view, a, b)
        self.zoom_center = (float(a), float(b))
        s    = int(self.current_slice)
        o    = ORIENT_KEYS.index(self.current_view)

        self.clicks[o][self.current_point] = (a, b, s)
        pdef = POINT_DEFS[self.current_point]
        print(f"  [{ORIENT_NAMES[self.current_view]}] "
              f"{pdef['id']} → ({a},{b}) slice={s}")
        self.flash_marker = (self.current_view, a, b, s, pdef["id"])

        self.current_point += 1
        if self.current_point == 4:
            self.current_point  = 0
            self.current_orient += 1
            if self.current_orient < 3:
                nv = ORIENT_KEYS[self.current_orient]
                print(f"\n  → Phase {self.current_orient+1}: "
                      f"{ORIENT_NAMES[nv]}")
                print(f"     Click [{nv.capitalize()}] button to switch")
            else:
                print("\n  ✓ All 12 clicks done — click Save & Quit")

        self._save_warned = False
        self._update_display()
        self._update_diagram()
        self._start_flash_timer()

    def _start_flash_timer(self) -> None:
        """Briefly add an extra highlight around the most recently clicked point."""
        if self.fig is None:
            return
        timer = self.fig.canvas.new_timer(interval=700)
        timer.single_shot = True
        timer.add_callback(self._clear_flash_marker)
        self._flash_timer = timer
        timer.start()

    def _clear_flash_marker(self) -> None:
        self.flash_marker = None
        self._update_display()
        return False

    def _on_slider(self, val) -> None:
        """Handle slice-slider movement without causing a feedback loop."""
        if self._updating_slider:
            return
        self.current_slice = int(round(val))
        self.slice_indices[self.current_view] = self.current_slice
        self._sync_other_slices_from_current()
        self._update_display()

    def _on_scroll(self, event) -> None:
        """Use the mouse wheel to move one slice at a time."""
        if event.inaxes is not self.ax and event.inaxes not in self.preview_axes.values():
            return
        if event.inaxes in self.preview_axes.values():
            for preview_view, ax_prev in self.preview_axes.items():
                if event.inaxes is ax_prev and ax_prev.get_visible():
                    self._switch_view(preview_view)
                    break
        if event.key in ("control", "ctrl", "cmd", "super"):
            scale = 1.25 if event.button == "up" else 1/1.25
            if event.xdata is not None and event.ydata is not None:
                self.zoom_center = (float(event.xdata), float(event.ydata))
            self._change_zoom(scale)
            return
        max_sl = self._max_slice(self.current_view)
        step   = 1 if event.button == "up" else -1
        self.current_slice = int(np.clip(self.current_slice + step, 0, max_sl))
        self.slice_indices[self.current_view] = self.current_slice
        self._sync_other_slices_from_current()
        self._set_slider_val(self.current_slice)
        self._update_display()

    def _on_motion(self, event) -> None:
        """Drag the zoomed image when pan mode is enabled."""
        if self._pan_anchor is None or event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        x0, y0, xlim0, ylim0 = self._pan_anchor
        dx = float(event.xdata) - x0
        dy = float(event.ydata) - y0
        width, height = self._display_shape(self.current_view)
        new_xlim = (xlim0[0] - dx, xlim0[1] - dx)
        new_ylim = (ylim0[0] - dy, ylim0[1] - dy)
        new_xlim = self._clamp_limits(new_xlim, width)
        new_ylim = self._clamp_limits(new_ylim, height)
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.zoom_center = (
            (new_xlim[0] + new_xlim[1]) / 2.0,
            (new_ylim[0] + new_ylim[1]) / 2.0,
        )
        self.fig.canvas.draw_idle()

    def _on_release(self, event) -> None:
        """Stop panning after mouse release."""
        self._pan_anchor = None

    def _on_key(self, event) -> None:
        """Keyboard shortcuts for view switching, slice navigation, undo, save."""
        k = (event.key or "").lower()
        if k == "c":
            self._switch_view("coronal")
        elif k == "s":
            self._switch_view("sagittal")
        elif k == "a":
            self._switch_view("axial")
        elif k in ("right", "up"):
            self.current_slice = min(
                self.current_slice + 1, self._max_slice(self.current_view))
            self.slice_indices[self.current_view] = self.current_slice
            self._sync_other_slices_from_current()
            self._set_slider_val(self.current_slice)
            self._update_display()
        elif k in ("left", "down"):
            self.current_slice = max(self.current_slice - 1, 0)
            self.slice_indices[self.current_view] = self.current_slice
            self._sync_other_slices_from_current()
            self._set_slider_val(self.current_slice)
            self._update_display()
        elif k == "u":
            self._on_undo(None)
        elif k in ("o",):
            self._on_toggle_overlays(None)
        elif k in ("p",):
            self._on_toggle_pan(None)
        elif k in ("+", "="):
            self._change_zoom(1.5)
        elif k in ("-", "_"):
            self._change_zoom(1/1.5)
        elif k in ("0",):
            self._reset_zoom()
        elif k in ("enter", "return", "ctrl+s", "cmd+s", "super+s"):
            self._on_save(None)

    def _on_toggle_overlays(self, _e) -> None:
        """Show/hide points selected in the other orientations."""
        self.show_overlays = not self.show_overlays
        if "overlay" in self.buttons:
            self.buttons["overlay"].label.set_text(
                "Hide overlays" if self.show_overlays else "Show overlays")
        print("  Cross-view overlays "
              f"{'shown' if self.show_overlays else 'hidden'}.")
        self._update_display()

    def _on_toggle_pan(self, _e) -> None:
        """Turn click-drag image panning on/off."""
        self.pan_mode = not self.pan_mode
        self._pan_anchor = None
        if "pan" in self.buttons:
            self.buttons["pan"].label.set_text("Pan on" if self.pan_mode else "Pan off")
        print(f"  Pan mode {'on' if self.pan_mode else 'off'}.")
        self._update_display()

    def _change_zoom(self, scale: float) -> None:
        """Zoom the display only; saved voxel coordinates are unchanged."""
        self.zoom_factor = float(np.clip(self.zoom_factor * scale, 1.0, 24.0))
        if self.zoom_center is None:
            self.zoom_center = self._default_zoom_center(self.current_view)
        self._update_display()

    def _reset_zoom(self) -> None:
        """Return to full-slice view."""
        self.zoom_factor = 1.0
        self.zoom_center = None
        self._update_display()

    def _on_undo(self, _e) -> None:
        """Undo the most recent point and return to the relevant view."""
        if self.current_orient == 0 and self.current_point == 0:
            print("  Nothing to undo.")
            return
        if self.current_point == 0:
            self.current_orient -= 1
            self.current_point   = 3
        else:
            self.current_point  -= 1
        self.clicks[self.current_orient][self.current_point] = None
        active = ORIENT_KEYS[self.current_orient]
        print(f"  Undid {POINT_DEFS[self.current_point]['id']} "
              f"in {ORIENT_NAMES[active]}")
        self._switch_view(active)

    def _on_save(self, _e) -> None:
        """Save results, warning once before allowing a partial/incomplete save."""
        placed = sum(c is not None for row in self.clicks for c in row)
        if placed < 12 and not self._save_warned:
            print(f"  WARNING: {placed}/12 clicks placed. "
                  "Click Save & Quit again to confirm partial save.")
            self._save_warned = True
            return
        try:
            self._save_results()
        except Exception as exc:
            print(f"\n  ERROR: could not save microprism corners: {exc}")
            print("  The window is staying open so you can try again after fixing the issue.")
            return
        plt.close(self.fig)

    # ─────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────

    def _max_slice(self, view: str) -> int:
        """Return the maximum slice index for the selected view."""
        return {"coronal": self.nz-1, "sagittal": self.nx-1,
                "axial": self.ny-1}[view]

    def _get_slice_2d(self, view: str, idx: int) -> np.ndarray:
        """Extract the 2D image displayed for a view/slice combination."""
        if view == "coronal":   return self.img_disp[idx, :, :]
        if view == "sagittal":  return self.img_disp[:, :, idx]
        if view == "axial":     return self.img_disp[:, idx, :]
        raise ValueError(view)

    def _display_shape(self, view: str) -> Tuple[int, int]:
        """Return displayed slice width/height for a view."""
        arr = self._get_slice_2d(view, self.current_slice)
        height, width = arr.shape
        return int(width), int(height)

    def _sync_other_slices_from_current(self) -> None:
        """Move preview slices to the same relative depth as the active view."""
        current_max = max(1, self._max_slice(self.current_view))
        frac = self.current_slice / current_max
        for view in ORIENT_KEYS:
            if view == self.current_view:
                continue
            self.slice_indices[view] = int(round(frac * self._max_slice(view)))

    @staticmethod
    def _clamp_limits(limits: Tuple[float, float], size: int) -> Tuple[float, float]:
        """Keep panned axis limits inside image bounds while preserving direction."""
        lo, hi = limits
        reversed_axis = lo > hi
        low, high = (hi, lo) if reversed_axis else (lo, hi)
        span = high - low
        full_low = -0.5
        full_high = size - 0.5
        if span >= size:
            clamped = (full_low, full_high)
        else:
            low = float(np.clip(low, full_low, full_high - span))
            clamped = (low, low + span)
        return (clamped[1], clamped[0]) if reversed_axis else clamped

    def _default_zoom_center(self, view: str) -> Tuple[float, float]:
        """Use a clicked point in this view if possible, otherwise image center."""
        o_idx = ORIENT_KEYS.index(view)
        for click in reversed(self.clicks[o_idx]):
            if click is not None and click[2] == self.current_slice:
                return float(click[0]), float(click[1])
        width, height = self._display_shape(view)
        return (width - 1) / 2.0, (height - 1) / 2.0

    def _clamp(self, view: str, a: int, b: int) -> Tuple[int, int]:
        """Clamp a raw display click to the valid bounds for the current view."""
        if view == "coronal":
            return int(np.clip(a,0,self.nx-1)), int(np.clip(b,0,self.ny-1))
        if view == "sagittal":
            return int(np.clip(a,0,self.ny-1)), int(np.clip(b,0,self.nz-1))
        if view == "axial":
            return int(np.clip(a,0,self.nx-1)), int(np.clip(b,0,self.nz-1))
        raise ValueError(view)

    def _click_to_xyz(self, o: int, click: Click) -> XYZ:
        """Convert a stored view-specific click into x/y/z voxel coordinates."""
        a, b, s = click
        if o == 0: return float(a), float(b), float(s)
        if o == 1: return float(s), float(a), float(b)
        if o == 2: return float(a), float(s), float(b)
        raise ValueError(o)

    def _xyz_to_view_display(self, view: str, xyz: XYZ) -> Tuple[float, float, float]:
        """Project x/y/z voxel coordinates into one displayed orientation."""
        x, y, z = xyz
        if view == "coronal":
            return float(x), float(y), float(z)
        if view == "sagittal":
            return float(y), float(z), float(x)
        if view == "axial":
            return float(x), float(z), float(y)
        raise ValueError(view)

    def _display_point_in_bounds(self, view: str, a: float, b: float) -> bool:
        """Check whether projected display coordinates fit the current panel."""
        width, height = self._display_shape(view)
        return 0 <= a <= width - 1 and 0 <= b <= height - 1

    def _ordered_click_rows(self, include_empty: bool = False) -> List[Dict]:
        """Return raw clicked voxel coordinates ordered by view, then P1-P4."""
        rows = []
        for o, view in enumerate(ORIENT_KEYS):
            for p, pdef in enumerate(POINT_DEFS):
                c = self.clicks[o][p]
                if c is None:
                    if include_empty:
                        rows.append({
                            "view": view,
                            "point_id": pdef["id"],
                            "point_number": p + 1,
                            "x_voxel": "",
                            "y_voxel": "",
                            "z_voxel": "",
                            "display_a": "",
                            "display_b": "",
                            "slice_index": "",
                        })
                    continue
                x, y, z = self._click_to_xyz(o, c)
                rows.append({
                    "view": view,
                    "point_id": pdef["id"],
                    "point_number": p + 1,
                    "x_voxel": x,
                    "y_voxel": y,
                    "z_voxel": z,
                    "display_a": c[0],
                    "display_b": c[1],
                    "slice_index": c[2],
                })
        return rows

    def _set_slider_val(self, v: int) -> None:
        """Set the slider programmatically while blocking recursive callbacks."""
        if self.slider is None:
            return
        if int(round(self.slider.val)) == v:
            return
        self._updating_slider = True
        try:
            self.slider.set_val(v)
        finally:
            self._updating_slider = False

    def _average_point(self, p: int) -> Optional[Dict]:
        """Average a P1/P2/P3/P4 point across all completed orientations."""
        coords = [self._click_to_xyz(o, self.clicks[o][p])
                  for o in range(3) if self.clicks[o][p] is not None]
        if not coords:
            return None
        x, y, z = np.array(coords, dtype=float).mean(axis=0)
        x_mm, y_mm, z_mm = self._xyz_to_mm(np.array((x, y, z), dtype=float))
        return {"x": round(x,3), "y": round(y,3), "z": round(z,3),
                "x_mm": round(x_mm,5),
                "y_mm": round(y_mm,5),
                "z_mm": round(z_mm,5),
                "num_orientation_clicks": len(coords)}

    def _xyz_to_mm(self, xyz: np.ndarray) -> np.ndarray:
        """Convert voxel coordinates in x/y/z order to physical millimeters."""
        return np.asarray(xyz, dtype=float) * self.spacing_xyz

    def _mm_to_xyz(self, xyz_mm: np.ndarray) -> np.ndarray:
        """Convert physical millimeters in x/y/z order to voxel coordinates."""
        return np.asarray(xyz_mm, dtype=float) / self.spacing_xyz

    @staticmethod
    def _midpt(pa, pb):
        if not pa or not pb:
            return None
        return {k: round((pa[k]+pb[k])/2, 5 if k.endswith("_mm") else 3)
                for k in ["x","y","z","x_mm","y_mm","z_mm"]}

    @staticmethod
    def _cell_ref(row: int, col: int) -> str:
        letters = ""
        while col:
            col, rem = divmod(col - 1, 26)
            letters = chr(65 + rem) + letters
        return f"{letters}{row}"

    @classmethod
    def _sheet_xml(cls, rows: List[List]) -> str:
        row_xml = []
        for r_idx, row in enumerate(rows, start=1):
            cells = []
            for c_idx, value in enumerate(row, start=1):
                ref = cls._cell_ref(r_idx, c_idx)
                if value is None or value == "":
                    cells.append(f'<c r="{ref}"/>')
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    cells.append(f'<c r="{ref}"><v>{float(value):g}</v></c>')
                else:
                    text = escape(str(value))
                    cells.append(
                        f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>')
            row_xml.append(f'<row r="{r_idx}">{"".join(cells)}</row>')
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" '
            'topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
            '</sheetView></sheetViews>'
            '<cols><col min="1" max="1" width="14" customWidth="1"/>'
            '<col min="2" max="2" width="14" customWidth="1"/>'
            '<col min="3" max="4" width="10" customWidth="1"/>'
            '<col min="5" max="14" width="13" customWidth="1"/></cols>'
            f'<sheetData>{"".join(row_xml)}</sheetData>'
            '</worksheet>'
        )

    @classmethod
    def _write_simple_xlsx(cls, output_path: Path, rows: List[List]) -> None:
        """Write a simple one-sheet XLSX using only the standard library."""
        sheet_xml = cls._sheet_xml(rows)
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml",
                       '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                       '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                       '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                       '<Default Extension="xml" ContentType="application/xml"/>'
                       '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                       '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                       '</Types>')
            z.writestr("_rels/.rels",
                       '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                       '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                       '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
                       '</Relationships>')
            z.writestr("xl/workbook.xml",
                       '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                       '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                       'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                       '<sheets><sheet name="Voxel Coordinates" sheetId="1" r:id="rId1"/></sheets>'
                       '</workbook>')
            z.writestr("xl/_rels/workbook.xml.rels",
                       '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                       '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                       '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
                       '</Relationships>')
            z.writestr("xl/worksheets/sheet1.xml", sheet_xml)

    def _point_dict_from_vec(self, v: np.ndarray) -> Dict:
        """Convert an x/y/z vector to the same voxel/mm dict used by clicks."""
        x, y, z = [float(n) for n in v]
        x_mm, y_mm, z_mm = self._xyz_to_mm(np.array((x, y, z), dtype=float))
        return {"x": round(x,3), "y": round(y,3), "z": round(z,3),
                "x_mm": round(float(x_mm),5),
                "y_mm": round(float(y_mm),5),
                "z_mm": round(float(z_mm),5),
                "num_orientation_clicks": "derived"}

    @staticmethod
    def _safe_unit(vector: np.ndarray) -> Optional[np.ndarray]:
        length = float(np.linalg.norm(vector))
        if length < 1e-8:
            return None
        return vector / length

    @staticmethod
    def _right_angle_candidates(
        base_mm: np.ndarray,
        outer_mm: np.ndarray,
        width_unit: np.ndarray,
        prism_length_mm: float,
    ) -> Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """Solve D for base-D perpendicular to D-outer with known D-outer length.

        The solve happens in the cross-section perpendicular to the prism width.
        Small width-axis drift between base and outer is treated as click noise.
        """
        diagonal = outer_mm - base_mm
        cross_section_diagonal = (
            diagonal - np.dot(diagonal, width_unit) * width_unit
        )
        diagonal_len = float(np.linalg.norm(cross_section_diagonal))
        length_mm = float(prism_length_mm)
        if diagonal_len < 1e-8 or length_mm <= 0 or length_mm >= diagonal_len:
            return None

        diagonal_unit = cross_section_diagonal / diagonal_len
        side_unit = MicroprismCornerTracker._safe_unit(np.cross(width_unit, diagonal_unit))
        if side_unit is None:
            return None

        height_mm = float(np.sqrt(diagonal_len ** 2 - length_mm ** 2))
        length_along_diagonal = (length_mm ** 2) / diagonal_len
        length_across_diagonal = (length_mm * height_mm) / diagonal_len
        candidates = []
        for sign in (1.0, -1.0):
            length_vec = (
                length_along_diagonal * diagonal_unit
                + sign * length_across_diagonal * side_unit
            )
            height_vec = cross_section_diagonal - length_vec
            derived_mm = base_mm + height_vec
            candidates.append((derived_mm, height_vec, length_vec))
        return candidates

    def _derive_imaging_face(self, green: Dict[str, Optional[Dict]]) -> Dict[str, Optional[Dict]]:
        """Derive the four red imaging-face points from the four clicked points.

        P1 and P2 are shared bottom imaging-face corners. P3 and P4 are not on
        the imaging face; they sit across the prism length from D1 and D2. Each
        side is solved as a right triangle where P1-D1/P2-D2 is the imaging-face
        height, D1-P3/D2-P4 is the known prism length, and the angle at D1/D2 is
        90 degrees.
        """
        if not all(green.get(pid) for pid in ("P1", "P2", "P3", "P4")):
            return {
                "shared_left_P1": green.get("P1"),
                "shared_right_P2": green.get("P2"),
                "derived_left": None,
                "derived_right": None,
                "_diagnostics": {},
            }

        p1 = self._xyz_to_mm(
            np.array([green["P1"][k] for k in ("x", "y", "z")], dtype=float)
        )
        p2 = self._xyz_to_mm(
            np.array([green["P2"][k] for k in ("x", "y", "z")], dtype=float)
        )
        p3 = self._xyz_to_mm(
            np.array([green["P3"][k] for k in ("x", "y", "z")], dtype=float)
        )
        p4 = self._xyz_to_mm(
            np.array([green["P4"][k] for k in ("x", "y", "z")], dtype=float)
        )

        width_hint = 0.5 * ((p2 - p1) + (p4 - p3))
        width_unit = self._safe_unit(width_hint)
        diagnostics = {
            "method": "right_angle_cross_section_solve",
            "input_points_mm": {
                "P1": [float(v) for v in p1],
                "P2": [float(v) for v in p2],
                "P3": [float(v) for v in p3],
                "P4": [float(v) for v in p4],
            },
            "prism_length_mm": float(self.prism_length_mm),
        }

        if width_unit is None:
            red_top_left_mm = None
            red_top_right_mm = None
            diagnostics["warning"] = "Could not compute width direction from P1/P2/P3/P4."
        else:
            diagnostics["width_unit"] = [float(v) for v in width_unit]
            left_candidates = self._right_angle_candidates(
                p1, p3, width_unit, self.prism_length_mm
            )
            right_candidates = self._right_angle_candidates(
                p2, p4, width_unit, self.prism_length_mm
            )
            if left_candidates is None or right_candidates is None:
                print(
                    "WARNING: Could not solve right-angle microprism geometry. "
                    "Check that the prism length is shorter than the P1-P3/P2-P4 "
                    "diagonal distance."
                )
                red_top_left_mm = None
                red_top_right_mm = None
                diagnostics["warning"] = (
                    "Could not solve right-angle geometry. The prism length must "
                    "be shorter than the observed P1-P3/P2-P4 cross-section diagonal."
                )
            else:
                # The expected imaging face is shallow in sagittal view, so choose
                # the solution with the smallest Z change along P1-D1/P2-D2.
                scores = [
                    abs(left_candidates[i][1][2]) + abs(right_candidates[i][1][2])
                    for i in range(2)
                ]
                best_index = int(np.argmin(scores))
                red_top_left_mm = left_candidates[best_index][0]
                red_top_right_mm = right_candidates[best_index][0]
                diagnostics["candidate_scores_abs_height_z_mm"] = [
                    float(v) for v in scores
                ]
                diagnostics["chosen_candidate_index"] = best_index

        if red_top_left_mm is None or red_top_right_mm is None:
            return {
                "shared_left_P1": green["P1"],
                "shared_right_P2": green["P2"],
                "derived_left": None,
                "derived_right": None,
                "_diagnostics": diagnostics,
            }

        red_top_left = self._mm_to_xyz(red_top_left_mm)
        red_top_right = self._mm_to_xyz(red_top_right_mm)
        height_left = red_top_left_mm - p1
        height_right = red_top_right_mm - p2
        prism_len_left = p3 - red_top_left_mm
        prism_len_right = p4 - red_top_right_mm
        bottom_width = p2 - p1
        top_width = red_top_right_mm - red_top_left_mm
        diagnostics.update({
            "derived_points_mm": {
                "D1": [float(v) for v in red_top_left_mm],
                "D2": [float(v) for v in red_top_right_mm],
            },
            "vectors_mm": {
                "P1_to_D1": [float(v) for v in height_left],
                "P2_to_D2": [float(v) for v in height_right],
                "D1_to_P3": [float(v) for v in prism_len_left],
                "D2_to_P4": [float(v) for v in prism_len_right],
                "P1_to_P2": [float(v) for v in bottom_width],
                "D1_to_D2": [float(v) for v in top_width],
            },
            "length_checks_mm": {
                "P1_to_D1_height": float(np.linalg.norm(height_left)),
                "P2_to_D2_height": float(np.linalg.norm(height_right)),
                "D1_to_P3_prism_length": float(np.linalg.norm(prism_len_left)),
                "D2_to_P4_prism_length": float(np.linalg.norm(prism_len_right)),
                "P1_to_P2_width": float(np.linalg.norm(bottom_width)),
                "D1_to_D2_width": float(np.linalg.norm(top_width)),
                "P3_to_P4_width_reference": float(np.linalg.norm(p4 - p3)),
            },
            "right_angle_dot_products": {
                "dot_P1D1_D1P3": float(np.dot(height_left, prism_len_left)),
                "dot_P2D2_D2P4": float(np.dot(height_right, prism_len_right)),
            },
        })

        return {
            "shared_left_P1": green["P1"],
            "shared_right_P2": green["P2"],
            "derived_left": self._point_dict_from_vec(red_top_left),
            "derived_right": self._point_dict_from_vec(red_top_right),
            "_diagnostics": diagnostics,
        }

    # ─────────────────────────────────────────────────────────────────────
    # SAVE — JSON + CSV
    # ─────────────────────────────────────────────────────────────────────

    def _save_results(self) -> None:
        """Write JSON + CSV outputs and print a terminal summary."""
        # ── Build data ────────────────────────────────────────────────────
        per = {}
        clicked_points_list = self._ordered_click_rows(include_empty=False)
        for o, view in enumerate(ORIENT_KEYS):
            per[view] = {}
            for p, pdef in enumerate(POINT_DEFS):
                c = self.clicks[o][p]
                if c:
                    x,y,z = self._click_to_xyz(o, c)
                    x_mm, y_mm, z_mm = self._xyz_to_mm(np.array([x, y, z], dtype=float))
                    per[view][pdef["id"]] = {
                        "x": x, "y": y, "z": z,
                        "x_mm": round(float(x_mm), 5),
                        "y_mm": round(float(y_mm), 5),
                        "z_mm": round(float(z_mm), 5),
                        "display_a": c[0],
                        "display_b": c[1],
                        "slice_index": c[2],
                    }

        green = {pdef["id"]: self._average_point(i)
                 for i,pdef in enumerate(POINT_DEFS)}
        imaging_face = self._derive_imaging_face(green)
        imaging_face_points = {
            key: value
            for key, value in imaging_face.items()
            if not key.startswith("_")
        }
        if not imaging_face.get("derived_left") or not imaging_face.get("derived_right"):
            print(
                "WARNING: D1/D2 imaging-plane points were not derived. "
                "The clicked points will be saved, but atlas mapper needs valid "
                "derived points before it can draw the imaging plane."
            )

        # ── Save JSON ─────────────────────────────────────────────────────
        json_output = {
            "source_image":       str(self.image_path),
            "image_shape_zyx":    [self.nz, self.ny, self.nx],
            "spacing_mm":         self.spacing,
            "spacing_mm_xyz":     [float(v) for v in self.spacing_xyz],
            "prism_width_mm":     self.prism_width_mm,
            "prism_length_mm":    self.prism_length_mm,
            "coordinate_convention": "(Z,Y,X) volume; x,y,z are voxel coords",
            "view_convention": {
                "coronal":  "slice through Z; display rows=Y, cols=X",
                "sagittal": "slice through X; display rows=Z, cols=Y",
                "axial":    "slice through Y; display rows=Z, cols=X",
            },
            "blue_clicked_points": green,
            "green_points":    green,
            "derived_points":  {
                "derived_left": imaging_face.get("derived_left"),
                "derived_right": imaging_face.get("derived_right"),
            },
            "imaging_face_points": imaging_face_points,
            "clicked_points_list": clicked_points_list,
            "imaging_face_math": (
                "P1/P2 are the shared bottom imaging-plane corners. P3/P4 are "
                "clicked prism points that are not on the imaging plane. D1/D2 are "
                "derived by solving each side cross-section as a right triangle: "
                "P1-D1 and P2-D2 are imaging-plane height, D1-P3 and D2-P4 are the "
                "known prism length, and the angles at D1/D2 are 90 degrees. The "
                "solve is done in physical millimeter space, then converted back to "
                "voxel coordinates."
            ),
            "imaging_face_diagnostics": imaging_face.get("_diagnostics", {}),
            "per_orientation": per,
        }
        json_path = self.output_dir / "microprism_corners.json"
        with open(json_path, "w") as f:
            json.dump(json_output, f, indent=2)

        # ── Save CSV ──────────────────────────────────────────────────────
        csv_path = self.output_dir / "microprism_corners.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            # Header
            writer.writerow([
                "point_id", "type", "description",
                "x_voxel", "y_voxel", "z_voxel",
                "x_mm", "y_mm", "z_mm",
                "num_orientation_clicks",
                # Per-orientation raw voxels
                "coronal_x", "coronal_y", "coronal_z",
                "sagittal_x", "sagittal_y", "sagittal_z",
                "axial_x", "axial_y", "axial_z",
            ])
            # Blue clicked points
            for pdef in POINT_DEFS:
                pid = pdef["id"]
                v   = green.get(pid)
                row = [pid, "blue_clicked", pdef["desc"]]
                if v:
                    row += [v["x"], v["y"], v["z"],
                            v["x_mm"], v["y_mm"], v["z_mm"],
                            v["num_orientation_clicks"]]
                else:
                    row += ["","","","","","",""]
                # Per-orientation raw
                for view in ORIENT_KEYS:
                    raw = per.get(view, {}).get(pid)
                    if raw:
                        row += [raw["x"], raw["y"], raw["z"]]
                    else:
                        row += ["","",""]
                writer.writerow(row)
            # Red imaging-face points
            for rid, rdata in imaging_face_points.items():
                desc = rid.replace("_", " ")
                row = [rid, "imaging_face", desc]
                if rdata:
                    row += [rdata["x"], rdata["y"], rdata["z"],
                            rdata["x_mm"], rdata["y_mm"], rdata["z_mm"],
                            rdata.get("num_orientation_clicks", "derived")]
                else:
                    row += ["","","","","","",""]
                row += ["","","","","","","","",""]   # no per-orient for derived
                writer.writerow(row)

        # ── Save XLSX for Excel / Google Sheets import ─────────────────────
        xlsx_path = self.output_dir / "microprism_corner_voxels.xlsx"
        xlsx_rows = [[
            "row_type", "view", "point_number", "point_id",
            "x_voxel", "y_voxel", "z_voxel",
            "x_mm", "y_mm", "z_mm",
            "display_a", "display_b", "slice_index",
            "num_orientation_clicks",
        ]]
        for row in clicked_points_list:
            x = row["x_voxel"]
            y = row["y_voxel"]
            z = row["z_voxel"]
            x_mm, y_mm, z_mm = self._xyz_to_mm(np.array([x, y, z], dtype=float))
            xlsx_rows.append([
                "raw_click",
                row["view"],
                row["point_number"],
                row["point_id"],
                x, y, z,
                round(float(x_mm), 5),
                round(float(y_mm), 5),
                round(float(z_mm), 5),
                row["display_a"],
                row["display_b"],
                row["slice_index"],
                "",
            ])
        for p, pdef in enumerate(POINT_DEFS):
            point = green.get(pdef["id"])
            if point:
                xlsx_rows.append([
                    "average",
                    "average_all_views",
                    p + 1,
                    pdef["id"],
                    point["x"], point["y"], point["z"],
                    point["x_mm"], point["y_mm"], point["z_mm"],
                    "", "", "",
                    point["num_orientation_clicks"],
                ])
        for point_id in ("derived_left", "derived_right"):
            point = imaging_face.get(point_id)
            if point:
                xlsx_rows.append([
                    "derived",
                    "derived", "",
                    point_id,
                    point["x"], point["y"], point["z"],
                    point["x_mm"], point["y_mm"], point["z_mm"],
                    "", "", "", "derived",
                ])
        self._write_simple_xlsx(xlsx_path, xlsx_rows)

        # ── Print summary ─────────────────────────────────────────────────
        print("\n" + "="*65)
        print("SAVED")
        print(f"  JSON: {json_path}")
        print(f"  CSV:  {csv_path}")
        print(f"  XLSX: {xlsx_path}")
        print("="*65)
        print(f"{'Point':<10} {'X':>7} {'Y':>7} {'Z':>7}  "
              f"{'X_mm':>8} {'Y_mm':>8} {'Z_mm':>8}")
        print("-"*65)
        for pid, v in green.items():
            if v:
                print(f"{pid:<10} {v['x']:>7.1f} {v['y']:>7.1f} "
                      f"{v['z']:>7.1f}  {v['x_mm']:>8.3f} "
                      f"{v['y_mm']:>8.3f} {v['z_mm']:>8.3f}")
        for label in ("derived_left", "derived_right"):
            v = imaging_face.get(label)
            if v:
                print(f"{label:<13} {v['x']:>7.1f} {v['y']:>7.1f} "
                      f"{v['z']:>7.1f}  {v['x_mm']:>8.3f} "
                      f"{v['y_mm']:>8.3f} {v['z_mm']:>8.3f}  ← derived")
        print("="*65)

    def _print_instructions(self) -> None:
        print("\n" + "="*72)
        print("INSTRUCTIONS")
        print("="*72)
        print("  [Coronal] [Sagittal] [Axial] buttons (or C/S/A keys) switch views.")
        print("  Phase 1 — Coronal:  click P1 → P2 → P3 → P4")
        print("  Phase 2 — Sagittal: click P1 → P2 → P3 → P4")
        print("  Phase 3 — Axial:    click P1 → P2 → P3 → P4")
        print()
        print("  Check the coordinate table as you go: P1-P4 should be relatively")
        print("  consistent across Coronal, Sagittal, and Axial for the same corner.")
        print()
        print("  Slider / scroll wheel / ←→ arrows — navigate slices")
        print("  Smaller side panels show the other two views at the same relative slice depth")
        print("  Zoom buttons or +/- keys — zoom in/out for pixel-level clicking")
        print("  Pan button or P — when zoomed, click-drag the image instead of placing a point")
        print("  Show overlays / O — project prior-view clicks into the current view")
        print("  Yellow one-pixel square = exact saved pixel center")
        print("  U — undo last point    Enter — save & quit")
        print()
        print("  Output: microprism_corners.json + microprism_corners.csv")
        print("="*72 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    """Parse command-line options for project path, image override, and output."""
    p = argparse.ArgumentParser(
        description="Mark microprism mirror-face corners in a microCT TIFF stack.")
    p.add_argument("project_path", nargs="?")
    p.add_argument("--image",       default=None,
                   help="Explicit TIFF path. Overrides project_path search.")
    p.add_argument("--output",      default=None,
                   help="Output directory. Default: project_path/outputs.")
    p.add_argument("--spacing",     type=float, default=None,
                   help="Isotropic voxel spacing in mm, e.g. 0.072. "
                        "Overrides metadata detection. If omitted and metadata "
                        "is missing, you will be prompted.")
    p.add_argument("--spacing-x",   type=float, default=None,
                   help="Voxel size along X in mm. Overrides --spacing for X.")
    p.add_argument("--spacing-y",   type=float, default=None,
                   help="Voxel size along Y in mm. Overrides --spacing for Y.")
    p.add_argument("--spacing-z",   type=float, default=None,
                   help="Voxel size along Z in mm. Overrides --spacing for Z.")
    p.add_argument("--prism-width", type=float, default=None,
                   help="Microprism width W in mm. If omitted, you will be prompted.")
    p.add_argument("--prism-length", type=float, default=None,
                   help="Microprism length l in mm. If omitted, you will be prompted.")
    p.add_argument("--no-maximize", action="store_true")
    return p.parse_args()


def main() -> None:
    """Choose the input TIFF, create the tracker object, and open the UI."""
    args         = parse_args()
    project_path = resolve_project_path(args.project_path)
    image_path   = find_best_microct_image(
        project_path, Path(args.image) if args.image else None)
    output_dir   = (Path(args.output) if args.output
                    else project_path.expanduser().resolve() / "outputs")

    if "corrected" in image_path.name.lower():
        print(f"✓ Using axis-corrected image: {image_path.name}")
    else:
        print(f"✓ Image: {image_path.name}")
        print("  Note: pass --image to use a specific corrected TIFF.")

    print("\nEnter microprism dimensions for imaging-face calculations.")
    prism_width_mm = (args.prism_width if args.prism_width is not None
                      else prompt_float("  Width W", DEFAULT_PRISM_WIDTH_MM))
    prism_length_mm = (args.prism_length if args.prism_length is not None
                       else prompt_float("  Length l", DEFAULT_PRISM_LENGTH_MM))
    spacing_xyz_mm = spacing_xyz_from_args(args, image_path)
    spacing_mm = float(np.mean(spacing_xyz_mm))

    MicroprismCornerTracker(
        microct_image_path=image_path,
        output_dir=output_dir,
        spacing_mm=spacing_mm,
        spacing_xyz_mm=spacing_xyz_mm,
        prism_width_mm=prism_width_mm,
        prism_length_mm=prism_length_mm,
        maximize=not args.no_maximize,
    ).start()


if __name__ == "__main__":
    main()
