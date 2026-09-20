"""
microprism_corner_tracker.py  (v7 - Clean UI Edition, performance fixed)
─────────────────────────────────────────────────────────────────────────────
Fixes from the original v7 clean UI:
  • Slider feedback loop removed (_updating_slider guard added)
  • slider.valmax update now also sets ax.set_xlim() to avoid layout thrash
  • draw_idle() called once at the end of a combined _refresh() instead of
    twice (once per panel) — halves render work per interaction
  • _on_key arrow keys added (were missing from original v7)
  • _save_results fully implemented (was `pass` placeholder)
  • CSV: 12 raw rows grouped by view, then 4 averaged summary rows
  • Red derived points removed entirely

Coordinate convention:  numpy volume shape = (Z, Y, X)
View convention (matches midline_alignment.py):
  coronal  = slice through Z, displayed as (Y, X)
  sagittal = slice through X, displayed as (Z, Y)
  axial    = slice through Y, displayed as (Z, X)

Workflow:
  Phase 1: Coronal  view → click P1 → P2 → P3 → P4
  Phase 2: Sagittal view → click P1 → P2 → P3 → P4
  Phase 3: Axial    view → click P1 → P2 → P3 → P4

Output:
  outputs/microprism_corners.json
  outputs/microprism_corners.csv

Dependencies: pip install numpy tifffile matplotlib
"""

from __future__ import annotations

import argparse
from project_paths import resolve_project_path
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.patches import Polygon as MplPoly
from matplotlib.widgets import Button, Slider

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
PRISM_EDGE_MM   = 0.5      # TODO: update once confirmed by postdoc
DEFAULT_SPACING = 0.025    # mm/voxel, 25 µm

POINT_DEFS = [
    {"id": "P1", "label": "Point 1", "desc": "bottom-left mirror corner",  "color": "#2ecc71"},
    {"id": "P2", "label": "Point 2", "desc": "bottom-right mirror corner", "color": "#e67e22"},
    {"id": "P3", "label": "Point 3", "desc": "top-left mirror corner",     "color": "#3498db"},
    {"id": "P4", "label": "Point 4", "desc": "top-right mirror corner",    "color": "#e74c3c"},
]

ORIENT_KEYS   = ["coronal", "sagittal", "axial"]
ORIENT_NAMES  = {
    "coronal":  "Coronal (Z)",
    "sagittal": "Sagittal (X)",
    "axial":    "Axial (Y)",
}
ORIENT_COLORS = {
    "coronal":  "#3498db",
    "sagittal": "#9b59b6",
    "axial":    "#e67e22",
}

Click = Tuple[int, int, int]
XYZ   = Tuple[float, float, float]


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE SELECTION
# ─────────────────────────────────────────────────────────────────────────────
def find_best_microct_image(project_path: Path,
                            explicit_image: Optional[Path] = None) -> Path:
    if explicit_image is not None:
        p = explicit_image.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        if p.suffix.lower() not in {".tif", ".tiff"}:
            raise ValueError(f"Not a TIFF file: {p}")
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
        "Use --image /path/to/file.tif to override.")


# ─────────────────────────────────────────────────────────────────────────────
# TRACKER
# ─────────────────────────────────────────────────────────────────────────────
class MicroprismCornerTracker:

    def __init__(
        self,
        microct_image_path: str | Path,
        output_dir: str | Path | None = None,
        spacing_mm: float = DEFAULT_SPACING,
        maximize: bool = True,
    ) -> None:
        print("=" * 65)
        print("MICROPRISM CORNER TRACKER  v7")
        print("=" * 65)

        self.image_path = Path(microct_image_path).expanduser().resolve()
        self.spacing    = float(spacing_mm)
        self.maximize   = bool(maximize)

        print(f"\nLoading: {self.image_path.name}")
        self.image = tifffile.imread(str(self.image_path))
        if self.image.ndim == 4 and self.image.shape[0] == 1:
            self.image = self.image[0]
        if self.image.ndim != 3:
            raise ValueError(f"Expected 3D TIFF. Got: {self.image.shape}")

        self.nz, self.ny, self.nx = self.image.shape
        print(f"  Shape (Z, Y, X): {self.image.shape}")
        print(f"  Dtype: {self.image.dtype}")
        print(f"  Spacing: {self.spacing} mm ({self.spacing*1000:.0f} µm)")

        img_f = self.image.astype(np.float32, copy=False)
        p_low, p_high = np.percentile(img_f, [1, 99])
        if p_high <= p_low:
            p_low, p_high = float(img_f.min()), float(img_f.max())
        self.img_disp = np.clip((img_f - p_low) / (p_high - p_low + 1e-8), 0, 1)

        if output_dir is None:
            parent = self.image_path.parent
            if parent.name == "processed" and parent.parent.name == "data":
                output_dir = parent.parent.parent / "outputs"
            elif parent.name == "data":
                output_dir = parent.parent / "outputs"
            else:
                output_dir = parent / "outputs"
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Output: {self.output_dir}")

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
        self._updating_slider = False   # ← guard prevents feedback loop

        # Matplotlib handles
        self.fig       = None
        self.ax_img    = None
        self.ax_guide  = None
        self.slider    = None
        self.info_text = None
        self.buttons: Dict[str, Button] = {}

    # ─────────────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._print_instructions()
        self._build_ui()
        if self.maximize:
            try:
                m = plt.get_current_fig_manager()
                if hasattr(m, "window"):
                    if hasattr(m.window, "showMaximized"):
                        m.window.showMaximized()
                    elif hasattr(m.window, "state"):
                        m.window.state("zoomed")
            except Exception:
                pass
        plt.show()

    # ─────────────────────────────────────────────────────────────────────
    # UI BUILD  (v7 clean UI layout, unchanged from original)
    # ─────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.fig = plt.figure(figsize=(18, 10), facecolor="#fdfdfd")

        # Main image — large left panel
        self.ax_img = self.fig.add_axes([0.05, 0.18, 0.55, 0.75])
        self.ax_img.set_facecolor("black")

        # Guide — right panel
        self.ax_guide = self.fig.add_axes([0.65, 0.25, 0.30, 0.65])
        self.ax_guide.set_facecolor("#f9f9f9")

        self._setup_buttons()

        # Slider — *** do NOT call set_val inside _update_display;
        # use _set_slider_val() which guards against re-entry ***
        ax_slider = self.fig.add_axes([0.05, 0.12, 0.55, 0.02])
        self.slider = Slider(ax_slider, "", 0,
                             self._max_slice("coronal"),
                             valinit=self.current_slice,
                             valstep=1, color="#3498db")
        self.slider.on_changed(self._on_slider)

        # Status text (right panel, below guide)
        self.info_text = self.fig.text(
            0.65, 0.18, "", fontsize=10, va="top", family="monospace")

        # Keyboard hint
        self.fig.text(
            0.65, 0.05,
            "KEYS: [C] Coronal  [S] Sagittal  [A] Axial\n"
            "[U] Undo  [Enter] Save  |  SCROLL or ARROWS to change slices",
            fontsize=9, color="#666", va="bottom")

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("scroll_event",       self._on_scroll)
        self.fig.canvas.mpl_connect("key_press_event",    self._on_key)

        self._refresh()   # single draw call to populate both panels

    def _setup_buttons(self) -> None:
        self.buttons = {}
        y1, y2     = 0.06, 0.015
        w, h, gap  = 0.08, 0.035, 0.01

        views = [
            ("coronal",  "Coronal",   0.05),
            ("sagittal", "Sagittal",  0.05 + w + gap),
            ("axial",    "Axial",     0.05 + 2*(w+gap)),
        ]
        for key, lbl, x in views:
            btn = Button(self.fig.add_axes([x, y1, w, h]),
                         lbl, color="#eee", hovercolor="#ddd")
            btn.on_clicked(lambda e, v=key: self._switch_view(v))
            self.buttons[key] = btn

        actions = [
            ("undo", "Undo",       0.05,             "#d1ecf1"),
            ("save", "Save & Quit",0.05 + w + gap,   "#d1ecf1"),
            ("quit", "Cancel",     0.05 + 2*(w+gap), "#f8d7da"),
        ]
        for key, lbl, x, col in actions:
            btn = Button(self.fig.add_axes([x, y2, w, h]),
                         lbl, color=col)
            if key == "undo":
                btn.on_clicked(self._on_undo)
            elif key == "save":
                btn.on_clicked(self._on_save)
            else:
                btn.on_clicked(lambda e: plt.close(self.fig))
            self.buttons[key] = btn

    # ─────────────────────────────────────────────────────────────────────
    # REFRESH  (single call updates both panels + does ONE draw_idle)
    # ─────────────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        """Update image panel and guide panel, then draw once."""
        self._draw_image()
        self._draw_guide()
        self.fig.canvas.draw_idle()   # ← exactly one render call

    # ─────────────────────────────────────────────────────────────────────
    # IMAGE PANEL
    # ─────────────────────────────────────────────────────────────────────

    def _draw_image(self) -> None:
        self.ax_img.clear()
        self.ax_img.imshow(
            self._get_slice_2d(self.current_view, self.current_slice),
            cmap="gray", vmin=0, vmax=1,
            interpolation="nearest", aspect="equal")
        self.ax_img.set_title(
            f"{ORIENT_NAMES[self.current_view]}  ·  "
            f"slice {self.current_slice}/{self._max_slice(self.current_view)}",
            color=ORIENT_COLORS[self.current_view], pad=10, fontsize=12)
        self.ax_img.axis("off")

        # Markers for this view
        o = ORIENT_KEYS.index(self.current_view)
        for i, click in enumerate(self.clicks[o]):
            if click is None:
                continue
            a, b, s = click
            col   = POINT_DEFS[i]["color"]
            alpha = 1.0 if s == self.current_slice else 0.28
            self.ax_img.plot(a, b, "x", color=col, ms=12,
                             mew=2.5, alpha=alpha, zorder=5)
            self.ax_img.add_patch(plt.Circle(
                (a, b), 7, fill=False, edgecolor=col,
                linewidth=1.8, alpha=alpha, zorder=5))
            self.ax_img.text(
                a+9, b-9, f"{POINT_DEFS[i]['id']} sl={s}",
                color=col, fontsize=8, fontweight="bold",
                alpha=alpha, zorder=6,
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="black", edgecolor=col, alpha=0.55))

        # Progress overlay
        for i, key in enumerate(ORIENT_KEYS):
            done   = all(c is not None for c in self.clicks[i])
            active = (i == self.current_orient and self.current_orient < 3)
            mark   = "✓" if done else ("→" if active else "·")
            alpha  = 1.0 if (done or active) else 0.35
            self.ax_img.text(
                10, 18+22*i,
                f"{mark} {ORIENT_NAMES[key]}",
                color=ORIENT_COLORS[key], fontsize=9,
                fontweight="bold", alpha=alpha, zorder=7,
                bbox=dict(boxstyle="round,pad=0.18",
                          facecolor="black", alpha=0.5))

        # Update slider range without triggering _on_slider
        max_sl = self._max_slice(self.current_view)
        if self.slider is not None:
            self.slider.valmin = 0
            self.slider.valmax = max_sl
            self.slider.ax.set_xlim(0, max_sl)   # ← prevents layout thrash
            self.slider.label.set_text(
                f"{self.current_view.capitalize()} slice")
            self._set_slider_val(self.current_slice)

    # ─────────────────────────────────────────────────────────────────────
    # GUIDE PANEL  (v7 original drawing, unchanged)
    # ─────────────────────────────────────────────────────────────────────

    def _draw_guide(self) -> None:
        ax = self.ax_guide
        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 400)
        ax.set_ylim(450, 0)

        # ── Prism geometry ────────────────────────────────────────────────
        iso = np.array([50, -30])
        FL  = np.array([120, 100]);  FL_ = FL + iso
        FR  = np.array([240, 100]);  FR_ = FR + iso
        MR  = np.array([240, 320]);  MR_ = MR + iso
        BL  = np.array([120, 400]);  BL_ = BL + iso

        def draw_poly(pts, color, alpha=0.3, lw=1):
            ax.add_patch(MplPoly(pts, closed=True,
                                 facecolor=color, edgecolor="#222",
                                 linewidth=lw, alpha=alpha, zorder=2))

        def draw_line(a, b, col, lw=1.3, ls="-", alpha=1.0):
            ax.plot([a[0],b[0]], [a[1],b[1]],
                    color=col, linewidth=lw, linestyle=ls,
                    alpha=alpha, zorder=3, solid_capstyle="round")

        # Faces — back to front
        draw_poly([FL,  FR,  FR_, FL_], "#bdc3c7", alpha=0.35)   # top
        draw_poly([FR,  MR,  MR_, FR_], "#95a5a6", alpha=0.28)   # right
        draw_poly([BL,  MR,  MR_, BL_], "#3498db", alpha=0.55, lw=2)  # mirror
        draw_poly([FL,  FR,  MR,  BL],  "#ecf0f1", alpha=0.45)   # front

        # Edges
        for a, b in [(FL,FR),(FR,FR_),(FR_,FL_),(FL_,FL)]:
            draw_line(a, b, "#2c3e50", lw=1.4)
        draw_line(FL,  BL,  "#2c3e50", lw=1.4)
        draw_line(FR,  MR,  "#2c3e50", lw=1.4)
        draw_line(MR,  BL,  "#1a5a8a", lw=2.2)     # front 45° cut
        draw_line(FR_, MR_, "#2c3e50", lw=1.1)
        draw_line(MR_, BL_, "#1a5a8a", lw=1.8)     # back 45° cut
        draw_line(BL,  BL_, "#1a5a8a", lw=2.0)     # mirror base
        draw_line(MR,  MR_, "#1a5a8a", lw=2.0)     # mirror top
        draw_line(FL_, BL_, "#7aaabb", lw=0.9, ls="--", alpha=0.5)  # hidden

        # Center dashes
        draw_line((FL+FR)/2, (BL+MR)/2,  "#8aaabb", lw=0.8, ls="--", alpha=0.45)
        draw_line((FL_+FR_)/2,(BL_+MR_)/2,"#8aaabb",lw=0.6, ls="--", alpha=0.22)

        # 45° label
        ax.add_patch(mpatches.Arc(BL, 36, 36, angle=0,
                                  theta1=308, theta2=360,
                                  color="#1a5a8a", lw=1.0, zorder=4))
        ax.text(BL[0]+24, BL[1]+5, "45°",
                fontsize=8, color="#1a5a8a", fontstyle="italic",
                va="top", zorder=5)

        # Mirror face label
        mid_mf = (BL + MR) / 2
        ax.annotate("mirror face",
                    xy=mid_mf + np.array([15, 0]),
                    xytext=(mid_mf[0]+60, mid_mf[1]),
                    fontsize=8, color="#1a5a8a", va="center",
                    arrowprops=dict(arrowstyle="-", color="#1a5a8a", lw=0.8),
                    zorder=5)

        # ── 4 point markers ───────────────────────────────────────────────
        pts_map = {"P1": BL, "P2": BL_, "P3": MR, "P4": MR_}
        next_pid = (POINT_DEFS[self.current_point]["id"]
                    if self.current_orient < 3 else None)

        for i, pdef in enumerate(POINT_DEFS):
            pos    = pts_map[pdef["id"]]
            n_done = sum(1 for o in range(3) if self.clicks[o][i] is not None)
            is_next = (pdef["id"] == next_pid)

            col  = "gold" if is_next else pdef["color"]
            sz   = 12     if is_next else 8
            ax.plot(pos[0], pos[1], "o",
                    ms=sz, mfc=col, mec="k", mew=1.5, zorder=10)
            ax.text(pos[0]+13, pos[1],
                    f"{pdef['id']} ({n_done}/3)",
                    fontsize=9, fontweight="bold",
                    color="darkred" if is_next else pdef["color"],
                    va="center", zorder=11)

        # ── Status text (below guide axes) ────────────────────────────────
        if self.info_text is not None:
            if self.current_orient >= 3:
                phase = "✓ All points placed — save when ready"
                color = "#27ae60"
            else:
                o_name = ORIENT_NAMES[ORIENT_KEYS[self.current_orient]]
                p_name = POINT_DEFS[self.current_point]["id"]
                p_desc = POINT_DEFS[self.current_point]["desc"]
                phase  = f"Phase {self.current_orient+1}/3 — {o_name}\nNext: {p_name} ({p_desc})"
                color  = ORIENT_COLORS[ORIENT_KEYS[self.current_orient]]
            self.info_text.set_text(phase)
            self.info_text.set_color(color)

    # ─────────────────────────────────────────────────────────────────────
    # VIEW SWITCHING
    # ─────────────────────────────────────────────────────────────────────

    def _switch_view(self, view: str) -> None:
        if view not in ORIENT_KEYS:
            return
        self.slice_indices[self.current_view] = self.current_slice
        self.current_view  = view
        self.current_slice = self.slice_indices[view]
        print(f"\n  Switched to {ORIENT_NAMES[view]}")
        if self.fig is not None:
            self._refresh()

    # ─────────────────────────────────────────────────────────────────────
    # EVENTS
    # ─────────────────────────────────────────────────────────────────────

    def _on_click(self, event) -> None:
        if event.inaxes is not self.ax_img:
            return
        if event.button != 1 or event.xdata is None:
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
        o    = ORIENT_KEYS.index(self.current_view)

        self.clicks[o][self.current_point] = (a, b, int(self.current_slice))
        pdef = POINT_DEFS[self.current_point]
        print(f"  [{ORIENT_NAMES[self.current_view]}] "
              f"{pdef['id']} → ({a},{b}) slice={self.current_slice}")

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
        self._refresh()

    def _on_slider(self, val) -> None:
        # *** Guard prevents set_val() from re-entering here ***
        if self._updating_slider:
            return
        self.current_slice = int(round(val))
        self.slice_indices[self.current_view] = self.current_slice
        self._draw_image()
        self.fig.canvas.draw_idle()

    def _on_scroll(self, event) -> None:
        if event.inaxes is not self.ax_img:
            return
        step = 1 if event.button == "up" else -1
        self.current_slice = int(np.clip(
            self.current_slice + step,
            0, self._max_slice(self.current_view)))
        self.slice_indices[self.current_view] = self.current_slice
        self._set_slider_val(self.current_slice)
        self._draw_image()
        self.fig.canvas.draw_idle()

    def _on_key(self, event) -> None:
        k = (event.key or "").lower()
        if k == "c":
            self._switch_view("coronal")
        elif k == "s":
            self._switch_view("sagittal")
        elif k == "a":
            self._switch_view("axial")
        elif k in ("right", "up"):
            self.current_slice = min(
                self.current_slice + 1,
                self._max_slice(self.current_view))
            self.slice_indices[self.current_view] = self.current_slice
            self._set_slider_val(self.current_slice)
            self._draw_image()
            self.fig.canvas.draw_idle()
        elif k in ("left", "down"):
            self.current_slice = max(self.current_slice - 1, 0)
            self.slice_indices[self.current_view] = self.current_slice
            self._set_slider_val(self.current_slice)
            self._draw_image()
            self.fig.canvas.draw_idle()
        elif k == "u":
            self._on_undo(None)
        elif k == "enter":
            self._on_save(None)

    def _on_undo(self, _e) -> None:
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
        placed = sum(c is not None for row in self.clicks for c in row)
        if placed < 12 and not self._save_warned:
            print(f"  WARNING: {placed}/12 clicks placed. "
                  "Click Save & Quit again to confirm partial save.")
            self._save_warned = True
            return
        self._save_results()
        plt.close(self.fig)

    # ─────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────

    def _set_slider_val(self, v: int) -> None:
        """Set slider value without triggering _on_slider."""
        if self.slider is None:
            return
        if int(round(self.slider.val)) == v:
            return
        self._updating_slider = True
        try:
            self.slider.set_val(v)
        finally:
            self._updating_slider = False

    def _max_slice(self, view: str) -> int:
        return {"coronal": self.nz-1, "sagittal": self.nx-1,
                "axial": self.ny-1}[view]

    def _get_slice_2d(self, view: str, idx: int) -> np.ndarray:
        if view == "coronal":   return self.img_disp[idx, :, :]
        if view == "sagittal":  return self.img_disp[:, :, idx]
        if view == "axial":     return self.img_disp[:, idx, :]
        raise ValueError(view)

    def _clamp(self, view: str, a: int, b: int) -> Tuple[int, int]:
        if view == "coronal":
            return int(np.clip(a,0,self.nx-1)), int(np.clip(b,0,self.ny-1))
        if view == "sagittal":
            return int(np.clip(a,0,self.ny-1)), int(np.clip(b,0,self.nz-1))
        if view == "axial":
            return int(np.clip(a,0,self.nx-1)), int(np.clip(b,0,self.nz-1))
        raise ValueError(view)

    def _click_to_xyz(self, o: int, click: Click) -> XYZ:
        a, b, s = click
        if o == 0: return float(a), float(b), float(s)   # coronal
        if o == 1: return float(s), float(a), float(b)   # sagittal
        if o == 2: return float(a), float(s), float(b)   # axial
        raise ValueError(o)

    def _average_point(self, point_idx: int) -> Optional[Dict]:
        coords = [self._click_to_xyz(o, self.clicks[o][point_idx])
                  for o in range(3) if self.clicks[o][point_idx] is not None]
        if not coords:
            return None
        x, y, z = np.array(coords, dtype=float).mean(axis=0)
        return {
            "x": round(x,3), "y": round(y,3), "z": round(z,3),
            "x_mm": round(x*self.spacing,5),
            "y_mm": round(y*self.spacing,5),
            "z_mm": round(z*self.spacing,5),
            "num_orientation_clicks": len(coords),
        }

    # ─────────────────────────────────────────────────────────────────────
    # SAVE — JSON + CSV
    # CSV: 12 raw rows grouped by view, blank row, 4 averaged summary rows
    # ─────────────────────────────────────────────────────────────────────

    def _save_results(self) -> None:
        # Build per-orientation data
        per_orientation: Dict = {}
        for o_idx, view in enumerate(ORIENT_KEYS):
            per_orientation[view] = {}
            for p_idx, pdef in enumerate(POINT_DEFS):
                click = self.clicks[o_idx][p_idx]
                if click is None:
                    continue
                x, y, z = self._click_to_xyz(o_idx, click)
                per_orientation[view][pdef["id"]] = {
                    "x": x, "y": y, "z": z,
                    "x_mm": round(x*self.spacing, 5),
                    "y_mm": round(y*self.spacing, 5),
                    "z_mm": round(z*self.spacing, 5),
                    "display_a":   click[0],
                    "display_b":   click[1],
                    "slice_index": click[2],
                }

        averaged = {pdef["id"]: self._average_point(i)
                    for i, pdef in enumerate(POINT_DEFS)}

        # JSON
        json_output = {
            "source_image_path": str(self.image_path),
            "source_image_name": self.image_path.name,
            "source_image_looks_axis_corrected":
                "corrected" in self.image_path.name.lower(),
            "image_shape_zyx": [int(self.nz), int(self.ny), int(self.nx)],
            "spacing_mm":    self.spacing,
            "prism_edge_mm": PRISM_EDGE_MM,
            "coordinate_convention":
                "volume shape is (Z,Y,X); x,y,z are voxel coordinates",
            "view_convention": {
                "coronal":  "slice through Z; display rows=Y, cols=X",
                "sagittal": "slice through X; display rows=Z, cols=Y",
                "axial":    "slice through Y; display rows=Z, cols=X",
            },
            "averaged_points":  averaged,
            "per_orientation":  per_orientation,
        }
        json_path = self.output_dir / "microprism_corners.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=2)

        # CSV
        header = [
            "section", "view", "point_id", "description",
            "x_voxel", "y_voxel", "z_voxel",
            "x_mm",    "y_mm",    "z_mm",
            "display_a", "display_b", "slice_index",
        ]
        csv_path = self.output_dir / "microprism_corners.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            # Section 1: raw rows grouped by view then point
            for view in ORIENT_KEYS:
                for pdef in POINT_DEFS:
                    raw = per_orientation.get(view, {}).get(pdef["id"])
                    if raw:
                        writer.writerow([
                            "raw", view, pdef["id"], pdef["desc"],
                            raw["x"], raw["y"], raw["z"],
                            raw["x_mm"], raw["y_mm"], raw["z_mm"],
                            raw["display_a"], raw["display_b"],
                            raw["slice_index"],
                        ])
                    else:
                        writer.writerow([
                            "raw", view, pdef["id"], pdef["desc"],
                            "","","","","","","","","",
                        ])
            # Blank separator
            writer.writerow([])
            # Section 2: averaged summary
            for pdef in POINT_DEFS:
                v = averaged.get(pdef["id"])
                if v:
                    writer.writerow([
                        "averaged", "all_views", pdef["id"], pdef["desc"],
                        v["x"], v["y"], v["z"],
                        v["x_mm"], v["y_mm"], v["z_mm"],
                        "", "", v["num_orientation_clicks"],
                    ])
                else:
                    writer.writerow([
                        "averaged", "all_views", pdef["id"], pdef["desc"],
                        "","","","","","","","","",
                    ])

        # Console summary
        print("\n" + "="*65)
        print("SAVED")
        print(f"  JSON: {json_path}")
        print(f"  CSV:  {csv_path}")
        print("\nAveraged points:")
        for pid, val in averaged.items():
            if val:
                print(f"  {pid}: x={val['x']}, y={val['y']}, z={val['z']}  "
                      f"({val['num_orientation_clicks']}/3 views)")
            else:
                print(f"  {pid}: not placed")
        print("="*65)

    def _print_instructions(self) -> None:
        print("\nInstructions:")
        print("  Use [C] [S] [A] buttons or keys to switch views.")
        print("  Phase 1 — Coronal:  click P1 → P2 → P3 → P4")
        print("  Phase 2 — Sagittal: click P1 → P2 → P3 → P4")
        print("  Phase 3 — Axial:    click P1 → P2 → P3 → P4")
        print("  Slider / scroll / arrows navigate slices.")
        print("  [U] undo  [Enter] save\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mark microprism mirror-face corners in a microCT TIFF stack.")
    parser.add_argument(
        "project_path", nargs="?",
        help="Project folder containing data/ and outputs/.")
    parser.add_argument("--image",       default=None)
    parser.add_argument("--output",      default=None)
    parser.add_argument("--spacing",     type=float, default=DEFAULT_SPACING)
    parser.add_argument("--no-maximize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args         = parse_args()
    project_path = resolve_project_path(args.project_path)
    image_path   = find_best_microct_image(
        project_path, Path(args.image) if args.image else None)
    output_dir   = (Path(args.output) if args.output
                    else project_path.expanduser().resolve() / "outputs")

    print(f"Selected image: {image_path}")
    if "corrected" in image_path.name.lower():
        print("✓ Using an axis-corrected TIFF.")
    else:
        print("Note: pass --image to use a specific corrected TIFF.")

    MicroprismCornerTracker(
        microct_image_path=image_path,
        output_dir=output_dir,
        spacing_mm=args.spacing,
        maximize=not args.no_maximize,
    ).start()


if __name__ == "__main__":
    main()
