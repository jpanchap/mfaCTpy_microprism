"""Shared, discoverable navigation reference for prism tools."""
from pathlib import Path


def guide_location():
    return Path(__file__).resolve().parents[1] / 'docs' / 'prism_gui_guide.md'


TRACKER_HELP = """
CORNER TRACKER — HELP
1. Choose P1–P4 in Coronal, Sagittal, then Axial (12 clicks).
   The face appears only after all three views are complete, using their averages.
2. Method switches Fixed dimensions / Original triangles. Preview switches A/B.
   Inspect the microCT, then Use orientation confirms the fixed candidate.
3. Save & Quit writes both methods and the selected face. Quit saves nothing.

NAVIGATION
Coronal/C, Sagittal/S, Axial/A: select view; click a small preview to activate it.
Slider, scroll, arrow keys, or ← Slice / Slice →: move one slice.
Shift+arrows: move 10 slices. Keyboard focus must be in the figure window.
The small previews track relative slice depth, not one shared anatomical point.
Zoom +/-: magnify; Ctrl/Cmd+scroll: zoom around pointer; Fit image/0: full image.
Pan/P: drag the image without placing clicks; Undo/U: remove the last click.
Labels/L: hide/show scan P1–P4 and D1/D2 text, retaining markers and outlines.
Show overlays/O: show clicks from other views (faint when off the current slice).
Help/?/H/F1: print this help and the complete guide path in the terminal.

VISUALS
Fixed face = cyan; Original = orange. Dashed outline = entire projected face,
including off-slice corners. Thick solid line = fitted face/current-slice
intersection, NOT another prism edge. It moves as you scroll.
Thin connecting lines = raw-to-fitted corner adjustment; white x = raw average;
colored + = fitted corner. Yellow one-voxel box = exact clicked voxel; circles
help locate the marker and do not indicate physical uncertainty.
P1/P2 fit are adjusted bottom corners; D1/D2 are derived top corners.
Coordinate table = raw observations; guide diagram = schematic, not measured fit.
Status shows click completion, assumed dimensions, branch, confirmation and errors.
P1–P4 errors and RMS are in mm; zero fit/plane error does not prove atlas accuracy.
Black padding keeps viewing panels fixed and preserves physical image proportions.

MODEL
Fixed fitting assumes prism depth equals imaging-face length (equal-leg right prism).
Any changed click invalidates orientation confirmation. Choose A/B from the image,
not from which candidate looks closer to a scan axis. Dimensions/spacing must be
verified for the image you are actually using.
"""

COMPARISON_HELP = """
ATLAS METHOD COMPARISON — HELP
Left: fixed-dimension face. Right: original triangle face.
Coronal/Sagittal/Axial share one slice slider and limits. Arrows change slices;
Shift+arrows change 10. In-plane uses each method's own sampling surface.
Labels/L toggles P1/P2/D1/D2 annotations. Help/? prints this reference.
Dashed = full projected face; solid = slice intersection for a planar face.
In-plane colors are atlas regions; the outline/mask marks the sampled area.
Region percentages always describe each oblique face, not the background slice.
Source RMS is distance from fitted to clicked corners, in mm.
Prefer fixed/original saves comparison.json and preferred_corners.json, keeping both
alternatives. Closing without choosing does not select a preferred result.
"""
