# Fixed-dimension prism fitting and atlas comparison

The corner tracker now defaults to a rigid, fixed-dimension fit. The original
independent right-triangle derivation is retained unchanged as **Original**.
Registration is unchanged.

## Physical model

The entered width W is the imaging-face width. The entered length L is both its
height and the perpendicular prism depth: this is an **equal-leg right-prism
assumption**, not a consequence of squareness alone. Confirm that the physical
P1/P3 and P2/P4 corners have this relationship. W=L gives a square; otherwise
the imaging face is rectangular.

P1/P2/P3/P4 form a model rectangle of dimensions W by sqrt(2)*L. SVD fits its
rotation and translation jointly to the full-precision mean clicked locations
in physical millimeters. There is no scaling or reflection. Each physical
corner has equal weight. Repeated per-view observations remain saved.

The two D1 candidates are the fitted P1/P3 midpoint plus or minus L/sqrt(2)
times the fitted hypotenuse-face normal. D2 uses the P2/P4 midpoint and the
same sign. D1 is **off the diagonal**, not on it. Both candidates fit P1–P4
equally well; an explicit visual orientation selection is required.

See the [complete GUI control and visual reference](prism_gui_guide.md), also
printed by the tracker Help button and mapper ? button.

## Tracker workflow

Run the tracker as usual in the mfaCT environment. Enter the physical dimensions,
place P1–P4 in all three views (12 observations), then use the bottom-row controls.
The face preview waits for the three-view averages:

- **Method: Fixed / Original** switches previews.
- **Preview: A / B** switches the fixed-fit orientation.
- **Use orientation A / B** confirms it after inspection.
- **Save & Quit** saves both methods and the selected complete face.

Cyan is fixed; orange is original. White crosses mark raw averaged clicks;
fitted markers and connecting lines show adjustments. Dashed outlines are full
projections, even when corners are off-slice. Solid lines are the fixed face's
actual slice intersection. No solid intersection is claimed for the potentially
nonplanar original face. Read the P1–P4 error and RMS text above the slider.
Changing any click or branch invalidates the earlier orientation confirmation.

To reuse an existing run without re-clicking:

```sh
python src/microprism_corner_tracker.py /path/to/project \
  --load-corners /path/to/project/outputs/microprism_corners.json \
  --output /path/to/project/outputs/prism_fit
```

The saved image, dimensions and spacing are reused unless overridden explicitly.
Per-view clicks must be present. Orientation must be inspected and confirmed
again. Using a separate output directory preserves the prior export.

## Atlas comparison

```sh
python src/microprism_atlas_mapper.py /path/to/project \
  --corners /path/to/project/outputs/prism_fit/microprism_corners.json \
  --compare-methods
```

All existing mapper options remain available. Both runs share registration and
atlas settings. If transform direction was automatic, the first resolved direction
is reused for the second run. Each method independently builds its plane, samples
labels, and masks its own face. Results are stored under
`outputs/derivation_comparison/fixed/` and `original/` (or under `--output`).

The comparison opens after both maps finish. Coronal/sagittal/axial views share
a slice slider and bounds. In-plane views show separate sampling surfaces.
Top region percentages remain those of each oblique face, regardless of the
background view. The complete region tables are in each method's CSV/JSON.
The shared slice intersection is only drawn for planar faces.

**Prefer fixed / Prefer original** records the preference in `comparison.json`
and writes `preferred_corners.json`, preserving both alternatives. Original
tracker inputs are not changed. A snapshot and SHA-256 of the input are saved.
Later, open the chosen face in the full mapper (including its existing 2p tools):

```sh
python src/microprism_atlas_mapper.py /path/to/project \
  --corners /path/to/project/outputs/derivation_comparison/preferred_corners.json
```

`--method fixed` or `--method original` selects an individual saved method.
`--fixed-branch A` (or B) is an explicit orientation override. `--no-show` with
comparison produces both output sets without a preference UI. Re-running the
comparison regenerates its outputs and clears its preference; retain another
output directory if comparing different sessions.

## Interpretation and compatibility

- Fitted P1/P2 can move; do not combine raw P1/P2 with fitted D1/D2.
- The JSON schema is version 2. The legacy `imaging_face_points` field holds the
  selected complete face. Raw clicks and all alternatives are stored separately.
- Legacy single-method files still open in the mapper. Reopen them in the tracker
  to generate a fixed fit before using comparison mode.
- Width/height/coplanarity are guaranteed in source physical space only. Affine
  registration can distort the atlas footprint; the mapper does not square that
  footprint again for region coverage.
- Small fit errors do not prove anatomical accuracy. Inspect the microCT and
  validate registration independently. The original method is a baseline, not
  ground truth. There is no automatic microCT boundary segmentation or automatic
  candidate selection in this implementation.

## Validation

```sh
MPLBACKEND=Agg python -m unittest discover -s tests -v
```

Tests cover arbitrary rotations/translations, exact dimensions and right angles,
unequal-depth helper geometry, noisy clicks, anisotropic source spacing,
degenerate inputs, slice intersections, original numeric regression,
orientation confirmation invalidation, save/reload and method selection,
independent sampling surfaces, comparison controls and preference export, and
a complete synthetic atlas mapping for both methods.
