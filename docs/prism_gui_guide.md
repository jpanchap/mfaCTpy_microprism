# Prism tools: controls, visuals, and troubleshooting

Both tools print this guide's location in the terminal. In the corner tracker,
**Help (?)**, `?`, `H`, or `F1` prints the navigation reference. In the full atlas
mapper, **?** prints its controls. The method-comparison window has its own
**Help (?)** button. Keep the launch terminal visible for errors and status messages.

## 1. Corner tracker

### Entering the data

- **Width W / Length l:** physical imaging-face dimensions in millimeters. Fixed
  fitting assumes the perpendicular prism depth also equals l. Equal W and l
  produce a square; unequal values produce a rectangle.
- **Voxel spacing:** physical voxel sizes, not the number of voxels or slices.
  `--spacing` and `--spacing-x/y/z` use **millimeters**. The interactive spacing
  prompt uses **micrometers**, and accepts one isotropic value or X,Y,Z values.
  There is no guessed default when metadata cannot be read.
- **P1/P2:** shared bottom imaging-face corners. **P3/P4:** corresponding outer
  prism corners. **D1/D2:** derived top imaging-face corners. Refer to the
  schematic for physical identities; its orientation is not the implant's orientation.
- Click P1 → P2 → P3 → P4 in **Coronal**, then **Sagittal**, then **Axial**.
  A click includes the selected slice as its third coordinate. Different corners
  can be on different slices.
- **Face preview:** waits for all 12 observations. It uses the full-precision
  average of each physical corner's three observations. Undoing a click hides
  the preview until the observations are complete again. The original method
  retains its original rounding and calculation for comparison.

### Panels, text, and symbols

| Item | Meaning |
|---|---|
| Large microCT panel | Active view; clicks are recorded here when its phase is active and Pan is off. |
| Two small microCT panels | Other orientations. Click a panel to activate it. Their slices track relative depth, not a common 3D crosshair. |
| Main title | View, current/max slice, completed clicks in that view, zoom, Pan state, and next action. |
| Voxel coordinate table | Raw selections from each view; disagreement can indicate click uncertainty. It is not a fitted-corner table. |
| Prism guide | Schematic and per-corner completion counts; not a measured 3D reconstruction. |
| Black margins | Padding to keep panel sizes stable while preserving physical image proportions. Clicking padding does not select a voxel. |
| Yellow one-voxel square | Exact selected voxel center. |
| Cyan cross/circle on a saved click | Locates a click on the current slice; circle radius is a visual aid, not uncertainty or prism size. |
| Temporary bright marker | Highlights the most recent click. |
| Other-view colored crosses/dashed circles | Clicks from another orientation when overlays are on; faded if off the current slice. Text identifies source view, point, and projected slice. |
| White x | Three-view averaged raw corner. |
| Colored + and thin connecting line | Fitted corner and displacement from the raw average. Both may be projected from another slice. |
| Cyan face | Fixed-dimension fit. |
| Orange face | Original independent-triangle derivation. |
| Dashed face boundary | The whole face projected into this viewing direction, including off-slice corners. |
| Thick solid line | Actual intersection of the fitted planar face with the current slice. It moves as you scroll and disappears when the slice misses the face. It is not a new edge or another fitted face. If the face lies in the slice, the intersection can be its whole outline. |
| P1 fit / P2 fit | Adjusted bottom corners in the fixed model. Original mode labels them P1/P2. |
| D1 / D2 | Derived top corners. Labels are projections and do not guarantee that the point lies in the current slice. |
| Fit status | Assumed dimensions, A/B orientation, whether confirmed, four point errors, and RMS error in mm. Small error does not establish anatomical accuracy. |

No solid intersection is claimed for the original face, which may be nonplanar.
The normalizing display contrast does not change stored image values or coordinates.

### Buttons and keys

| Button/key | Action |
|---|---|
| Coronal / C | Display X horizontally, Y vertically; slice through Z. |
| Sagittal / S | Display Y horizontally, Z vertically; slice through X. |
| Axial / A | Display X horizontally, Z vertically; slice through Y. |
| Slice slider | Select the active view's slice index. |
| ← Slice / Slice → | Move one slice backward/forward. |
| Left/Down; Right/Up | Move one slice backward/forward. Click the figure window first if focus is in the terminal. |
| Shift + arrow | Move ten slices, clamped at volume boundaries. |
| Mouse wheel over large panel | Move slices. |
| Ctrl/Cmd + mouse wheel | Zoom around the pointer. |
| Zoom + / + | Magnify the active image. |
| Zoom − / − | Zoom out. |
| Fit image / 0 | Reset zoom and show the full image with padding. |
| Pan off/on / P | Toggle dragging instead of placing points. |
| Undo / U | Remove the latest observation and return to its phase. |
| Show/Hide overlays / O | Show/hide clicks from other orientations. Does not hide the fitted face. |
| Labels on/off / L | Toggle scan point text, including fitted P1/P2 and D1/D2. Keeps markers, outlines, coordinate table, and schematic. |
| Method: Fixed/Original | Switch the face derivation being previewed and selected for export. |
| Preview: A/B | Switch fixed-fit orientation; both satisfy the model and fit P1–P4 equally well. |
| Use orientation A/B | Confirm the candidate after visual inspection (enabled only in Fixed mode). Editing clicks or changing the branch invalidates confirmation. |
| Save & Quit / Enter / Ctrl-or-Cmd+S | Save JSON/CSV/XLSX, both alternatives and selected complete face, then close. Both methods require all 12 clicks; fixed fitting also requires confirmed orientation. |
| Quit (no save) | Close without saving changes. |
| Help (?) / ? / H / F1 | Print explanations and this guide's path in the terminal. |

## 2. Full atlas mapper

This is the normal mapper window, distinct from the two-method comparison window.

### Main toolbar

| Button/key | Action |
|---|---|
| In-plane / I | View the oblique sampling plane along the mapped imaging face. |
| Coronal / C; Sagittal / S; Axial / A | View a standard atlas slice with the face projected into it. |
| − / +; mouse wheel | Zoom the current atlas view out/in. |
| Fit / 0 | Show the complete current slice. |
| R / Home | Reset the view. |
| Drag / arrow keys | Pan the atlas view. **These arrows pan; tracker and method-comparison arrows change slices.** |
| Colors / G | Toggle atlas-region colors. |
| Labels / L | Toggle corner names and region-ID labels. |
| Layers | Show/hide the contextual layer controls; finish a 2p selection/alignment first. |
| uCT / M | Open/close the source microCT comparison if the source image is available. |
| Map 2p | Start loading/mapping the unzoomed and zoomed two-photon images. |
| ? | Print controls and this guide's path in the terminal. |
| Panel / Panel + | Hide/show the region list. |
| Done / Q | Finish/close; summaries and saved paths are reported. |

### Layers and microCT controls

- **Fill / F:** translucent tint inside the prism outline; it is a visual layer,
  not a second ROI.
- **2p / T:** toggle the wide two-photon image overlay.
- **Zoom:** toggle the zoomed two-photon image overlay.
- **2p opacity / zoom alpha:** opacity of those image overlays, not geometry or region weights.
- **uCT + / uCT − / uCT 1x:** zoom/reset the source microCT panel independently.
- **uCT Pan off/on / P in microCT comparison:** choose dragging vs normal panel interaction.
- **Scroll over the microCT panel:** change its source slice.
- Blue highlighted controls generally indicate active views/toggles. Orange and
  green controls distinguish the wide and zoomed 2p workflows.

### Two-photon workflow controls

- During corner selection, click **D1, D2, P1, P2** in the prompted order.
- **Undo / Z / Backspace / Delete:** undo the last 2p corner selection.
- **Pan off/on / P:** move the 2p image without selecting a corner.
- During zoom-image alignment, **drag/arrows** adjust its offset; **Shift+arrows**
  move ten pixels. **Mag** changes magnification. **Check** evaluates alignment
  against the image data. **Re-centre / R / Home** resets the offset.
- **Confirm / Enter** accepts alignment; **Escape** cancels the active step.
- Alignment opacity controls visibility while matching, not the anatomical transform.

### Visuals and reported values

- **Atlas background:** anatomical template with optional region colors/boundaries.
- **Red outline/tint:** mapped imaging-face boundary/optional fill. A tilted face
  can appear compressed in a standard view. An affine mapping can change shape
  even in the In-plane view.
- **P1/P2/D1/D2 markers and leader labels:** mapped corner positions and atlas
  region IDs; standard views can show off-slice projected points.
- **Regions within prism ROI:** region names/acronyms, Allen IDs, colors and area
  fractions on the oblique imaging face. Changing the background view does not
  recalculate these as coronal/sagittal/axial slice fractions.
- **Source XYZ vox/mm:** original scan coordinates before brain-to-atlas mapping.
  **CCF XYZ vox:** transformed atlas coordinates.
- **Transformed geometry check:** lengths/angles/planarity of the actual mapped face.
- **Rectified display check:** an adjusted rectangular reference, not evidence
  that the measured ROI was replaced with that rectangle.
- **Fourth-corner plane error:** signed distance of D2 from the plane through
  P1/P2/D1; zero means coplanar, not necessarily square or anatomically accurate.
- **Saved paths:** files generated by the mapper, not quality scores.

## 3. Two-method atlas comparison

Launch with `--compare-methods` after saving both alternatives in the tracker.

- **Left:** fixed-dimension result (cyan). **Right:** original result (orange).
- **Coronal / Sagittal / Axial:** synchronized background slice and limits.
- **Slider/arrows:** one slice; **Shift+arrows:** ten. In-plane has no shared slice slider.
- **In-plane:** each method's own plane and region mask, with common atlas colors.
- **Labels / L:** hide/show P1/P2/D1/D2 text. **Help / ? / H / F1:** terminal guide.
- **Dashed boundary:** full projection. **Solid line:** slice intersection only
  when the face is planar. In-plane outlines/mask contours delimit sampled area.
- **Region summaries:** leading regions of each method's own face; full CSV/JSON
  tables contain all regions. **Source RMS:** fixed-fit mismatch to the clicks.
- **Prefer fixed / Prefer original:** save the preference and a complete
  `preferred_corners.json`, preserving both outputs and the original input.
  Close without choosing to leave preference unset. A preference is not independent
  validation of registration accuracy.

## 4. Permission errors and high-resolution images

`PermissionError: Operation not permitted` while opening
`microct_aligned_corrected.tif.metadata.json` means access to that file was denied.
It is not a slice-count or prism-dimension error. Possible underlying reasons
include application access restrictions or permissions on the mounted share;
the traceback alone does not distinguish them.

The tracker now warns and tries TIFF metadata. If no complete XYZ spacing can be
read, enter the **verified spacing of the selected processed image**, or provide
`--spacing MM` (isotropic) / all three `--spacing-x/y/z MM` options. Complete
explicit values bypass metadata reads. Do not copy a lower-resolution dataset's
spacing simply to get past the prompt. Changing the prism length does not fix
missing voxel calibration.

If the TIFF itself is inaccessible, the program still needs file access: verify
that the application launching Python can read it, check the mounted share's
access, or work from an accessible local copy with its metadata. `--spacing`
bypasses the sidecar, not permission checks on the image or output directory.
Use `--output` for an accessible save destination if the source share is read-only.

Higher resolution can separately increase loading time and memory requirements.
The tracker memory-maps compatible TIFFs and normalizes only displayed slices;
compressed/non-mappable TIFFs still need to be decoded. Panel sizes are fixed,
with padding instead of changing their size or stretching the anatomy.

For the fitting model, saved-file format and example commands, see
[Fixed-dimension workflow](prism_dimension_fit.md).
