# Raw CT landmark coordinates

Midline alignment now writes `<processed TIFF>.alignment.json`. Its 4×4
`processed_to_raw_zyx` matrix maps homogeneous Z,Y,X voxel column vectors
from that exact processed image to the original raw grid. The inverse is
`raw_to_processed_zyx`. These are voxel mappings, not millimeter mappings.

The saved rotation is the actual SimpleITK output-to-input resampling
mapping. Orientation corrections happen before midline fitting. The final mapping is
`orientation_to_raw @ aligned_to_orientation`. A flip uses `N-1-index`,
not just a sign change. Reset clears the axis corrections. The Rotate buttons
resample the volume and update its mapping; edges can be cropped. Zoom and pan
only change the display. Normal midline alignment uses `reverse=False`;
`reverse=True` deliberately applies the opposite rotation for diagnostics.

## Workflow

1. Preserve the TIFF on which the existing landmarks were selected.
2. Run `python src/midline_alignment.py /path/to/project`. Reuse saved
   midline points when prompted and reproduce the previous axis corrections.
   Existing output TIFFs, mapping records, and midline points are backed up
   with unique suffixes before replacement. Large TIFF backups require disk space.
3. Locate the new TIFF and its `.alignment.json` companion.
4. If selection did not use `--resample-moving`, convert with:

```sh
python src/alignment_coordinates.py /path/to/project/data/landmarks.json \
  /path/to/project/data/processed/microct_aligned_corrected.tif.alignment.json \
  --selection-image /path/to/preserved/old_selection.tif \
  --selection-grid-unchanged
```

The converter checks the selection TIFF's voxel digest against the new
mapping's TIFF digest. Matching voxel data, dtype, and shape are required;
paths and TIFF compression may differ. This is deliberately conservative:
a mismatch can be caused by intensity changes as well as geometry changes.
Do not substitute the new TIFF merely to bypass this check when points were
selected on an old TIFF. A filename alone does not establish correspondence.

The command backs up and updates landmarks.json, retaining moving_landmarks
and fixed_landmarks, adding moving_landmarks_raw, bounds flags and complete
mapping provenance. Use --output for a separate destination. Coordinates
retain subvoxel precision. No atlas registration is performed.

If the old and new volumes differ, the new mapping is still valid for the
new image, but cannot establish raw locations for old landmarks. Recover the
old mapping or establish a verified old-to-new mapping first. This utility
does not handle resampled landmark selection grids yet.

Check raw-space overlays before scientific use. Round-trip tests establish
algebraic consistency, not that the correct image/landmark pair was supplied.
Coordinates may be outside raw bounds; they are flagged rather than clamped.
