# mfaCTpy: MicroCT Mouse Brain Registration to Allen CCF

> **Fork notice.** This is a fork of
> [Sakata-Lab/mfaCTpy](https://github.com/Sakata-Lab/mfaCTpy), adapted for
> microprism imaging. Work is ongoing to locate a microprism's imaging face
> in the Allen CCF and map two-photon images onto it, in
> `microprism_corner_tracker.py` and `microprism_atlas_mapper.py`. The rest
> of the pipeline is upstream's and unchanged in purpose. Licensed GPL v3,
> as the original is.

**New version available!** https://github.com/Sakata-Lab/mfaCTpy2

## User Guide v1.1

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Data Structure](#data-structure)
4. [Module Reference](#module-reference)
5. [Workflow Overview](#workflow-overview)
6. [Step-by-Step Instructions](#step-by-step-instructions)
7. [Troubleshooting](#troubleshooting)
8. [Tips for Best Results](#tips-for-best-results)

---

## Overview

**mfaCTpy** is a Python package for registering microCT-scanned mouse brain images to the Allen Common Coordinate Framework (CCF). It provides a complete workflow for processing microCT data with implanted optical fibers, enabling automated fiber tracking and brain region identification.

### Key Features

- DICOM to TIFF conversion for microCT volumes
- Interactive midline alignment with axis verification
- Landmark-based registration to Allen CCF with optional intensity-based refinement
- Manual fiber tracking with automatic brain region identification
- Interactive 3D visualization of fiber locations
- Movie generation for presentations
- Allen CCF annotation viewer with structure lookup

### Use Cases

- Identifying brain regions where optical fibers are implanted
- Registering microCT mouse brain scans to a standard atlas
- Visualizing fiber placement in anatomical context
- Creating presentation materials from 3D brain volumes

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Windows, macOS, or Linux

### Required Python Packages

```bash
pip install numpy tifffile matplotlib scipy SimpleITK pandas nrrd requests pydicom tqdm opencv-python bg-atlasapi
```

### Optional Packages

For additional functionality:
```bash
pip install scikit-image pillow
```

---

## Data Structure

### Recommended Project Structure

```
C:\DATA\MFA\uCT\uCT2CCF\
├── data\
│   ├── *.tif                    # Original microCT scan (3D TIFF)
│   ├── processed\
│   │   ├── microct_aligned.tif  # Midline-aligned image
│   │   └── microct_registered.tif # CCF-registered image
│   └── ccf\                     # Allen CCF data
│       ├── annotation_25.nrrd   # Brain region annotations
│       ├── average_template_25.nrrd # Reference template
│       └── structure_tree.json  # Region hierarchy
├── outputs\                     # Analysis results
└── src\                         # Python scripts
```

### Input Data

- **MicroCT DICOM files**: Raw scanner output (folder of .dcm files)
- **OR pre-converted .tif**: 3D TIFF stack

### Expected Image Properties

- **Resolution**: ~20 μm isotropic (typical for microCT)
- **Content**: Mouse head with skull, brain, and implanted fibers
- **Fiber diameter**: 50 μm (appears as 2-3 pixels at 20 μm resolution)
- **Fiber appearance**: Bright thin white lines; tiny white dots in horizontal slices

### Allen CCF Data

Allen CCF files are shared by the whole installation rather than copied into
every animal/project folder. They are cached in:

```text
resources/allen_ccf/
```

The first run of `microprism_atlas_mapper.py` automatically downloads:

- **Annotation volume** (annotation_25.nrrd): Brain region labels
- **Average template** (average_template_25.nrrd): Anatomical intensity image
- **Structure tree** (structure_tree.json): Region hierarchy and names

They can also be prepared in advance:

```bash
python src/allen_ccf_resources.py
```

Once downloaded, the same files are reused for all datasets.

---

## Module Reference

| Module | Purpose | Key Features |
|--------|---------|--------------|
| `dicom_loader.py` | DICOM to TIFF conversion | GUI file selection, 16-bit output |
| `data_loader.py` | Load microCT and CCF data | Uses BrainGlobe API, visualization |
| `preprocessing.py` | Image preprocessing | Resampling, normalization, denoising (experimental) |
| `midline_alignment.py` | Align brain midline | Interactive marking, axis verification |
| `landmark_registration.py` | Register to CCF | Landmark selection, affine transform, refinement |
| `fiber_tracker.py` | Track optical fibers | Manual tracking, CCF region lookup |
| `microprism_corner_tracker.py` | Track a microprism face | Four-corner marking and imaging-face derivation |
| `microprism_atlas_mapper.py` | Map a microprism face to CCF | Full-plane region sampling and labeled atlas map |
| `fiber_visualizer_3d.py` | 3D visualization | Interactive 3D view, slice planes |
| `movie_creator.py` | Create MP4 movies | GUI-based, multiple axes |
| `annotation_loader.py` | View CCF annotations | Interactive browser, structure colors |
| `registered_img_visualization.py` | Registration overlay | CCF overlay visualization |

---

## Workflow Overview

```
1. DICOM → TIFF Conversion (dicom_loader.py)
   ↓
2. Load Data (data_loader.py)
   ↓
3. Midline Alignment (midline_alignment.py)
   ↓
4. Axis Verification (midline_alignment.py)
   ↓
5. Landmark Registration (landmark_registration.py)
   ↓
6. Fiber Tracking (fiber_tracker.py)
   ↓
7. 3D Visualization (fiber_visualizer_3d.py)
   ↓
8. Create Movies / Reports (movie_creator.py, registered_img_visualization.py)
```

**Estimated Total Time**: 2-4 hours (depending on number of landmarks/fibers)

---

## Step-by-Step Instructions

### Step 1: Convert DICOM to TIFF

**Script**: `dicom_loader.py`

**Purpose**: Convert DICOM series from microCT scanner to a single 3D TIFF file.

**Usage**:
```bash
python dicom_loader.py
```

**Process**:
1. GUI dialog opens to select folder containing DICOM files
2. Select output location for TIFF file
3. Script loads all slices, normalizes to 16-bit, and saves

**Functions**:
- `load_dicom_volume(dicom_folder, normalize=True)`: Load DICOM series into numpy array
- `save_volume_as_tif(volume, output_path, bit_depth=16)`: Save as multi-page TIFF

**Output**: `*.tif` (3D TIFF stack)

**Time**: 2-5 minutes

---

### Step 2: Load and Inspect Data

**Script**: `data_loader.py`

**Purpose**: Load microCT .tif files and Allen CCF atlas for inspection.

**Usage**:
```python
from data_loader import DataLoader

loader = DataLoader(PROJECT_PATH)
microct = loader.load_microct()           # Auto-detects first .tif
atlas = loader.load_allen_ccf(resolution=25)  # Downloads if needed
loader.get_info()                          # Print summary
loader.visualize_slices()                  # Show comparison
```

**Key Features**:
- Uses BrainGlobe API for atlas download
- Supports 10, 25, or 50 μm CCF resolution
- Generates overview visualization

**Output**: Loaded numpy arrays, visualization in `outputs/01_data_overview.png`

---

### Step 3: Midline Alignment

**Script**: `midline_alignment.py`

**Purpose**: Align the brain midline to image center and correct orientation.

**Usage**:
```bash
python midline_alignment.py
```

**Interactive Steps**:

1. **View Selection**: Three orthogonal views are shown; script uses horizontal view for midline marking

2. **Mark Midline Points**:
   - LEFT-CLICK to mark points along brain midline
   - Navigate slices using slider
   - Mark 2-3 points per slice across 5-10 slices
   - Use "Clear Slice" or "Clear All" to redo

3. **Apply Alignment**: 
   - Script fits a plane to marked points
   - Calculates and applies rotation
   - Full resolution processing takes 5-30 minutes

4. **Axis Verification** (optional but recommended):
   - Interactive interface to verify coronal/axial/sagittal orientations
   - Swap or flip axes if needed
   - Save corrected image

**Output**:
- `microct_aligned.tif` or `microct_aligned_corrected.tif`
- `midline_points.json` (saved landmark points)
- `00_midline_alignment_fullres.png` (visualization)

**Time**: 15-30 minutes (marking) + 5-30 minutes (processing)

---

### Step 4: Landmark Registration

**Script**: `landmark_registration.py`

**Purpose**: Register aligned microCT image to Allen CCF atlas using manual landmarks.

**Usage**:
```bash
python landmark_registration.py
```

**Interactive Landmark Selection**:

1. **Interface**: Side-by-side display of microCT (MOVING) and CCF (FIXED)

2. **Marking Landmarks**:
   - Click on microCT image first, then corresponding CCF location
   - Use Z/Y/X buttons to switch viewing planes
   - Navigate slices with slider or scroll wheel
   - Need minimum 4 pairs (6-12 recommended)

3. **Good Landmarks**:
   - Bregma (skull suture intersection)
   - Lambda (posterior suture)
   - Brain tips (anterior, posterior, ventral)
   - Ventricle corners
   - Olfactory bulb tips

4. **Registration**:
   - Computes affine transformation
   - Reports mean/max landmark error
   - Option for intensity-based refinement (recommended)

**Quality Metrics**:
- Mean error < 0.5 mm: Excellent
- Mean error < 1.0 mm: Good
- Max error < 2.0 mm: Acceptable

**Output**:
- `microct_registered.tif`: Registered image
- `transform_landmark.tfm`: SimpleITK transform file
- `registration_metrics.json`: Error metrics
- `landmarks.json`: Saved landmark pairs
- Visualization images in `outputs/`

**Time**: 30-60 minutes (landmarks) + 5-20 minutes (computation)

---

### Step 5: Fiber Tracking

**Script**: `fiber_tracker.py`

**Purpose**: Manually track optical fibers and identify brain regions using CCF.

**Usage**:
```bash
python fiber_tracker.py
```

**Setup**:
1. Script checks for registered image (preferred) or aligned image + transform
2. Downloads CCF annotation if needed
3. Loads Allen CCF ontology for region names

**Tracking Interface**:

1. **Mark Fiber Entry Point**: Click at brain surface where fiber enters
2. **Mark Fiber Tip**: Navigate to tip location and click
3. **Save Fiber**: Adds to fiber list with automatic region identification
4. **Repeat**: Track all visible fibers

**Controls**:
- Slider/scroll: Navigate slices
- "New Fiber": Start tracking new fiber
- "Save Fiber": Complete current fiber
- "Undo": Remove last point
- "Save Progress": Save all tracked fibers
- "Export": Generate reports

**Output**:
- `fiber_data.json`: Complete fiber coordinates and regions
- `fiber_report.csv`: Spreadsheet-compatible data
- `fiber_summary.txt`: Human-readable summary
- `fiber_horizontal_view.png`: Overview visualization
- `fiber_ccf_overlay.png`: CCF overlay

**Time**: 5-10 minutes per fiber

---

### Microprism Imaging-Plane Mapping

Run the corner tracker first, then map the saved imaging face into the Allen
CCF annotation:

```bash
python src/microprism_corner_tracker.py /path/to/project
python src/microprism_atlas_mapper.py /path/to/project
```

If the microCT voxel dimensions differ by axis, pass the exact physical pixel
sizes in millimeters when tracking:

```bash
python src/microprism_corner_tracker.py /path/to/project \
  --spacing-x 0.020 --spacing-y 0.020 --spacing-z 0.025
```

Those values are saved in `microprism_corners.json` and reused by the mapper.
They can also be overridden at mapping time with `--source-spacing-x`,
`--source-spacing-y`, and `--source-spacing-z`.

The mapper automatically looks for:

- `outputs/microprism_corners.json`
- `data/ccf/annotation_25.nrrd`
- `data/ccf/structure_tree.json`
- `data/processed/transform_landmark.tfm` when the tracked image is unregistered

Paths can also be supplied explicitly with `--corners`, `--annotation`,
`--ontology`, and `--transform`. The output displays the complete corresponding
oblique atlas plane in an upright prism-aligned view, outlines the actual prism
face, labels the region at each corner, and saves CSV and JSON region summaries.
The face is rectified to a true rectangle using the prism width/length ratio
entered in the tracker, so affine registration skew is not shown as physical
prism tilt.
The interactive viewer starts centered on the prism. Scroll or use `+`/`-` to
zoom. Use `In-plane`, `Coronal`, `Sagittal`, and `Axial` to switch perspectives.
Click-drag the image, use the arrow controls, or press the arrow keys to pan.
Choose `Fit` (or press `0`) for the complete current slice, and press `R` to
reset its zoom. The Allen average anatomical template is displayed beneath
translucent annotation colors and region boundaries. Use `--no-template` for
the label-only view. The full calculation is documented in
`docs/microprism_atlas_mapping.md`.

`Map 2p` overlays a two-photon image onto the prism face. It asks for the
unzoomed image, then optionally a zoomed one and its magnification, which is
dragged into position over the unzoomed image. After the four field-of-view
corners are clicked, both images are warped onto the plane and the scale,
translation and transform matrices are written to
`microprism_atlas_plane_2p_alignment.json`. Images and magnification can also
be given up front with `--two-photon-image`, `--two-photon-zoom-image` and
`--two-photon-magnification`.

---

### Step 6: 3D Visualization

**Script**: `fiber_visualizer_3d.py`

**Purpose**: Interactive 3D visualization of tracked fibers in anatomical context.

**Usage**:
```bash
python fiber_visualizer_3d.py
```

**Requirements**: Requires `fiber_data.json` from fiber tracking step.

**Features**:
- 3D view with mouse rotation
- Toggle slice planes (coronal, sagittal, axial)
- Show/hide individual fibers
- Switch between microCT and CCF reference
- Preset camera views (Top, Side, Front, 3D)
- Zoom controls

**Controls**:
- Drag: Rotate view
- Scroll: Zoom
- Sliders: Navigate slice positions
- Checkboxes: Toggle fiber visibility
- Buttons: Change views, reset zoom

---

### Step 7: Create Movies

**Script**: `movie_creator.py`

**Purpose**: Generate MP4 movies from 3D volumes for presentations.

**Usage**:
```bash
python movie_creator.py
```

**GUI Features**:
- Visual preview of all three axes
- Select which axes to export
- Name each axis (coronal, sagittal, axial)
- Set frame rate (default: 20 fps)
- Compression quality control
- Optional vertical/horizontal flip

**Output**: MP4 movie files (e.g., `microct_registered_coronal_movie.mp4`)

**Time**: 2-5 minutes per movie

---

### Step 8: View Registration Overlay

**Script**: `registered_img_visualization.py`

**Purpose**: Create overlay visualizations of registered microCT with CCF annotations.

**Usage**:
```bash
python registered_img_visualization.py
```

**Process**:
1. Select registered .tif file
2. Select CCF folder with annotation and structure tree
3. Choose extraction plane (coronal, horizontal, sagittal)
4. Script creates colored CCF overlays for all slices

**Output**:
- Individual slice images with overlay
- Montage of representative sections
- Saved in `outputs/coronal_overlay_sections/`

---

### Utility: CCF Annotation Viewer

**Script**: `annotation_loader.py`

**Purpose**: Interactive browser for Allen CCF annotations.

**Usage**:
```bash
python annotation_loader.py
```

**Features**:
- Navigate through annotation volume
- Switch between coronal/sagittal/axial views
- Color by structure ID or anatomical colors
- Hover for structure name and ID
- Zoom, pan, and rotate controls
- Mouse wheel slice navigation

---

## Troubleshooting

### Midline alignment looks wrong

**Solutions**:
1. Mark more points (2-3 per slice on 5-10 slices)
2. Ensure points follow true anatomical midline
3. Try marking in different view (axial vs coronal)
4. Use `reverse=True` in `apply_alignment()` if rotation is backwards

### Axes appear swapped after alignment

**Solutions**:
1. Use axis verification step
2. Swap axes using interactive buttons
3. Save corrected image before registration

### Registration error is high (> 1 mm)

**Solutions**:
1. Review landmarks - remove outliers
2. Add more landmarks in poorly-aligned regions
3. Use clear, unambiguous anatomical features
4. Try intensity-based refinement
5. Ensure landmarks are visible in both images

### Fiber regions identified incorrectly

**Solutions**:
1. Verify using registered image (not aligned + transform)
2. Check CCF annotation loaded correctly
3. Verify registration quality first
4. Confirm fiber tip marked at actual location

### Scripts run very slowly

**Solutions**:
1. Check available RAM (8-16 GB recommended)
2. Close other applications
3. Full resolution alignment is expected to be slow (5-30 min)
4. Consider using machine with more CPU cores

### Allen CCF download fails

**Solutions**:
1. Check internet connection
2. Manual download from: http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/
3. Place files in `data/ccf/` folder
4. Required files: `annotation_25.nrrd`, `average_template_25.nrrd`, `structure_tree.json`

---

## Tips for Best Results

### Imaging Tips

1. Scan at highest resolution possible (20 μm or better)
2. Include clear skull landmarks (bregma, lambda) if visible
3. Ensure good contrast between brain tissue and background
4. Center the brain in the scan volume

### Midline Alignment Tips

1. Mark points carefully - accuracy affects all downstream steps
2. Use multiple slices spanning anterior-posterior extent
3. Verify alignment before proceeding to registration
4. Always run axis verification step

### Registration Tips

1. Select 8-12 landmarks for best results (minimum 4-6)
2. Distribute landmarks throughout the volume (don't cluster)
3. Use clear, unambiguous features visible in both images
4. Check different anatomical planes to find best landmark views
5. Always use intensity-based refinement for best accuracy

### Fiber Tracking Tips

1. Fibers appear as bright thin lines (dots in horizontal slices)
2. Navigate through multiple slices to confirm fiber path
3. Mark entry point at brain surface (not in skull)
4. Mark tip accurately - small errors change region ID
5. Save progress frequently
6. Use Undo liberally for mistakes

---

## File Format Reference

### Input Files

| File | Format | Description |
|------|--------|-------------|
| DICOM series | `.dcm` | Raw microCT scan |
| MicroCT volume | `.tif` | 3D TIFF stack (Z, Y, X) |

### Intermediate Files

| File | Format | Description |
|------|--------|-------------|
| `microct_aligned.tif` | `.tif` | Midline-aligned image |
| `microct_aligned_corrected.tif` | `.tif` | Axis-corrected aligned image |
| `microct_registered.tif` | `.tif` | CCF-registered image |
| `transform_landmark.tfm` | `.tfm` | SimpleITK transform |
| `transform_refined.tfm` | `.tfm` | Refined registration transform |
| `landmarks.json` | `.json` | Registration landmarks |
| `midline_points.json` | `.json` | Midline alignment points |
| `registration_metrics.json` | `.json` | Registration quality metrics |

### Output Files

| File | Format | Description |
|------|--------|-------------|
| `fiber_data.json` | `.json` | Complete fiber data with coordinates |
| `fiber_report.csv` | `.csv` | Spreadsheet-compatible fiber data |
| `fiber_summary.txt` | `.txt` | Human-readable fiber summary |
| `fiber_horizontal_view.png` | `.png` | Fiber overview visualization |
| `fiber_ccf_overlay.png` | `.png` | CCF overlay visualization |
| `microprism_atlas_plane.png` | `.png` | Labeled CCF region map across the imaging face |
| `microprism_atlas_regions.csv` | `.csv` | Region coverage percentages for the imaging face |
| `microprism_atlas_regions.json` | `.json` | CCF corners, mapping settings, and region summary |
| `*_movie.mp4` | `.mp4` | Slice movies |
| `coronal_overlay_*.png` | `.png` | Registration overlay images |

---

## Coordinate Conventions

### Image Array Convention

All 3D arrays use **(Z, Y, X)** convention:
- **Z**: Dorsal-Ventral (top-bottom) / Coronal slice index
- **Y**: Anterior-Posterior (front-back)
- **X**: Medial-Lateral (left-right)

### Physical Space Convention

SimpleITK and Allen CCF use **(x, y, z)** convention:
- **x**: Left-Right
- **y**: Anterior-Posterior
- **z**: Dorsal-Ventral

### Slice Views

- **Coronal**: Slicing through Z, viewing Y-X plane (front/back view)
- **Axial/Horizontal**: Slicing through Y, viewing Z-X plane (top-down view)
- **Sagittal**: Slicing through X, viewing Z-Y plane (side view)

---

## Citation

If you use this package in your research, please cite:

```
Multi-fiber array-based photometry system for multi-regional functional mapping in the mouse brain
Manil Bradai, Mirna Merkler, Gabriela Gil, Rebecca Davie, Shuzo Sakata
bioRxiv 2025.12.23.696162; doi: https://doi.org/10.64898/2025.12.23.696162

Sakata, S. mfaCTpy: Python-based package for multi-fiber array tracing based on microCT images
https://github.com/Sakata-Lab/mfaCTpy
```
---

## Version History

- **v1.1** (2025): Updated documentation
  - Accurate module descriptions
  - Complete workflow documentation
  - Improved troubleshooting section

- **v1.0** (2025): Initial release
  - DICOM to TIFF conversion
  - Interactive midline alignment with axis verification
  - Landmark-based registration
  - Fiber tracking with CCF integration
  - 3D visualization
  - Movie generation
  - CCF annotation viewer

---

## Support

For questions or issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the [Tips for Best Results](#tips-for-best-results)
3. Verify all file paths are correct
4. Check that all dependencies are installed

---
