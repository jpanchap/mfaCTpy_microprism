# Cross-platform setup and testing

Python 3.11 is recommended. The source and regression tests also support Python
3.9. The dependency ranges in `requirements.txt` are not a lockfile: pip selects
releases compatible with your Python version and operating system.

## Install

From the repository folder, with Conda installed, use these commands on macOS,
Windows, or Linux:

```sh
conda env create -f environment.yml
conda activate mfaCT
python -m pip check
python -m tkinter
```

The last command opens a small test window; close it before continuing.
For an existing environment, run `python -m pip install -r requirements.txt`.
Tk must be supplied by the Python installation, not pip. Conda includes it via
`environment.yml`; Debian/Ubuntu system Python needs the matching `python3-tk`
package. Use a graphical desktop session for interactive viewers. A headless
Linux server needs a working forwarded display for those steps.

## Run

Pass the project containing `data/` to each script, or launch the script while
your working directory is inside that project. Discovery walks upward to find
the project. Quote paths containing spaces. Source files need no path edits.

| Platform | Example from the repository folder |
| --- | --- |
| macOS / Linux | `python src/data_loader.py "/path/to/project"` |
| Windows | `python src/data_loader.py "C:\Data\My project"` |

The same project argument works for alignment, registration, tracking, mapping,
preprocessing, and annotation inspection. Preprocessing additionally requires
`--original-spacing` in micrometers; provide the verified isotropic voxel size
for your scan. Movie creation accepts a TIFF path directly or opens a file
picker when omitted. Use `--help` for individual options.

Matplotlib must use an interactive backend for manual selection. If automatic
selection fails, verify Tk with `python -m tkinter` and select `TkAgg` through
the `MPLBACKEND` environment variable. `Agg` is only for headless tests.
The mapper uses macOS native dialogs on macOS and Tk/terminal prompts elsewhere.

Text files use UTF-8. For redirected Windows console output, set
`$env:PYTHONIOENCODING="utf-8"` in PowerShell. Saved JSON can contain absolute
paths for provenance; when moving a project between computers, provide the new
local paths through the scripts' image/transform arguments.

Movies require OpenCV: compressed output uses MPEG-4 (`mp4v`) in MP4; lossless
output uses HFYU in AVI. Availability depends on the OpenCV build. The movie
regression test encodes and decodes a tiny example of each format.

## Verification

Set `MPLBACKEND=Agg` and run `python -m unittest discover -s tests -v`.
GitHub Actions installs dependencies, checks them, imports every published
module, and tests Linux, Windows, and Intel macOS with Python 3.9 and 3.11,
plus Apple Silicon macOS with Python 3.11. The workflow runs after publication;
local Mac tests do not establish Windows/Linux compatibility. Check every CI
job before declaring those platforms verified. Headless tests do not replace
desktop interaction checks or scientific validation on representative scans.

Both `output/` and `outputs/` are ignored by Git. Local reports, images, and
diagnostics remain on disk but are excluded from normal staging. Do not
force-add these directories. Publish source, tests, and documentation instead.
