"""
DICOM Volume Loader for MicroCT Mouse Brain Data
Loads a series of DICOM files from a folder and converts to 3D numpy array
"""

import numpy as np
import pydicom
from pathlib import Path
from typing import Tuple, Optional
import glob
from tqdm import tqdm
import argparse
import json



def load_dicom_volume(dicom_folder: str, normalize: bool = True) -> Tuple[np.ndarray, dict]:
    """
    Load DICOM series from a folder into a 3D volume.
    
    Parameters:
    -----------
    dicom_folder : str
        Path to folder containing DICOM (.dcm) files
    normalize : bool
        If True, normalize volume to [0, 1] range
        
    Returns:
    --------
    volume : np.ndarray
        3D array of shape (height, width, slices)
    metadata : dict
        Dictionary containing DICOM metadata
    """
    
    dicom_folder = Path(dicom_folder)
    
    if not dicom_folder.exists():
        raise FileNotFoundError(f"Directory not found: {dicom_folder}")
    
    # Find all DICOM files
    dcm_files = sorted(glob.glob(str(dicom_folder / "*.dcm")))
    
    if len(dcm_files) == 0:
        raise ValueError(f"No DICOM files found in {dicom_folder}")
    
    print(f"Loading {len(dcm_files)} DICOM files from: {dicom_folder}")
    
    # Read first file to get dimensions
    first_slice = pydicom.dcmread(dcm_files[0])
    img_shape = first_slice.pixel_array.shape
    
    # Initialize volume as (Z, Y, X), which is the convention used by the
    # alignment and registration scripts.
    volume = np.zeros((len(dcm_files), img_shape[0], img_shape[1]), dtype=np.float64)
    
    # Load all slices
    metadata = {}
    for i, dcm_file in enumerate(tqdm(dcm_files, desc="Loading DICOM slices")):
        ds = pydicom.dcmread(dcm_file)
        volume[i, :, :] = ds.pixel_array.astype(np.float64)
        
        # Store metadata from first slice
        if i == 0:
            metadata = {
                'PixelSpacing': getattr(ds, 'PixelSpacing', None),
                'SliceThickness': getattr(ds, 'SliceThickness', None),
                'SpacingBetweenSlices': getattr(ds, 'SpacingBetweenSlices', None),
                'ImageOrientationPatient': getattr(ds, 'ImageOrientationPatient', None),
                'ImagePositionPatient': getattr(ds, 'ImagePositionPatient', None),
                'Rows': ds.Rows,
                'Columns': ds.Columns,
                'NumberOfSlices': len(dcm_files)
            }
    
    # Normalize if requested
    if normalize:
        volume = volume / np.max(volume)
    
    print(
        "Volume loaded successfully. Dimensions (Z, Y, X): "
        f"{volume.shape[0]} x {volume.shape[1]} x {volume.shape[2]}"
    )
    
    return volume, metadata


def metadata_to_spacing_um(metadata: dict) -> tuple[float, float, float] | None:
    """Return DICOM spacing in (Z, Y, X) micrometers if available."""
    pixel_spacing = metadata.get("PixelSpacing")
    if pixel_spacing is None:
        return None

    z_spacing_mm = metadata.get("SpacingBetweenSlices")
    if z_spacing_mm is None:
        z_spacing_mm = metadata.get("SliceThickness")
    if z_spacing_mm is None:
        return None

    row_spacing_mm = float(pixel_spacing[0])
    col_spacing_mm = float(pixel_spacing[1])
    return (
        float(z_spacing_mm) * 1000.0,
        row_spacing_mm * 1000.0,
        col_spacing_mm * 1000.0,
    )


def write_metadata_sidecar(metadata: dict, output_path: str | Path) -> None:
    """Save spacing metadata next to the TIFF for later pipeline steps."""
    output_path = Path(output_path)
    sidecar_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    spacing_um_zyx = metadata_to_spacing_um(metadata)
    sidecar = {
        "source": "dicom_loader.py",
        "axis_order": "Z,Y,X",
        "spacing_um_zyx": list(spacing_um_zyx) if spacing_um_zyx else None,
        "spacing_um_xyz": (
            [spacing_um_zyx[2], spacing_um_zyx[1], spacing_um_zyx[0]]
            if spacing_um_zyx else None
        ),
        "dicom": {
            "PixelSpacing": (
                [float(v) for v in metadata.get("PixelSpacing")]
                if metadata.get("PixelSpacing") is not None else None
            ),
            "SliceThickness": (
                float(metadata.get("SliceThickness"))
                if metadata.get("SliceThickness") is not None else None
            ),
            "SpacingBetweenSlices": (
                float(metadata.get("SpacingBetweenSlices"))
                if metadata.get("SpacingBetweenSlices") is not None else None
            ),
            "Rows": metadata.get("Rows"),
            "Columns": metadata.get("Columns"),
            "NumberOfSlices": metadata.get("NumberOfSlices"),
        },
    }
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"Metadata sidecar saved to: {sidecar_path}")


def save_volume_as_tif(volume: np.ndarray, output_path: str, bit_depth: int = 16):
    """
    Save 3D volume as multi-page TIFF file.
    
    Parameters:
    -----------
    volume : np.ndarray
        3D volume array
    output_path : str
        Output file path
    bit_depth : int
        Bit depth for output (8 or 16)
    """
    from tifffile import imwrite
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if bit_depth == 16:
        volume_out = (volume * 65535).astype(np.uint16)
    elif bit_depth == 8:
        volume_out = (volume * 255).astype(np.uint8)
    else:
        raise ValueError("bit_depth must be 8 or 16")
    
    print(f"Saving volume to: {output_path}")
    imwrite(output_path, volume_out, compression='none')
    print(f"Successfully saved {volume_out.shape[0]} slices")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a DICOM slice folder to a 3D TIFF in (Z, Y, X) order.")
    parser.add_argument("dicom_folder", help="Folder containing DICOM .dcm files.")
    parser.add_argument("output_path", help="Output TIFF path.")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Keep raw DICOM intensities instead of scaling to 0-1.")
    args = parser.parse_args()

    volume, metadata = load_dicom_volume(
        args.dicom_folder, normalize=not args.no_normalize)
    save_volume_as_tif(volume, args.output_path)
    write_metadata_sidecar(metadata, args.output_path)
