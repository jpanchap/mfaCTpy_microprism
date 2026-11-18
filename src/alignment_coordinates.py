"""Coordinate mappings use homogeneous column vectors in Z,Y,X voxel units."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import tifffile


def swap_mapping(a, b):
    matrix = np.eye(4)
    matrix[[a, b]] = matrix[[b, a]]
    return matrix


def flip_mapping(axis, size):
    matrix = np.eye(4)
    matrix[axis, axis] = -1
    matrix[axis, 3] = size - 1
    return matrix


def voxel_digest(image):
    """Hash voxel data, shape and dtype, independent of TIFF compression/tags."""
    digest = hashlib.sha256()
    digest.update(str((image.shape, image.dtype.str)).encode())
    for plane in image:
        digest.update(np.ascontiguousarray(plane).tobytes())
    return digest.hexdigest()


def backup(path):
    path = Path(path)
    if path.exists():
        destination = path.with_name(path.name + f'.backup-{time.time_ns()}')
        shutil.copy2(path, destination)
        print(f'Preserved existing file: {destination}')


def save_mapping(processed_path, raw_path, image, raw_shape, matrix):
    path = Path(str(processed_path) + '.alignment.json')
    backup(path)
    record = dict(schema_version=1, coordinate_order='ZYX', units='voxels',
                  processed_image=str(Path(processed_path).resolve()),
                  raw_image=str(Path(raw_path).resolve()),
                  processed_shape_zyx=list(image.shape), raw_shape_zyx=list(raw_shape),
                  processed_voxel_sha256=voxel_digest(image),
                  processed_to_raw_zyx=np.asarray(matrix).tolist(),
                  raw_to_processed_zyx=np.linalg.inv(matrix).tolist())
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f'Saved processed-to-raw coordinate mapping: {path}')


def transform_points(points, matrix):
    points = np.asarray(points, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError('Expected finite N by 3 ZYX points.')
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError('Expected a finite 4 by 4 mapping.')
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def convert_landmarks(landmark_path, mapping_path, selection_image, output_path,
                      selection_grid_unchanged=False):
    if not selection_grid_unchanged:
        raise ValueError('Confirm selection used this TIFF grid without --resample-moving.')
    record = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    if record.get('coordinate_order') != 'ZYX' or record.get('units') != 'voxels':
        raise ValueError('Mapping must use ZYX voxel coordinates.')
    try:
        image = tifffile.memmap(selection_image, mode="r")
    except ValueError:
        image = tifffile.imread(selection_image)
    try:
        digest = voxel_digest(image)
    finally:
        mapping = getattr(image, '_mmap', None)
        if mapping is not None:
            mapping.close()
    if digest != record['processed_voxel_sha256']:
        raise ValueError('Selection TIFF differs from mapping TIFF. Do not apply a new alignment to old landmarks. Preserve the old TIFF and recover its mapping first.')
    data = json.loads(Path(landmark_path).read_text(encoding="utf-8"))
    if data.get('coordinate_convention') != 'Z, Y, X':
        raise ValueError('Landmark coordinate convention must be Z, Y, X.')
    points = np.asarray(data['moving_landmarks'], dtype=float)
    raw = transform_points(points, record['processed_to_raw_zyx'])
    roundtrip = transform_points(raw, record['raw_to_processed_zyx'])
    if not np.allclose(points, roundtrip, atol=1e-7, rtol=0):
        raise ValueError('Coordinate round-trip check failed.')
    data['moving_landmarks_raw'] = raw.tolist()
    data['raw_coordinate_convention'] = 'Z, Y, X'
    data['raw_coordinate_units'] = 'voxels'
    data['raw_mapping_provenance'] = record
    data['raw_mapping_provenance']['selection_image_checked'] = str(Path(selection_image).resolve())
    inside = np.all((raw >= 0) & (raw <= np.asarray(record['raw_shape_zyx']) - 1), axis=1)
    data['moving_landmarks_raw_in_bounds'] = inside.tolist()
    backup(output_path)
    Path(output_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f'Saved both coordinate sets: {output_path}; {inside.sum()}/{len(raw)} raw points in bounds.')


def main():
    parser = argparse.ArgumentParser(description='Add raw CT coordinates to existing landmark JSON; no atlas registration is run.')
    parser.add_argument('landmarks')
    parser.add_argument('mapping')
    parser.add_argument('--selection-image', required=True, help='Exact TIFF on which existing points were selected (can be a preserved old copy).')
    parser.add_argument('--selection-grid-unchanged', action='store_true', help='Confirm landmark selection did NOT use --resample-moving.')
    parser.add_argument('--output', help='Defaults to updating landmarks JSON after backing it up.')
    args = parser.parse_args()
    convert_landmarks(args.landmarks, args.mapping, args.selection_image,
                      args.output or args.landmarks, args.selection_grid_unchanged)


if __name__ == '__main__':
    main()
