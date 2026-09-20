import os
import numpy as np
import json
from pathlib import Path
import argparse
import tifffile
import nrrd
import matplotlib.pyplot as plt

from project_paths import resolve_project_path


def first_existing(candidates):
    for candidate in candidates:
        candidate = Path(candidate).expanduser()
        if candidate.exists():
            return candidate.resolve()
    return None


def search_project(project, names):
    direct = []
    resource_dir = Path(__file__).resolve().parent.parent / "resources" / "allen_ccf"
    for name in names:
        direct.extend([
            project / "data" / "processed" / name,
            project / "outputs" / name,
            project / "data" / "ccf" / name,
            project / name,
            resource_dir / name,
        ])
    found = first_existing(direct)
    if found:
        return found
    for name in names:
        matches = sorted(project.rglob(name))
        if matches:
            return matches[0].resolve()
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create 2D registered microCT/Allen CCF slice overlays."
    )
    parser.add_argument(
        "project_path",
        nargs="?",
        help="Project folder. Defaults to the current project folder.",
    )
    parser.add_argument(
        "--image",
        help="Registered microCT TIFF. Defaults to data/processed/microct_registered.tif.",
    )
    parser.add_argument(
        "--ccf-folder",
        help="Folder containing annotation_25.nrrd and structure_tree.json.",
    )
    parser.add_argument(
        "--annotation",
        help="Path to annotation_25.nrrd.",
    )
    parser.add_argument(
        "--structure-tree",
        help="Path to structure_tree.json.",
    )
    parser.add_argument(
        "--plane",
        choices=("coronal", "horizontal", "sagittal"),
        default="coronal",
        help="Plane to extract as individual grayscale slices.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Slice interval. Use a larger number for a faster preview.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="MicroCT contribution in overlay images.",
    )
    return parser.parse_args()

def load_structure_tree(structure_tree_path):
    """Load and parse structure_tree.json to get region colors"""
    json_path = Path(structure_tree_path)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create mapping: annotation_id -> RGB color
    color_map = {}
    
    def parse_structure(struct):
        """Recursively parse structure tree"""
        struct_id = struct.get('id')
        color_hex = struct.get('color_hex_triplet', '000000')
        
        # Convert hex to RGB (0-255)
        r = int(color_hex[0:2], 16)
        g = int(color_hex[2:4], 16)
        b = int(color_hex[4:6], 16)
        
        color_map[struct_id] = [r, g, b]
        
        # Recursively process children
        if 'children' in struct:
            for child in struct['children']:
                parse_structure(child)
    
    # Parse the structure tree
    for msg in data.get('msg', []):
        parse_structure(msg)
    
    return color_map

def create_colored_annotation(annotation_data, color_map):
    """Convert annotation volume to RGB using structure_tree colors (optimized)"""
    print("Creating colored annotation volume (this may take a moment)...")
    
    h, w, d = annotation_data.shape
    rgb_volume = np.zeros((h, w, d, 3), dtype=np.uint8)
    
    # Get unique annotation IDs
    unique_ids = np.unique(annotation_data)
    print(f"Found {len(unique_ids)} unique brain regions")
    
    # Create a lookup table for faster color mapping
    max_id = int(annotation_data.max())
    color_lut = np.zeros((max_id + 1, 3), dtype=np.uint8)
    
    for ann_id in unique_ids:
        if ann_id == 0:  # Skip background
            continue
        color = color_map.get(int(ann_id), [128, 128, 128])  # Default gray if not found
        color_lut[int(ann_id)] = color
    
    # Vectorized color mapping - much faster!
    print("Applying colors to volume...")
    for c in range(3):  # For each color channel
        rgb_volume[:, :, :, c] = color_lut[annotation_data.astype(int), c]
    
    return rgb_volume

def show_plane_selection(microct_data):
    """Show three sample slices from each plane for user selection"""
    print("\nDisplaying sample slices from each plane...")
    print("Close the window after reviewing to make your selection.")
    
    h, w, d = microct_data.shape
    
    # Get middle slices from each dimension
    coronal_slice = microct_data[h // 2, :, :]
    horizontal_slice = microct_data[:, w // 2, :]
    sagittal_slice = microct_data[:, :, d // 2]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Normalize and display each plane
    for ax, slice_data, title in zip(
        axes, 
        [coronal_slice, horizontal_slice, sagittal_slice],
        ['Coronal (axis 0)', 'Horizontal (axis 1)', 'Sagittal (axis 2)']
    ):
        slice_norm = ((slice_data - slice_data.min()) / 
                      (slice_data.max() - slice_data.min() + 1e-8))
        ax.imshow(slice_norm, cmap='gray')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Sample Slices from Each Plane - Review and Choose', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Ask user to select plane
    print("\nWhich plane would you like to extract?")
    print("0 = Coronal (axis 0)")
    print("1 = Horizontal (axis 1)")
    print("2 = Sagittal (axis 2)")
    
    while True:
        try:
            choice = int(input("Enter 0, 1, or 2: "))
            if choice in [0, 1, 2]:
                plane_names = ['coronal', 'horizontal', 'sagittal']
                print(f"\nSelected: {plane_names[choice]}")
                return choice
            else:
                print("Invalid choice. Please enter 0, 1, or 2.")
        except ValueError:
            print("Invalid input. Please enter a number (0, 1, or 2).")

def extract_slices(tif_data, output_folder, axis, step=1):
    """Extract slices from selected plane
    
    Args:
        tif_data: 3D numpy array
        output_folder: Path to save slices
        axis: Which axis to slice (0=coronal, 1=horizontal, 2=sagittal)
        step: Slice interval (1 = extract ALL slices)
    """
    os.makedirs(output_folder, exist_ok=True)
    
    plane_names = ['coronal', 'horizontal', 'sagittal']
    plane_name = plane_names[axis]
    
    num_slices = tif_data.shape[axis]
    slice_indices = range(0, num_slices, step)
    
    print(f"\nExtracting {len(slice_indices)} {plane_name} slices (every {step} slice)...")
    print(f"Data shape: {tif_data.shape}")
    
    for i in slice_indices:
        if axis == 0:
            slice_data = tif_data[i, :, :]
        elif axis == 1:
            slice_data = tif_data[:, i, :]
        else:  # axis == 2
            slice_data = tif_data[:, :, i]
        
        output_path = os.path.join(output_folder, f"{plane_name}_slice_{i:04d}.png")
        
        # Normalize and save
        slice_norm = ((slice_data - slice_data.min()) / 
                      (slice_data.max() - slice_data.min() + 1e-8) * 255).astype(np.uint8)
        
        plt.imsave(output_path, slice_norm, cmap='gray')
    
    print(f"All slices saved to: {output_folder}")
    return slice_indices

def create_overlayed_coronal_sections(microct_data, annotation_rgb, output_folder, 
                                     step=1, alpha=0.6):
    """Create multiple coronal section images with overlay
    
    Args:
        microct_data: 3D numpy array of microCT data
        annotation_rgb: 3D RGB numpy array of colored annotations
        output_folder: Folder to save individual section images
        step: Step size for sections (1 = ALL sections)
        alpha: Transparency of microCT overlay (0=only atlas, 1=only microCT)
    """
    os.makedirs(output_folder, exist_ok=True)
    
    h, w, d = microct_data.shape
    
    # Extract ALL coronal sections (or every 'step' sections)
    section_indices = range(0, h, step)
    num_sections = len(section_indices)
    
    print(f"\nCreating {num_sections} overlayed coronal sections (every {step} slice)...")
    print(f"This may take a while for large volumes...")
    
    for idx, section_num in enumerate(section_indices):
        if idx % 10 == 0:  # Progress update every 10 slices
            print(f"  Processing section {idx + 1}/{num_sections}...")
        
        # Extract coronal slices
        microct_slice = microct_data[section_num, :, :]
        ccf_slice = annotation_rgb[section_num, :, :, :]
        
        # Normalize microCT slice
        microct_norm = ((microct_slice - microct_slice.min()) / 
                        (microct_slice.max() - microct_slice.min() + 1e-8))
        
        # Convert to RGB for overlay
        microct_rgb = np.stack([microct_norm] * 3, axis=-1) * 255
        
        # Create overlay
        overlay = (alpha * microct_rgb + (1 - alpha) * ccf_slice).astype(np.uint8)
        
        # Create figure for this section
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # MicroCT only
        axes[0].imshow(microct_norm, cmap='gray')
        axes[0].set_title('MicroCT', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Allen CCF only
        axes[1].imshow(ccf_slice)
        axes[1].set_title('Allen CCF Atlas', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title(f'Overlay (α={alpha})', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.suptitle(f'Coronal Section {idx + 1}/{num_sections} (Index: {section_num})', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save individual section
        output_path = os.path.join(output_folder, f'coronal_overlay_{section_num:04d}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\nAll {num_sections} overlay sections saved to: {output_folder}")
    
    # Create a montage of representative sections (not all, as that would be huge)
    print("\nCreating montage of representative sections...")
    create_section_montage(output_folder, min(num_sections, 20))

def create_section_montage(output_folder, max_sections=20):
    """Create a single montage image with representative sections
    
    Args:
        output_folder: Folder containing section images
        max_sections: Maximum number of sections to include in montage
    """
    import glob
    from PIL import Image
    
    # Find all section images
    section_files = sorted(glob.glob(os.path.join(output_folder, 'coronal_overlay_*.png')))
    
    if not section_files:
        print("No section images found for montage.")
        return
    
    # If there are many sections, sample evenly for the montage
    total_sections = len(section_files)
    if total_sections > max_sections:
        indices = np.linspace(0, total_sections - 1, max_sections, dtype=int)
        section_files = [section_files[i] for i in indices]
        print(f"Creating montage with {max_sections} representative sections out of {total_sections} total...")
    else:
        print(f"Creating montage with all {total_sections} sections...")
    
    # Load images
    images = [Image.open(f) for f in section_files]
    
    # Calculate montage dimensions
    img_width, img_height = images[0].size
    cols = min(4, len(images))
    rows = (len(images) + cols - 1) // cols
    
    # Create montage
    montage_width = img_width * cols
    montage_height = img_height * rows
    montage = Image.new('RGB', (montage_width, montage_height), color='white')
    
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * img_width
        y = row * img_height
        montage.paste(img, (x, y))
    
    # Save montage
    montage_path = os.path.join(output_folder, 'coronal_sections_montage.png')
    montage.save(montage_path, dpi=(150, 150))
    print(f"Montage saved to: {montage_path}")

def main():
    args = parse_args()
    print("=" * 60)
    print("uCT Registration Viewer with Plane Selection")
    print("=" * 60)

    project = resolve_project_path(args.project_path)
    tif_path = (
        Path(args.image).expanduser().resolve()
        if args.image else search_project(
            project,
            ("microct_registered.tif", "microct_registered.tiff"),
        )
    )
    if tif_path is None or not tif_path.exists():
        print("\nERROR: registered microCT image not found.")
        print(f"Expected: {project / 'data' / 'processed' / 'microct_registered.tif'}")
        print("Or pass it explicitly with --image.")
        return
    print(f"Selected: {tif_path}")

    ccf_folder = Path(args.ccf_folder).expanduser().resolve() if args.ccf_folder else None
    annotation_path = (
        Path(args.annotation).expanduser().resolve()
        if args.annotation else (
            ccf_folder / "annotation_25.nrrd"
            if ccf_folder else search_project(project, ("annotation_25.nrrd",))
        )
    )
    structure_tree_path = (
        Path(args.structure_tree).expanduser().resolve()
        if args.structure_tree else (
            ccf_folder / "structure_tree.json"
            if ccf_folder else search_project(project, ("structure_tree.json",))
        )
    )

    if not annotation_path.exists():
        print(f"ERROR: {annotation_path} not found!")
        return
    if not structure_tree_path.exists():
        print(f"ERROR: {structure_tree_path} not found!")
        return

    print(f"CCF annotation: {annotation_path}")
    print(f"Structure tree: {structure_tree_path}")

    outputs_folder = project / "outputs"
    outputs_folder.mkdir(exist_ok=True)
    print(f"\nProject folder: {project}")

    # Step 3: Load data
    print("\nStep 3: Loading microCT data...")
    microct_data = tifffile.imread(tif_path)
    print(f"MicroCT shape: {microct_data.shape}")

    # Step 4: Select extraction plane from command-line option
    plane_names = ['coronal', 'horizontal', 'sagittal']
    selected_axis = plane_names.index(args.plane)
    print(f"\nStep 4: Selected extraction plane: {args.plane}")

    # Step 5: Extract slices from selected plane
    if args.step <= 0:
        raise ValueError("--step must be a positive integer.")
    print("\nStep 5: Extracting slices from selected plane...")
    slice_folder = outputs_folder / f"{plane_names[selected_axis]}_slices"
    slice_indices = extract_slices(
        microct_data, 
        str(slice_folder), 
        axis=selected_axis,
        step=args.step
    )

    # Step 6: Load Allen CCF data
    print("\nStep 6: Loading Allen CCF annotation...")
    annotation_data, _ = nrrd.read(str(annotation_path))
    print(f"Annotation shape: {annotation_data.shape}")
    
    print("\nStep 7: Loading structure tree and creating color map...")
    color_map = load_structure_tree(structure_tree_path)
    print(f"Loaded {len(color_map)} brain regions")
    
    print("\nStep 8: Creating colored annotation volume...")
    annotation_rgb = create_colored_annotation(annotation_data, color_map)
    print("Color mapping complete")
    
    # Step 7: Create overlayed coronal sections
    print("\nStep 9: Creating overlayed coronal sections...")
    sections_folder = outputs_folder / "coronal_overlay_sections"
    create_overlayed_coronal_sections(
        microct_data, 
        annotation_rgb, 
        str(sections_folder),
        step=args.step,
        alpha=args.alpha
    )
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)
    print(f"Extracted slices: {slice_folder}")
    print(f"Overlayed sections: {sections_folder}")

if __name__ == "__main__":
    main()
