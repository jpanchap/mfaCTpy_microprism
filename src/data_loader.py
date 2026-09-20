"""
uCT2CCF - Data Loading Module
Loads microCT .tif files and Allen CCF atlas
"""

import numpy as np
import tifffile
from pathlib import Path
from bg_atlasapi import BrainGlobeAtlas
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class DataLoader:
    """Load and inspect microCT and Allen CCF data"""
    
    def __init__(self, project_path):
        """
        Initialize data loader
        
        Parameters:
        -----------
        project_path : str or Path
            Path to the uCT2CCF project folder
        """
        self.project_path = Path(project_path)
        self.data_path = self.project_path / "data"
        self.microct_image = None
        self.atlas = None
        self.atlas_image = None
        
    def load_microct(self, filename=None):
        """
        Load microCT .tif file
        
        Parameters:
        -----------
        filename : str, optional
            Specific filename to load. If None, loads the first .tif file found
        
        Returns:
        --------
        numpy.ndarray : 3D image array
        """
        if filename is None:
            # Find first .tif file in data folder
            tif_files = list(self.data_path.glob("*.tif")) + list(self.data_path.glob("*.tiff"))
            if not tif_files:
                raise FileNotFoundError(f"No .tif files found in {self.data_path}")
            filepath = tif_files[0]
            print(f"Loading first .tif file found: {filepath.name}")
        else:
            filepath = self.data_path / filename
            
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        print(f"Loading microCT image: {filepath}")
        self.microct_image = tifffile.imread(str(filepath))
        
        print(f"✓ MicroCT image loaded successfully")
        print(f"  Shape: {self.microct_image.shape}")
        print(f"  Data type: {self.microct_image.dtype}")
        print(f"  Intensity range: [{self.microct_image.min()}, {self.microct_image.max()}]")
        print(f"  Memory size: {self.microct_image.nbytes / 1024**2:.2f} MB")
        
        return self.microct_image
    
    def load_allen_ccf(self, resolution=25):
        """
        Load Allen CCF atlas
        
        Parameters:
        -----------
        resolution : int
            Atlas resolution in microns (10, 25, or 50)
        
        Returns:
        --------
        BrainGlobeAtlas : Atlas object
        """
        print(f"\nDownloading/Loading Allen CCF atlas at {resolution}μm resolution...")
        print("(This may take a few minutes on first run)")
        
        self.atlas = BrainGlobeAtlas("allen_mouse_" + str(resolution) + "um")
        self.atlas_image = self.atlas.reference
        
        print(f"✓ Allen CCF atlas loaded successfully")
        print(f"  Atlas name: {self.atlas.atlas_name}")
        print(f"  Resolution: {self.atlas.resolution} μm")
        print(f"  Shape: {self.atlas_image.shape}")
        print(f"  Orientation: {self.atlas.orientation}")
        print(f"  Data type: {self.atlas_image.dtype}")
        print(f"  Intensity range: [{self.atlas_image.min()}, {self.atlas_image.max()}]")
        
        return self.atlas
    
    def get_info(self):
        """Print summary information about loaded data"""
        print("\n" + "="*60)
        print("DATA SUMMARY")
        print("="*60)
        
        if self.microct_image is not None:
            print("\nMicroCT Image:")
            print(f"  Shape (Z, Y, X): {self.microct_image.shape}")
            print(f"  Voxel count: {np.prod(self.microct_image.shape):,}")
            print(f"  Data type: {self.microct_image.dtype}")
            print(f"  Value range: [{self.microct_image.min()}, {self.microct_image.max()}]")
            print(f"  Mean intensity: {self.microct_image.mean():.2f}")
            print(f"  Std intensity: {self.microct_image.std():.2f}")
        else:
            print("\nMicroCT Image: Not loaded")
        
        if self.atlas_image is not None:
            print("\nAllen CCF Atlas:")
            print(f"  Shape (Z, Y, X): {self.atlas_image.shape}")
            print(f"  Resolution: {self.atlas.resolution} μm")
            print(f"  Physical size: {np.array(self.atlas_image.shape) * self.atlas.resolution[0] / 1000} mm")
            print(f"  Data type: {self.atlas_image.dtype}")
            print(f"  Value range: [{self.atlas_image.min()}, {self.atlas_image.max()}]")
        else:
            print("\nAllen CCF Atlas: Not loaded")
            
        print("="*60 + "\n")
    
    def visualize_slices(self, slice_index=None):
        """
        Visualize middle slices of microCT and atlas
        
        Parameters:
        -----------
        slice_index : int, optional
            Specific slice index to show. If None, shows middle slice
        """
        if self.microct_image is None or self.atlas_image is None:
            print("Please load both microCT and atlas first!")
            return
        
        # Use middle slice if not specified
        if slice_index is None:
            microct_idx = self.microct_image.shape[0] // 2
            atlas_idx = self.atlas_image.shape[0] // 2
        else:
            microct_idx = slice_index
            atlas_idx = slice_index
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('MicroCT vs Allen CCF - Slice Visualization', fontsize=16, fontweight='bold')
        
        # MicroCT slices
        axes[0, 0].imshow(self.microct_image[microct_idx, :, :], cmap='gray')
        axes[0, 0].set_title(f'MicroCT - Coronal (slice {microct_idx}/{self.microct_image.shape[0]})')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(self.microct_image[:, self.microct_image.shape[1]//2, :], cmap='gray')
        axes[0, 1].set_title(f'MicroCT - Sagittal')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(self.microct_image[:, :, self.microct_image.shape[2]//2], cmap='gray')
        axes[0, 2].set_title(f'MicroCT - Axial')
        axes[0, 2].axis('off')
        
        # Atlas slices
        axes[1, 0].imshow(self.atlas_image[atlas_idx, :, :], cmap='gray')
        axes[1, 0].set_title(f'Allen CCF - Coronal (slice {atlas_idx}/{self.atlas_image.shape[0]})')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(self.atlas_image[:, self.atlas_image.shape[1]//2, :], cmap='gray')
        axes[1, 1].set_title(f'Allen CCF - Sagittal')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(self.atlas_image[:, :, self.atlas_image.shape[2]//2], cmap='gray')
        axes[1, 2].set_title(f'Allen CCF - Axial')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.project_path / "outputs" / "01_data_overview.png"
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {output_path}")
        
        plt.show()


# Example usage
if __name__ == "__main__":
    # Set your project path
    PROJECT_PATH = "/Users/jojop/Downloads/stuber-lab/uCT2CCF"
    
    # Initialize loader
    loader = DataLoader(PROJECT_PATH)
    
    # Load microCT image (will auto-detect first .tif file)
    loader.load_microct()
    
    # Load Allen CCF atlas (25μm resolution is a good balance)
    # Available resolutions: 10, 25, or 50 μm
    loader.load_allen_ccf(resolution=25)
    
    # Print summary information
    loader.get_info()
    
    # Visualize slices
    loader.visualize_slices()