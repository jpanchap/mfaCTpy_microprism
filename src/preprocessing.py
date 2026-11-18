"""
uCT2CCF - Preprocessing Module
Handles intensity normalization, resampling, and skull stripping
THIS IS NOT USEFUL!!
"""

import numpy as np
import SimpleITK as sitk
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import filters, morphology, exposure, measure


class Preprocessor:
    """Preprocessing pipeline for microCT images"""

    def __init__(self, project_path):
        """
        Initialize preprocessor

        Parameters:
        -----------
        project_path : str or Path
            Path to the uCT2CCF project folder
        """
        self.project_path = Path(project_path)
        self.output_path = self.project_path / "outputs"
        self.output_path.mkdir(exist_ok=True)

    def numpy_to_sitk(self, image_array, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0)):
        """
        Convert numpy array to SimpleITK image with metadata

        Parameters:
        -----------
        image_array : numpy.ndarray
            3D image array
        spacing : tuple
            Voxel spacing in (x, y, z) order, in mm
        origin : tuple
            Image origin in (x, y, z) order

        Returns:
        --------
        sitk.Image : SimpleITK image with metadata
        """
        # SimpleITK expects (x, y, z) ordering
        image_sitk = sitk.GetImageFromArray(image_array)
        image_sitk.SetSpacing(spacing)
        image_sitk.SetOrigin(origin)
        return image_sitk

    def sitk_to_numpy(self, image_sitk):
        """Convert SimpleITK image to numpy array"""
        return sitk.GetArrayFromImage(image_sitk)

    def resample_image(self, image_array, original_spacing, target_spacing,
                      interpolation='linear'):
        """
        Resample image to target spacing

        Parameters:
        -----------
        image_array : numpy.ndarray
            Input 3D image
        original_spacing : tuple or float
            Original voxel spacing in micrometers (z, y, x) or single value for isotropic
        target_spacing : tuple or float
            Target voxel spacing in micrometers (z, y, x) or single value for isotropic
        interpolation : str
            'linear' or 'nearest'

        Returns:
        --------
        numpy.ndarray : Resampled image
        """
        print(f"\nResampling image...")
        print(f"  Original spacing: {original_spacing} μm")
        print(f"  Target spacing: {target_spacing} μm")
        print(f"  Original shape: {image_array.shape}")

        # Convert to mm for SimpleITK (expects mm)
        if isinstance(original_spacing, (int, float)):
            original_spacing = (original_spacing, original_spacing, original_spacing)
        if isinstance(target_spacing, (int, float)):
            target_spacing = (target_spacing, target_spacing, target_spacing)

        original_spacing_mm = tuple(s / 1000.0 for s in original_spacing)
        target_spacing_mm = tuple(s / 1000.0 for s in target_spacing)

        # Convert to SimpleITK image with proper spacing
        # Note: SimpleITK spacing is in (x, y, z) order
        image_sitk = self.numpy_to_sitk(
            image_array,
            spacing=(original_spacing_mm[2], original_spacing_mm[1], original_spacing_mm[0])
        )

        # Calculate new size
        original_size = image_sitk.GetSize()
        original_spacing_sitk = image_sitk.GetSpacing()
        new_size = [
            int(round(original_size[0] * (original_spacing_sitk[0] / target_spacing_mm[2]))),
            int(round(original_size[1] * (original_spacing_sitk[1] / target_spacing_mm[1]))),
            int(round(original_size[2] * (original_spacing_sitk[2] / target_spacing_mm[0])))
        ]

        # Set up resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing((target_spacing_mm[2], target_spacing_mm[1], target_spacing_mm[0]))
        resampler.SetOutputOrigin(image_sitk.GetOrigin())
        resampler.SetOutputDirection(image_sitk.GetDirection())
        resampler.SetTransform(sitk.Transform())

        if interpolation == 'linear':
            resampler.SetInterpolator(sitk.sitkLinear)
        elif interpolation == 'nearest':
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resampler.SetInterpolator(sitk.sitkBSpline)

        # Resample
        resampled_sitk = resampler.Execute(image_sitk)
        resampled_array = self.sitk_to_numpy(resampled_sitk)

        print(f"  New shape: {resampled_array.shape}")
        print(f"✓ Resampling complete")

        return resampled_array

    def normalize_intensity(self, image_array, method='percentile',
                           lower_percentile=1, upper_percentile=99):
        """
        Normalize image intensities

        Parameters:
        -----------
        image_array : numpy.ndarray
            Input image
        method : str
            'percentile', 'minmax', or 'zscore'
        lower_percentile : float
            Lower percentile for clipping (for percentile method)
        upper_percentile : float
            Upper percentile for clipping (for percentile method)

        Returns:
        --------
        numpy.ndarray : Normalized image (float32, range 0-1)
        """
        print(f"\nNormalizing intensities using {method} method...")

        image_float = image_array.astype(np.float32)

        if method == 'percentile':
            p_low = np.percentile(image_float, lower_percentile)
            p_high = np.percentile(image_float, upper_percentile)
            print(f"  Percentile range: [{p_low:.2f}, {p_high:.2f}]")
            image_norm = np.clip(image_float, p_low, p_high)
            image_norm = (image_norm - p_low) / (p_high - p_low)

        elif method == 'minmax':
            min_val = image_float.min()
            max_val = image_float.max()
            print(f"  Min-Max range: [{min_val:.2f}, {max_val:.2f}]")
            image_norm = (image_float - min_val) / (max_val - min_val)

        elif method == 'zscore':
            mean_val = image_float.mean()
            std_val = image_float.std()
            print(f"  Mean: {mean_val:.2f}, Std: {std_val:.2f}")
            image_norm = (image_float - mean_val) / std_val
            # Clip to reasonable range and rescale to 0-1
            image_norm = np.clip(image_norm, -3, 3)
            image_norm = (image_norm + 3) / 6

        print(f"✓ Intensity normalization complete")
        print(f"  Output range: [{image_norm.min():.4f}, {image_norm.max():.4f}]")

        return image_norm

    def denoise(self, image_array, method='gaussian', sigma=1.0):
        """
        Apply denoising filter

        Parameters:
        -----------
        image_array : numpy.ndarray
            Input image
        method : str
            'gaussian', 'median', or 'bilateral'
        sigma : float
            Standard deviation for Gaussian filter

        Returns:
        --------
        numpy.ndarray : Denoised image
        """
        print(f"\nApplying {method} denoising...")

        if method == 'gaussian':
            denoised = ndimage.gaussian_filter(image_array, sigma=sigma)
        elif method == 'median':
            denoised = ndimage.median_filter(image_array, size=3)
        elif method == 'bilateral':
            # Convert to SimpleITK for bilateral filter
            image_sitk = sitk.GetImageFromArray(image_array.astype(np.float32))
            denoised_sitk = sitk.Bilateral(image_sitk, domainSigma=sigma, rangeSigma=0.1)
            denoised = sitk.GetArrayFromImage(denoised_sitk)
        else:
            print(f"  Unknown method '{method}', skipping denoising")
            return image_array

        print(f"✓ Denoising complete")
        return denoised

    def simple_skull_strip(self, image_array, threshold_method='otsu',
                          morphology_iterations=2):
        """
        Simple skull stripping using thresholding and morphology

        Parameters:
        -----------
        image_array : numpy.ndarray
            Input image (normalized 0-1)
        threshold_method : str
            'otsu' or 'manual'
        morphology_iterations : int
            Number of morphological operations

        Returns:
        --------
        tuple : (brain_only_image, brain_mask)
        """
        print(f"\nPerforming skull stripping...")
        print(f"  Using {threshold_method} thresholding")

        # Threshold to create initial mask
        if threshold_method == 'otsu':
            threshold = filters.threshold_otsu(image_array)
            print(f"  Otsu threshold: {threshold:.4f}")
        else:
            # Manual threshold (adjust as needed)
            threshold = 0.3
            print(f"  Manual threshold: {threshold:.4f}")

        mask = image_array > threshold

        # Morphological operations to clean up mask
        print(f"  Applying morphological operations...")

        # Remove small objects
        mask = morphology.remove_small_objects(mask, min_size=1000)

        # Fill holes
        mask = ndimage.binary_fill_holes(mask)

        # Closing operation to smooth boundaries
        for _ in range(morphology_iterations):
            mask = morphology.binary_closing(mask, morphology.ball(3))

        # Opening operation to separate connected components
        mask = morphology.binary_opening(mask, morphology.ball(2))

        # Keep only the largest connected component (should be the brain)
        labeled_mask = morphology.label(mask)
        regions = measure.regionprops(labeled_mask)

        if len(regions) > 0:
            # Find largest region
            largest_region = max(regions, key=lambda r: r.area)
            mask = labeled_mask == largest_region.label
            print(f"  Largest component volume: {largest_region.area:,} voxels")

        # Apply mask
        brain_only = image_array.copy()
        brain_only[~mask] = 0

        print(f"✓ Skull stripping complete")
        print(f"  Brain voxels: {np.sum(mask):,} ({100*np.sum(mask)/mask.size:.2f}% of volume)")

        return brain_only, mask.astype(np.uint8)

    def visualize_preprocessing(self, original, processed, mask=None,
                               slice_idx=None, save_name='preprocessing_result'):
        """
        Visualize preprocessing results

        Parameters:
        -----------
        original : numpy.ndarray
            Original image
        processed : numpy.ndarray
            Processed image
        mask : numpy.ndarray, optional
            Brain mask
        slice_idx : int, optional
            Slice index to visualize
        save_name : str
            Filename for saving
        """
        if slice_idx is None:
            slice_idx = original.shape[0] // 2

        n_cols = 3 if mask is not None else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))

        # Original
        axes[0].imshow(original[slice_idx], cmap='gray')
        axes[0].set_title('Original microCT')
        axes[0].axis('off')

        # Processed
        axes[1].imshow(processed[slice_idx], cmap='gray')
        axes[1].set_title('Preprocessed')
        axes[1].axis('off')

        # Mask overlay
        if mask is not None:
            axes[2].imshow(original[slice_idx], cmap='gray', alpha=0.7)
            axes[2].imshow(mask[slice_idx], cmap='Reds', alpha=0.3)
            axes[2].set_title('Brain Mask Overlay')
            axes[2].axis('off')

        plt.tight_layout()

        save_path = self.output_path / f"{save_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {save_path}")
        plt.show()

    def preprocess_pipeline(self, image_array, original_spacing=20,
                           target_spacing=25, denoise_sigma=1.0,
                           normalize_method='percentile',
                           perform_skull_strip=True):
        """
        Complete preprocessing pipeline

        Parameters:
        -----------
        image_array : numpy.ndarray
            Input microCT image
        original_spacing : float
            Original voxel spacing in micrometers
        target_spacing : float
            Target voxel spacing in micrometers
        denoise_sigma : float
            Sigma for Gaussian denoising
        normalize_method : str
            Normalization method
        perform_skull_strip : bool
            Whether to perform skull stripping

        Returns:
        --------
        dict : Dictionary containing processed images and metadata
        """
        print("\n" + "="*60)
        print("PREPROCESSING PIPELINE")
        print("="*60)

        results = {
            'original': image_array,
            'original_spacing': original_spacing,
            'target_spacing': target_spacing
        }

        # Step 1: Resample
        resampled = self.resample_image(image_array, original_spacing, target_spacing)
        results['resampled'] = resampled

        # Step 2: Denoise
        denoised = self.denoise(resampled, method='gaussian', sigma=denoise_sigma)
        results['denoised'] = denoised

        # Step 3: Normalize intensity
        normalized = self.normalize_intensity(denoised, method=normalize_method)
        results['normalized'] = normalized

        # Step 4: Skull stripping (optional)
        if perform_skull_strip:
            brain_only, mask = self.simple_skull_strip(normalized)
            results['brain_only'] = brain_only
            results['mask'] = mask
            final_image = brain_only
        else:
            final_image = normalized
            results['brain_only'] = None
            results['mask'] = None

        results['final'] = final_image

        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        print(f"Final image shape: {final_image.shape}")
        print(f"Final image range: [{final_image.min():.4f}, {final_image.max():.4f}]")

        # Visualize
        self.visualize_preprocessing(
            resampled,
            final_image,
            mask=results['mask'],
            save_name='02_preprocessing_result'
        )

        return results


# Example usage
if __name__ == "__main__":
    from data_loader import DataLoader

    import argparse
    from project_paths import resolve_project_path

    parser = argparse.ArgumentParser(description="Preprocess a microCT scan.")
    parser.add_argument("project_path", nargs="?", help="Project containing data/.")
    parser.add_argument("--original-spacing", type=float, required=True,
                        help="Verified input voxel spacing in micrometers (isotropic).")
    parser.add_argument("--target-spacing", type=float, default=25,
                        help="Output voxel spacing in micrometers (default: 25).")
    args = parser.parse_args()
    if not all(np.isfinite(v) and v > 0 for v in (args.original_spacing, args.target_spacing)):
        parser.error("Voxel spacings must be finite and positive.")
    PROJECT_PATH = resolve_project_path(args.project_path)

    # Load data
    print("Loading data...")
    loader = DataLoader(PROJECT_PATH)
    microct = loader.load_microct()

    # Preprocess
    preprocessor = Preprocessor(PROJECT_PATH)
    results = preprocessor.preprocess_pipeline(
        microct,
        original_spacing=args.original_spacing,
        target_spacing=args.target_spacing,
        denoise_sigma=1.0,
        normalize_method='percentile',
        perform_skull_strip=True
    )

    # Save preprocessed image for next step
    preprocessed_path = PROJECT_PATH / "data" / "processed"
    preprocessed_path.mkdir(exist_ok=True)

    import tifffile
    output_file = preprocessed_path / "microct_preprocessed.tif"
    tifffile.imwrite(output_file, (results['final'] * 65535).astype(np.uint16))
    print(f"\n✓ Preprocessed image saved to: {output_file}")
