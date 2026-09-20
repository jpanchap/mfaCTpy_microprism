"""
uCT2CCF - Midline Alignment Module with Axis Verification
Interactive tool for marking midline and correcting image orientation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox
from scipy import ndimage
from scipy.spatial.transform import Rotation
import SimpleITK as sitk
from pathlib import Path
import json
import tifffile
import time
import shutil
from project_paths import resolve_project_path


def copy_metadata_sidecar(source_image: Path, output_image: Path) -> None:
    """Carry image spacing metadata forward when saving derived TIFFs."""
    source_sidecar = source_image.with_suffix(source_image.suffix + ".metadata.json")
    output_sidecar = output_image.with_suffix(output_image.suffix + ".metadata.json")
    if source_sidecar.exists():
        shutil.copy2(source_sidecar, output_sidecar)
        print(f"✓ Metadata sidecar saved to: {output_sidecar}")


class AxisVerifier:
    """Interactive tool for verifying and correcting axis orientation"""
    
    def __init__(self, image, name="MicroCT"):
        """
        Initialize axis verifier
        
        Parameters:
        -----------
        image : numpy.ndarray
            3D image array (Z, Y, X)
        name : str
            Name for the image
        """
        self.image = image
        self.name = name
        self.corrected_image = image.copy()
        
    def verify_and_correct_axes(self):
        """
        Interactive interface to verify and correct axis orientation
        
        Returns:
        --------
        corrected_image : numpy.ndarray
            Image with corrected axis orientation
        """
        print("\n" + "="*70)
        print("AXIS ORIENTATION VERIFICATION")
        print("="*70)
        print("\nThis step helps you verify that the axes are correctly labeled.")
        print("\nExpected orientation after midline alignment:")
        print("  - CORONAL (front-back): Should show brain from front/back")
        print("  - AXIAL (top-down): Should show brain from top, midline vertical")
        print("  - SAGITTAL (side): Should show brain from the side")
        print("\nIf the views don't match these descriptions, use the buttons to:")
        print("  - Swap axes (e.g., swap Z↔Y if coronal/axial are switched)")
        print("  - Flip axes (mirror the image along an axis)")
        print("  - Rotate a displayed view by a precise angle in degrees")
        print("    Positive values rotate clockwise; negative values rotate counter-clockwise")
        print("="*70 + "\n")
        
        fig = plt.figure(figsize=(18, 13))
        
        # Create subplots for three views
        ax_coronal = plt.axes([0.055, 0.58, 0.25, 0.30])
        ax_axial = plt.axes([0.375, 0.58, 0.25, 0.30])
        ax_sagittal = plt.axes([0.695, 0.58, 0.25, 0.30])
        
        # Create slider axes
        ax_slider_cor = plt.axes([0.10, 0.505, 0.20, 0.02])
        ax_slider_ax = plt.axes([0.40, 0.505, 0.20, 0.02])
        ax_slider_sag = plt.axes([0.70, 0.505, 0.20, 0.02])
        
        # Current slice indices
        slices = {
            'coronal': self.corrected_image.shape[0] // 2,
            'axial': self.corrected_image.shape[1] // 2,
            'sagittal': self.corrected_image.shape[2] // 2
        }
        rotation_angle = {'degrees': 0.0}
        rotation_preview_view = {'view': None}

        def preview_slice(slice_2d, view_name):
            angle = rotation_angle['degrees']
            if rotation_preview_view['view'] != view_name or abs(angle) < 1e-6:
                return slice_2d
            return ndimage.rotate(
                slice_2d,
                -angle,
                reshape=False,
                order=1,
                mode='constant',
                cval=float(np.min(slice_2d)),
                prefilter=False,
            )
        
        def update_display():
            """Update all three views"""
            # Coronal (through Z axis)
            ax_coronal.clear()
            cor_slice = self.corrected_image[slices['coronal'], :, :]
            cor_display = preview_slice(cor_slice, 'coronal')
            ax_coronal.imshow(cor_display, cmap='gray')
            ax_coronal.axvline(x=cor_display.shape[1]//2, color='green', 
                             linestyle='--', linewidth=1, alpha=0.5)
            ax_coronal.set_title(f'CORONAL (Z={slices["coronal"]})\n'
                               f'Shape: {cor_display.shape} | Front/back | midline vertical',
                               fontsize=9, pad=5)
            ax_coronal.axis('off')
            
            # Axial (through Y axis)
            ax_axial.clear()
            ax_slice = self.corrected_image[:, slices['axial'], :]
            ax_display = preview_slice(ax_slice, 'axial')
            ax_axial.imshow(ax_display, cmap='gray')
            ax_axial.axvline(x=ax_display.shape[1]//2, color='green', 
                           linestyle='--', linewidth=1, alpha=0.5)
            ax_axial.set_title(f'AXIAL (Y={slices["axial"]})\n'
                             f'Shape: {ax_display.shape} | Top-down | midline vertical',
                             fontsize=9, pad=5)
            ax_axial.axis('off')
            
            # Sagittal (through X axis)
            ax_sagittal.clear()
            sag_slice = self.corrected_image[:, :, slices['sagittal']]
            sag_display = preview_slice(sag_slice, 'sagittal')
            ax_sagittal.imshow(sag_display, cmap='gray')
            ax_sagittal.set_title(f'SAGITTAL (X={slices["sagittal"]})\n'
                                f'Shape: {sag_display.shape} | Side view',
                                fontsize=9, pad=5)
            ax_sagittal.axis('off')
            
            # Update info text
            info_text = (
                f"Current image shape: {self.corrected_image.shape} (Z, Y, X)\n"
                f"Preview view: {rotation_preview_view['view'] or 'none'} | "
                f"Angle: {rotation_angle['degrees']:.2f}°"
            )
            fig.text(0.5, 0.455, info_text, ha='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            apply_zoom(ax_coronal, cor_display.shape)
            apply_zoom(ax_axial, ax_display.shape)
            apply_zoom(ax_sagittal, sag_display.shape)

            fig.canvas.draw()
        
        # Create sliders
        slider_cor = Slider(ax_slider_cor, 'Coronal', 0, 
                          self.corrected_image.shape[0]-1, 
                          valinit=slices['coronal'], valstep=1)
        slider_ax = Slider(ax_slider_ax, 'Axial', 0, 
                         self.corrected_image.shape[1]-1, 
                         valinit=slices['axial'], valstep=1)
        slider_sag = Slider(ax_slider_sag, 'Sagittal', 0, 
                          self.corrected_image.shape[2]-1, 
                          valinit=slices['sagittal'], valstep=1)
        
        def update_coronal(val):
            slices['coronal'] = int(val)
            update_display()
        
        def update_axial(val):
            slices['axial'] = int(val)
            update_display()
        
        def update_sagittal(val):
            slices['sagittal'] = int(val)
            update_display()
        
        slider_cor.on_changed(update_coronal)
        slider_ax.on_changed(update_axial)
        slider_sag.on_changed(update_sagittal)
        
        # Create correction buttons
        button_height = 0.034
        button_width = 0.098
        button_y_start = 0.405
        button_spacing = 0.012
        
        # Swap buttons
        ax_swap_zy = plt.axes([0.1, button_y_start, button_width, button_height])
        ax_swap_zx = plt.axes([0.1 + button_width + button_spacing, button_y_start, 
                              button_width, button_height])
        ax_swap_yx = plt.axes([0.1 + 2*(button_width + button_spacing), button_y_start, 
                              button_width, button_height])
        
        btn_swap_zy = Button(ax_swap_zy, 'Swap Z↔Y')
        btn_swap_zx = Button(ax_swap_zx, 'Swap Z↔X')
        btn_swap_yx = Button(ax_swap_yx, 'Swap Y↔X')
        
        # Flip buttons
        button_y_flip = button_y_start - button_height - button_spacing
        ax_flip_z = plt.axes([0.1, button_y_flip, button_width, button_height])
        ax_flip_y = plt.axes([0.1 + button_width + button_spacing, button_y_flip, 
                             button_width, button_height])
        ax_flip_x = plt.axes([0.1 + 2*(button_width + button_spacing), button_y_flip, 
                             button_width, button_height])
        
        btn_flip_z = Button(ax_flip_z, 'Flip Z')
        btn_flip_y = Button(ax_flip_y, 'Flip Y')
        btn_flip_x = Button(ax_flip_x, 'Flip X')
        
        # Fine rotation controls
        button_y_rotate = button_y_flip - button_height - button_spacing
        ax_rotate_cor = plt.axes([0.1, button_y_rotate, button_width, button_height])
        ax_rotate_ax = plt.axes([0.1 + button_width + button_spacing, button_y_rotate,
                                 button_width, button_height])
        ax_rotate_sag = plt.axes([0.1 + 2*(button_width + button_spacing), button_y_rotate,
                                  button_width, button_height])
        ax_angle_slider = plt.axes([0.48, button_y_rotate + 0.009, 0.18, 0.018])
        ax_angle_text = plt.axes([0.70, button_y_rotate, 0.07, button_height])

        btn_rotate_cor = Button(ax_rotate_cor, 'Rotate Coronal')
        btn_rotate_ax = Button(ax_rotate_ax, 'Rotate Axial')
        btn_rotate_sag = Button(ax_rotate_sag, 'Rotate Sagittal')
        angle_slider = Slider(
            ax_angle_slider, 'Angle°', -30.0, 30.0, valinit=0.0, valstep=0.1)
        angle_text = TextBox(ax_angle_text, '', initial='0.0')
        fig.text(
            0.78, button_y_rotate + 0.015,
            'Positive = clockwise, negative = counter-clockwise',
            ha='left', va='center', fontsize=8)

        # Preview zoom controls
        button_y_zoom = button_y_rotate - button_height - button_spacing
        ax_zoom_in = plt.axes([0.1, button_y_zoom, button_width, button_height])
        ax_zoom_out = plt.axes([0.1 + button_width + button_spacing, button_y_zoom,
                                button_width, button_height])
        ax_zoom_reset = plt.axes([0.1 + 2*(button_width + button_spacing), button_y_zoom,
                                  button_width, button_height])

        btn_zoom_in = Button(ax_zoom_in, 'Zoom +')
        btn_zoom_out = Button(ax_zoom_out, 'Zoom -')
        btn_zoom_reset = Button(ax_zoom_reset, 'Zoom 1x')

        # Control buttons
        button_y_control = button_y_zoom - button_height - button_spacing
        ax_reset = plt.axes([0.1, button_y_control, button_width, button_height])
        ax_done = plt.axes([0.8, button_y_control, button_width, button_height])
        
        btn_reset = Button(ax_reset, 'Reset')
        btn_done = Button(ax_done, 'Done')
        
        zoom_state = {'factor': 1.0}

        def apply_zoom(axis, image_shape):
            zoom = zoom_state['factor']
            if zoom <= 1.0:
                return

            rows, cols = image_shape
            center_x = (cols - 1) / 2.0
            center_y = (rows - 1) / 2.0
            half_width = cols / (2.0 * zoom)
            half_height = rows / (2.0 * zoom)
            axis.set_xlim(center_x - half_width, center_x + half_width)
            axis.set_ylim(center_y + half_height, center_y - half_height)

        def change_zoom(multiplier):
            zoom_state['factor'] = float(np.clip(
                zoom_state['factor'] * multiplier, 1.0, 16.0))
            print(f"\nPreview zoom: {zoom_state['factor']:.2f}x")
            update_display()

        def reset_zoom(event):
            zoom_state['factor'] = 1.0
            print("\nPreview zoom reset to 1.00x")
            update_display()

        # Transpose operations
        def swap_axes(axis1, axis2):
            print(f"\nSwapping axes {axis1} ↔ {axis2}")
            print(f"  Before: {self.corrected_image.shape}")
            
            axes = [0, 1, 2]
            axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
            self.corrected_image = np.transpose(self.corrected_image, axes)
            
            # Update slider ranges
            slider_cor.valmin = 0
            slider_cor.valmax = self.corrected_image.shape[0] - 1
            slider_cor.set_val(self.corrected_image.shape[0] // 2)
            
            slider_ax.valmin = 0
            slider_ax.valmax = self.corrected_image.shape[1] - 1
            slider_ax.set_val(self.corrected_image.shape[1] // 2)
            
            slider_sag.valmin = 0
            slider_sag.valmax = self.corrected_image.shape[2] - 1
            slider_sag.set_val(self.corrected_image.shape[2] // 2)
            
            slices['coronal'] = self.corrected_image.shape[0] // 2
            slices['axial'] = self.corrected_image.shape[1] // 2
            slices['sagittal'] = self.corrected_image.shape[2] // 2
            
            print(f"  After: {self.corrected_image.shape}")
            update_display()
        
        def flip_axis(axis):
            print(f"\nFlipping axis {axis}")
            self.corrected_image = np.flip(self.corrected_image, axis=axis)
            update_display()

        def set_rotation_angle(angle):
            angle = float(np.clip(angle, -30.0, 30.0))
            rotation_angle['degrees'] = angle
            return angle

        def update_angle_from_slider(val):
            set_rotation_angle(val)
            update_display()

        def update_angle_from_text(text):
            try:
                angle = set_rotation_angle(float(text))
            except ValueError:
                print(f"Could not parse rotation angle: {text!r}")
                return
            angle_slider.set_val(angle)
            update_display()

        def rotate_view(view_name):
            angle = rotation_angle['degrees']
            rotation_preview_view['view'] = view_name
            if abs(angle) < 1e-6:
                print(f"\nSelected {view_name} rotation preview. Enter a nonzero angle to preview.")
                update_display()
                return

            print(f"\nPreviewing {view_name} view at {angle:.3f}°")
            print("  Positive angles are clockwise in the displayed view.")
            print("  This is display-only and will not change saved image geometry.")
            update_display()
        
        def reset_image(event):
            print("\nResetting to original image")
            self.corrected_image = self.image.copy()
            set_rotation_angle(0.0)
            angle_slider.set_val(0.0)
            angle_text.set_val('0.0')
            zoom_state['factor'] = 1.0
            rotation_preview_view['view'] = None
            
            # Reset sliders
            slider_cor.valmin = 0
            slider_cor.valmax = self.corrected_image.shape[0] - 1
            slider_cor.set_val(self.corrected_image.shape[0] // 2)
            
            slider_ax.valmin = 0
            slider_ax.valmax = self.corrected_image.shape[1] - 1
            slider_ax.set_val(self.corrected_image.shape[1] // 2)
            
            slider_sag.valmin = 0
            slider_sag.valmax = self.corrected_image.shape[2] - 1
            slider_sag.set_val(self.corrected_image.shape[2] // 2)
            
            slices['coronal'] = self.corrected_image.shape[0] // 2
            slices['axial'] = self.corrected_image.shape[1] // 2
            slices['sagittal'] = self.corrected_image.shape[2] // 2
            
            update_display()
        
        def done(event):
            plt.close(fig)
        
        # Connect buttons
        btn_swap_zy.on_clicked(lambda e: swap_axes(0, 1))
        btn_swap_zx.on_clicked(lambda e: swap_axes(0, 2))
        btn_swap_yx.on_clicked(lambda e: swap_axes(1, 2))
        
        btn_flip_z.on_clicked(lambda e: flip_axis(0))
        btn_flip_y.on_clicked(lambda e: flip_axis(1))
        btn_flip_x.on_clicked(lambda e: flip_axis(2))
        btn_rotate_cor.on_clicked(lambda e: rotate_view('coronal'))
        btn_rotate_ax.on_clicked(lambda e: rotate_view('axial'))
        btn_rotate_sag.on_clicked(lambda e: rotate_view('sagittal'))
        angle_slider.on_changed(update_angle_from_slider)
        angle_text.on_submit(update_angle_from_text)
        btn_zoom_in.on_clicked(lambda e: change_zoom(1.25))
        btn_zoom_out.on_clicked(lambda e: change_zoom(0.8))
        btn_zoom_reset.on_clicked(reset_zoom)
        
        btn_reset.on_clicked(reset_image)
        btn_done.on_clicked(done)
        
        # Add instructions
        instructions = (
            "Enter degrees, then choose Rotate Coronal / Axial / Sagittal. "
            "Positive = clockwise, negative = counter-clockwise.\n"
            "Rotation is a selected-view preview only; saved volume geometry is unchanged."
        )
        fig.text(0.52, 0.055, instructions, ha='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Initial display
        update_display()
        
        plt.show()
        
        print("\n" + "="*70)
        print("AXIS VERIFICATION COMPLETE")
        print(f"Final image shape: {self.corrected_image.shape}")
        print("="*70)
        
        return self.corrected_image


class MidlineAligner:
    """Interactive midline marking and alignment tool"""
    
    def __init__(self, image, name="MicroCT"):
        """
        Initialize midline aligner
        
        Parameters:
        -----------
        image : numpy.ndarray
            3D image array (Z, Y, X)
        name : str
            Name for the image
        """
        self.image = image
        self.name = name
        
        # Store midline points: dict with (view, slice_index) -> list of (row, col) points
        self.midline_points = {}
        
        # Current view and slice
        self.current_view = None
        self.current_slice = None
        
        # Figure elements
        self.fig = None
        self.ax = None
        self.slider = None
    
    def get_slice_and_info(self, view, slice_idx):
        """Get 2D slice for a given view and provide axis information
        
        NOTE: AXIAL and CORONAL have been swapped compared to the original file
        to match the data orientation requested by the user.
        """
        
        # Image shape convention: (Z, Y, X)
        if view == 'axial':
            # --- CHANGED ---
            # Treat "axial" as slicing through Y (this was previously labeled coronal).
            # Displayed 2D slice shape: (Z, X) -> rows are Z (dorsal-ventral), cols are X (L-R)
            slice_2d = self.image[:, slice_idx, :]
            info = {
                'slice_axis': 'Y (slice index 1)',      # slicing through Y
                'horizontal': 'X (Left-Right)',
                'vertical': 'Z (Dorsal-Ventral)',
                'max_slice': self.image.shape[1] - 1,
            }
        elif view == 'coronal':
            # --- CHANGED ---
            # Treat "coronal" as slicing through Z (this was previously labeled axial).
            # Displayed 2D slice shape: (Y, X) -> rows are Y (A-P), cols are X (L-R)
            slice_2d = self.image[slice_idx, :, :]
            info = {
                'slice_axis': 'Z (slice index 0)',      # slicing through Z
                'horizontal': 'X (Left-Right)',
                'vertical': 'Y (Anterior-Posterior)',
                'max_slice': self.image.shape[0] - 1,
            }
        elif view == 'sagittal':
            # Sagittal: side slices through X
            slice_2d = self.image[:, :, slice_idx]
            info = {
                'slice_axis': 'X (slice index 2)',
                'horizontal': 'Y (Anterior-Posterior)',
                'vertical': 'Z (Dorsal-Ventral)',
                'max_slice': self.image.shape[2] - 1,
            }
        else:
            raise ValueError(f"Unknown view: {view}")
        
        return slice_2d, info
    
    def onclick(self, event):
        """Handle mouse clicks for midline point selection"""
        if event.inaxes != self.ax or self.current_view is None:
            return
        
        if event.button == 1:  # Left click - add point
            col, row = int(event.xdata), int(event.ydata)
            key = (self.current_view, self.current_slice)
            
            if key not in self.midline_points:
                self.midline_points[key] = []
            
            self.midline_points[key].append((row, col))
            
            print(f"Added midline point at {self.current_view} slice {self.current_slice}: (row={row}, col={col})")
            print(f"  Total points on this slice: {len(self.midline_points[key])}")
            
            self.update_display()
    
    def update_display(self):
        """Update the display with current slice and midline points"""
        if self.current_view is None:
            return
            
        self.ax.clear()
        
        slice_2d, info = self.get_slice_and_info(self.current_view, self.current_slice)
        
        self.ax.imshow(slice_2d, cmap='gray')
        
        title = (f'{self.name} - {self.current_view.capitalize()} View\n'
                f'Slice {self.current_slice}/{info["max_slice"]} along {info["slice_axis"]}\n'
                f'Horizontal: {info["horizontal"]} | Vertical: {info["vertical"]}\n'
                f'Slices with points: {len(self.midline_points)}')
        self.ax.set_title(title, fontsize=10)
        self.ax.axis('off')
        
        # Draw midline points
        key = (self.current_view, self.current_slice)
        if key in self.midline_points:
            points = np.array(self.midline_points[key])
            self.ax.plot(points[:, 1], points[:, 0], 'r+', markersize=15, markeredgewidth=2)
            
            if len(points) > 1:
                self.ax.plot(points[:, 1], points[:, 0], 'r-', linewidth=2, alpha=0.7)
            
            for idx, (row, col) in enumerate(points, 1):
                self.ax.text(col+5, row-5, str(idx), color='red', 
                           fontsize=10, fontweight='bold')
        
        # Show indicator of other slices with points
        y_offset = 30
        for other_key in self.midline_points.keys():
            if other_key != key:
                view, slice_idx = other_key
                self.ax.text(10, y_offset,
                           f'{view.capitalize()} {slice_idx}: {len(self.midline_points[other_key])} pts',
                           color='yellow', fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
                y_offset += 20
        
        self.fig.canvas.draw()
    
    def _select_view_plane(self):
        """Interactive view plane selection"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Select the HORIZONTAL (AXIAL) section for midline marking\n(Click on the view that shows horizontal/top-down slices)',
                    fontsize=14, fontweight='bold')
        
        mid_z = self.image.shape[0] // 2
        mid_y = self.image.shape[1] // 2
        mid_x = self.image.shape[2] // 2
        
        # Show three orthogonal views with simple axis labels
        # View 1: Slicing through Y axis
        axes[0].imshow(self.image[:, mid_y, :], cmap='gray')
        axes[0].set_title('Slice through Y axis\n(Axis index 1)', fontsize=12)
        axes[0].axis('off')
        
        # View 2: Slicing through Z axis
        axes[1].imshow(self.image[mid_z, :, :], cmap='gray')
        axes[1].set_title('Slice through Z axis\n(Axis index 0)', fontsize=12)
        axes[1].axis('off')
        
        # View 3: Slicing through X axis
        axes[2].imshow(self.image[:, :, mid_x], cmap='gray')
        axes[2].set_title('Slice through X axis\n(Axis index 2)', fontsize=12)
        axes[2].axis('off')
        
        selected_view = [None]
        
        def on_click(event):
            if event.inaxes == axes[0]:
                selected_view[0] = 'axial'
                print("Selected: Slice through Y axis (will be treated as AXIAL/horizontal view)")
                plt.close(fig)
            elif event.inaxes == axes[1]:
                selected_view[0] = 'coronal'
                print("Selected: Slice through Z axis (will be treated as CORONAL view)")
                plt.close(fig)
            elif event.inaxes == axes[2]:
                selected_view[0] = 'sagittal'
                print("Selected: Slice through X axis (will be treated as SAGITTAL view)")
                plt.close(fig)
        
        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.tight_layout()
        plt.show()
        
        return selected_view[0]
    
    def mark_midline(self):
        """Launch interactive midline marking interface"""
        print("\n" + "="*70)
        print("MIDLINE ALIGNMENT TOOL - VIEW SELECTION")
        print("="*70)
        print("\nStep 1: Select the HORIZONTAL (AXIAL) section")
        print("  - Look at the three views showing slices through X, Y, and Z axes")
        print("  - Click on the view that shows horizontal/top-down brain sections")
        print("  - This is typically the best view for marking the brain midline")
        print("\nStep 2: Mark midline points")
        print("  - Navigate slices with slider or scroll wheel")
        print("  - LEFT-CLICK to mark points along the midline")
        print("  - Mark 2-3 points per slice, across 5-10 slices")
        print("  - You can switch views and mark in multiple planes if needed")
        print("\nStep 3: Finish")
        print("  - Click 'Done' when satisfied with your markings")
        print("="*70 + "\n")
        
        self.current_view = self._select_view_plane()
        if self.current_view is None:
            print("View selection cancelled")
            return {}
        
        _, info = self.get_slice_and_info(self.current_view, 0)
        self.current_slice = info['max_slice'] // 2
        
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = plt.subplot(1, 1, 1)
        self.update_display()
        
        # Add slider
        ax_slider = plt.axes([0.2, 0.12, 0.6, 0.02])
        _, info = self.get_slice_and_info(self.current_view, self.current_slice)
        self.slider = Slider(
            ax_slider,
            f'{self.current_view.capitalize()} Slice',
            0,
            info['max_slice'],
            valinit=self.current_slice,
            valstep=1
        )
        
        def update_slice(val):
            self.current_slice = int(self.slider.val)
            self.update_display()
        
        self.slider.on_changed(update_slice)
        
        # Add buttons
        ax_done = plt.axes([0.7, 0.02, 0.08, 0.04])
        ax_clear = plt.axes([0.62, 0.02, 0.08, 0.04])
        ax_clear_all = plt.axes([0.54, 0.02, 0.08, 0.04])
        ax_change_view = plt.axes([0.2, 0.02, 0.1, 0.04])
        ax_axial = plt.axes([0.31, 0.02, 0.07, 0.04])
        ax_coronal = plt.axes([0.38, 0.02, 0.07, 0.04])
        ax_sagittal = plt.axes([0.45, 0.02, 0.07, 0.04])
        
        btn_done = Button(ax_done, 'Done')
        btn_clear = Button(ax_clear, 'Clear Slice')
        btn_clear_all = Button(ax_clear_all, 'Clear All')
        btn_change_view = Button(ax_change_view, 'Change View')
        btn_axial = Button(ax_axial, 'Axial')
        btn_coronal = Button(ax_coronal, 'Coronal')
        btn_sagittal = Button(ax_sagittal, 'Sagittal')
        
        def on_done(event):
            plt.close(self.fig)
        
        def on_clear(event):
            key = (self.current_view, self.current_slice)
            if key in self.midline_points:
                del self.midline_points[key]
                print(f"Cleared points from {self.current_view} slice {self.current_slice}")
                self.update_display()
        
        def on_clear_all(event):
            self.midline_points.clear()
            print("Cleared all midline points")
            self.update_display()
        
        def change_to_view(new_view):
            self.current_view = new_view
            _, info = self.get_slice_and_info(self.current_view, 0)
            self.current_slice = info['max_slice'] // 2
            
            self.slider.valmin = 0
            self.slider.valmax = info['max_slice']
            self.slider.set_val(self.current_slice)
            self.slider.label.set_text(f'{self.current_view.capitalize()} Slice')
            
            print(f"\nSwitched to {self.current_view.capitalize()} view")
            self.update_display()
        
        def on_change_view(event):
            new_view = self._select_view_plane()
            if new_view:
                change_to_view(new_view)
        
        btn_done.on_clicked(on_done)
        btn_clear.on_clicked(on_clear)
        btn_clear_all.on_clicked(on_clear_all)
        btn_change_view.on_clicked(on_change_view)
        btn_axial.on_clicked(lambda e: change_to_view('axial'))
        btn_coronal.on_clicked(lambda e: change_to_view('coronal'))
        btn_sagittal.on_clicked(lambda e: change_to_view('sagittal'))
        
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        
        def on_scroll(event):
            if event.inaxes == self.ax:
                _, info = self.get_slice_and_info(self.current_view, self.current_slice)
                if event.button == 'up':
                    self.current_slice = min(self.current_slice + 1, info['max_slice'])
                else:
                    self.current_slice = max(self.current_slice - 1, 0)
                self.slider.set_val(self.current_slice)
                self.update_display()
        
        self.fig.canvas.mpl_connect('scroll_event', on_scroll)
        
        plt.show()
        
        total_points = sum(len(pts) for pts in self.midline_points.values())
        print(f"\n✓ Marked {total_points} midline points across {len(self.midline_points)} view/slice combinations")
        
        return self.midline_points
    
    def fit_midline_plane(self):
        """Fit a plane to the marked midline points"""
        print("\n" + "="*60)
        print("FITTING MIDLINE PLANE")
        print("="*60)
        
        if len(self.midline_points) < 2:
            raise ValueError("Need at least 2 slices with midline points")
        
        points_3d = []
        
        # Image array convention: (Z, Y, X)
        # 3D space convention: (x, y, z)
        for (view, slice_idx), points_2d in self.midline_points.items():
            for row, col in points_2d:
                if view == 'axial':
                    # --- CHANGED ---
                    # axial now = slicing through Y, displayed (Z, X)
                    # row = Z, col = X, slice_idx = Y
                    points_3d.append([col, slice_idx, row])  # (x, y, z)
                elif view == 'coronal':
                    # --- CHANGED ---
                    # coronal now = slicing through Z, displayed (Y, X)
                    # row = Y, col = X, slice_idx = Z
                    points_3d.append([col, row, slice_idx])  # (x, y, z)
                elif view == 'sagittal':
                    # Sagittal: slice through X, display is (Z, Y)
                    # row = Z, col = Y, slice_idx = X
                    points_3d.append([slice_idx, col, row])  # (x, y, z)
        
        points_3d = np.array(points_3d)
        print(f"Total midline points: {len(points_3d)}")
        print(f"Points from views: {set(view for view, _ in self.midline_points.keys())}")
        
        # Fit plane using SVD
        centroid = points_3d.mean(axis=0)
        points_centered = points_3d - centroid
        
        _, _, vh = np.linalg.svd(points_centered)
        normal = vh[-1, :]
        
        # Ensure consistent normal direction: aim to have normal point roughly along +X
        # (midline plane should be parallel to YZ plane -> normal along X)
        # Note: flipping the normal doesn't change the plane, but affects rotation direction.
        if normal[0] < 0:
            normal = -normal
        
        print(f"Plane normal vector (pre-normalize): {normal}")
        print(f"Plane centroid: {centroid}")
        
        # Calculate rotation to align with YZ plane (target normal along +X)
        target_normal = np.array([1.0, 0.0, 0.0])
        
        # Compute axis/angle
        rotation_axis = np.cross(normal, target_normal)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm > 1e-6:
            rotation_axis = rotation_axis / rotation_axis_norm
            cos_angle = np.clip(np.dot(normal, target_normal), -1.0, 1.0)
            rotation_angle = np.arccos(cos_angle)
            rotation_angle_deg = np.degrees(rotation_angle)
            
            print(f"Initial rotation axis: {rotation_axis}")
            print(f"Initial rotation angle: {rotation_angle_deg:.2f}°")
            
            # --- NEW: Verify rotation actually moves normal toward target ---
            rot_test = Rotation.from_rotvec(rotation_angle * rotation_axis)
            rotated_normal = rot_test.apply(normal)
            dot_after = np.dot(rotated_normal, target_normal)
            dot_before = np.dot(normal, target_normal)
            
            if dot_after < dot_before - 1e-8:
                # Rotation is not improving alignment -> invert direction
                rotation_axis = -rotation_axis
                rotation_angle = -rotation_angle
                rotation_angle_deg = np.degrees(rotation_angle)
                print("Detected rotation would move normal away from target -- inverting rotation direction.")
                print(f"Corrected rotation axis: {rotation_axis}")
                print(f"Corrected rotation angle: {rotation_angle_deg:.2f}°")
            
            # Warn if rotation unusually large (useful diagnostic)
            if abs(rotation_angle_deg) > 45:
                print(f"⚠ WARNING: Large rotation angle ({rotation_angle_deg:.1f}°)")
                print("   This might indicate incorrect midline marking or coordinate issues.")
        else:
            rotation_axis = np.array([0.0, 0.0, 1.0])
            rotation_angle = 0.0
            rotation_angle_deg = 0.0
            print("Midline already aligned - no rotation needed")
        
        result = {
            'normal': normal,
            'centroid': centroid,
            'rotation_axis': rotation_axis,
            'rotation_angle': rotation_angle,
            'rotation_angle_deg': rotation_angle_deg,
            'points_3d': points_3d
        }
        
        print("="*60)
        
        return result
    
    def apply_alignment(self, plane_params, reverse=False):
        """
        Apply rotation to align the image at FULL RESOLUTION
        
        Parameters:
        -----------
        plane_params : dict
            Plane parameters from fit_midline_plane()
        reverse : bool
            If True, reverse the rotation direction
        """
        print("\n" + "="*60)
        print("APPLYING ALIGNMENT (FULL RESOLUTION)")
        print("="*60)

        rotation_angle = plane_params['rotation_angle']
        rotation_axis = plane_params['rotation_axis']

        # --- FIX: coordinate handedness mismatch (NumPy vs SimpleITK) ---
        rotation_angle = -rotation_angle
        print("NOTE: Inverting rotation angle to match SimpleITK coordinate system")

        # Reverse rotation if requested
        if reverse:
            rotation_angle = -rotation_angle
            print("⚠ REVERSING rotation direction")
        
        if abs(rotation_angle) < 0.001:
            print("Rotation angle is negligible, returning original image")
            return self.image
        
        print(f"Rotating image by {np.degrees(rotation_angle):.2f}°")
        print(f"Rotation axis: {rotation_axis}")
        print(f"Original image size: {self.image.shape}")
        print(f"⚠ Processing at FULL RESOLUTION - this may take several minutes!")
        print("=" * 60)
        
        total_start = time.time()
        
        original_shape = self.image.shape
        original_dtype = self.image.dtype
        
        # NO DOWNSAMPLING - use full resolution
        print(f"\n[Step 1/3] Converting to SimpleITK format...")
        step_start = time.time()
        
        # Convert to float32 for processing
        image_sitk = sitk.GetImageFromArray(self.image.astype(np.float32))
        image_sitk.SetSpacing([1.0, 1.0, 1.0])
        
        step_time = time.time() - step_start
        print(f"  Image shape: {self.image.shape} ({np.prod(self.image.shape):,} voxels)")
        print(f"  ✓ Completed in {step_time:.1f} seconds")
        
        # Set up transformation
        print(f"\n[Step 2/3] Setting up transformation...")
        step_start = time.time()
        
        transform = sitk.AffineTransform(3)
        
        # Use midline centroid as rotation center
        centroid = np.array(plane_params.get('centroid', [s / 2.0 for s in original_shape][::-1]))
        
        # SimpleITK expects (x, y, z) physical point
        center_physical = image_sitk.TransformContinuousIndexToPhysicalPoint(
            [float(centroid[0]), float(centroid[1]), float(centroid[2])]
        )
        
        rotation = Rotation.from_rotvec(rotation_angle * rotation_axis)
        rotation_matrix = rotation.as_matrix()
        matrix_flat = rotation_matrix.flatten().tolist()
        
        transform.SetMatrix(matrix_flat)
        transform.SetCenter(center_physical)
        
        step_time = time.time() - step_start
        print(f"  Rotation center: {centroid}")
        print(f"  ✓ Completed in {step_time:.1f} seconds")
        
        # Apply rotation at full resolution
        print(f"\n[Step 3/3] Applying rotation at full resolution...")
        print(f"  Using all available CPU threads...")
        print(f"  ⏳ This will take time - please be patient...")
        step_start = time.time()
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image_sitk)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(transform)
        resampler.SetDefaultPixelValue(0)
        resampler.SetNumberOfThreads(0)  # Use all available threads
        
        aligned_sitk = resampler.Execute(image_sitk)
        aligned = sitk.GetArrayFromImage(aligned_sitk)
        
        step_time = time.time() - step_start
        print(f"  ✓ Rotation completed in {step_time:.1f} seconds ({step_time/60:.1f} minutes)")
        
        # Convert back to original dtype
        print(f"\nConverting back to original dtype ({original_dtype})...")
        if original_dtype == np.uint16:
            aligned = np.clip(aligned, 0, 65535).astype(np.uint16)
        elif original_dtype == np.uint8:
            aligned = np.clip(aligned, 0, 255).astype(np.uint8)
        else:
            aligned = aligned.astype(original_dtype)
        
        total_time = time.time() - total_start
        print(f"\n{'='*60}")
        print(f"✓ ALIGNMENT COMPLETE")
        print(f"  Final shape: {aligned.shape}, dtype: {aligned.dtype}")
        print(f"  Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"{'='*60}")
        
        return aligned
    
    def visualize_alignment(self, original, aligned, plane_params, save_path=None):
        """Visualize the alignment results"""
        fig = plt.figure(figsize=(16, 12))
        
        mid_z = original.shape[0] // 2
        mid_y = original.shape[1] // 2
        mid_x = original.shape[2] // 2
        
        # Coronal view
        ax1 = plt.subplot(3, 2, 1)
        ax1.imshow(original[mid_z, :, :], cmap='gray')
        ax1.set_title(f'Original - Coronal (Z={mid_z})')
        ax1.axis('off')
        
        key = ('coronal', mid_z)
        if key in self.midline_points:
            points = np.array(self.midline_points[key])
            ax1.plot(points[:, 1], points[:, 0], 'r+', markersize=12, markeredgewidth=2)
            if len(points) > 1:
                ax1.plot(points[:, 1], points[:, 0], 'r-', linewidth=2)
        
        ax2 = plt.subplot(3, 2, 2)
        ax2.imshow(aligned[mid_z, :, :], cmap='gray')
        ax2.set_title(f'Aligned - Coronal (Z={mid_z})')
        ax2.axis('off')
        ax2.axvline(x=aligned.shape[2]//2, color='green', linestyle='--', 
                   linewidth=2, alpha=0.7, label='Expected midline')
        ax2.legend()
        
        # Axial view
        ax3 = plt.subplot(3, 2, 3)
        ax3.imshow(original[:, mid_y, :], cmap='gray')
        ax3.set_title(f'Original - Axial (Y={mid_y})')
        ax3.axis('off')
        
        key = ('axial', mid_y)
        if key in self.midline_points:
            points = np.array(self.midline_points[key])
            ax3.plot(points[:, 1], points[:, 0], 'r+', markersize=12, markeredgewidth=2)
            if len(points) > 1:
                ax3.plot(points[:, 1], points[:, 0], 'r-', linewidth=2)
        
        ax4 = plt.subplot(3, 2, 4)
        ax4.imshow(aligned[:, mid_y, :], cmap='gray')
        ax4.set_title(f'Aligned - Axial (Y={mid_y})')
        ax4.axis('off')
        ax4.axvline(x=aligned.shape[2]//2, color='green', linestyle='--', 
                   linewidth=2, alpha=0.7, label='Expected midline')
        ax4.legend()
        
        # Sagittal view
        ax5 = plt.subplot(3, 2, 5)
        ax5.imshow(original[:, :, mid_x], cmap='gray')
        ax5.set_title(f'Original - Sagittal (X={mid_x})')
        ax5.axis('off')
        
        key = ('sagittal', mid_x)
        if key in self.midline_points:
            points = np.array(self.midline_points[key])
            ax5.plot(points[:, 1], points[:, 0], 'r+', markersize=12, markeredgewidth=2)
            if len(points) > 1:
                ax5.plot(points[:, 1], points[:, 0], 'r-', linewidth=2)
        
        ax6 = plt.subplot(3, 2, 6)
        ax6.imshow(aligned[:, :, mid_x], cmap='gray')
        ax6.set_title(f'Aligned - Sagittal (X={mid_x})')
        ax6.axis('off')
        
        plt.suptitle(f'Midline Alignment Results (Full Resolution)\n'
                    f'Rotation: {plane_params["rotation_angle_deg"]:.2f}°',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Visualization saved to: {save_path}")
        
        plt.show()
    
    def save_midline_points(self, filepath):
        """Save midline points to JSON file"""
        data = {
            'midline_points': {f"{k[0]}_{k[1]}": [list(p) for p in v] 
                             for k, v in self.midline_points.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Midline points saved to: {filepath}")
    
    def load_midline_points(self, filepath):
        """Load midline points from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.midline_points = {}
        for k, v in data['midline_points'].items():
            view, slice_idx = k.rsplit('_', 1)
            self.midline_points[(view, int(slice_idx))] = [tuple(p) for p in v]
        
        print(f"✓ Loaded midline points from: {filepath}")


# Example usage
if __name__ == "__main__":
    from data_loader import DataLoader
    
    import sys
    PROJECT_PATH = resolve_project_path(sys.argv[1] if len(sys.argv) > 1 else None)
    
    # Load microCT data
    print("Loading microCT data...")
    loader = DataLoader(PROJECT_PATH)
    source_image_path = next(
        iter(sorted(loader.data_path.glob("*.tif")) + sorted(loader.data_path.glob("*.tiff"))),
        None,
    )
    microct = loader.load_microct()
    
    # Initialize aligner
    aligner = MidlineAligner(microct, name="MicroCT")
    
    # Interactive midline marking
    midline_points = aligner.mark_midline()
    
    # Save midline points
    midline_file = PROJECT_PATH / "data" / "midline_points.json"
    aligner.save_midline_points(midline_file)
    
    # Fit plane and calculate alignment
    if len(midline_points) >= 2:
        plane_params = aligner.fit_midline_plane()
        
        # Ask user if rotation direction looks correct
        print("\n" + "="*60)
        print("ROTATION DIRECTION CHECK")
        print("="*60)
        print(f"Computed rotation: {plane_params['rotation_angle_deg']:.2f}°")
        print(f"Rotation axis: {plane_params['rotation_axis']}")
        print("\n⚠ WARNING: Processing at full resolution will take MUCH longer!")
        print("Estimated time: 5-30 minutes depending on image size and CPU")
        print("\nIf the alignment result looks wrong (rotated in wrong direction),")
        print("you can set reverse=True in apply_alignment() call below.")
        print("="*60)
        
        # Apply alignment at full resolution (set reverse=True if rotation is in wrong direction)
        aligned_image = aligner.apply_alignment(plane_params, reverse=False)
        
        # Visualize results
        output_path = PROJECT_PATH / "outputs"
        output_path.mkdir(exist_ok=True)
        aligner.visualize_alignment(
            microct, 
            aligned_image, 
            plane_params,
            save_path=output_path / "00_midline_alignment_fullres.png"
        )
        
        # === NEW: AXIS VERIFICATION STEP ===
        print("\n" + "="*70)
        print("AXIS VERIFICATION STEP")
        print("="*70)
        print("Now you can verify and correct the axis orientation if needed.")
        print("This is useful if coronal/axial/sagittal views appear swapped.")
        
        response = input("\nDo you want to verify/correct axis orientation? (y/n): ").lower()
        
        if response == 'y':
            verifier = AxisVerifier(aligned_image, name="Aligned MicroCT")
            corrected_image = verifier.verify_and_correct_axes()
            
            # Ask if user wants to save the corrected image
            response = input("\nDo you want to save the axis-corrected image? (y/n): ").lower()
            
            if response == 'y':
                # Save corrected image
                processed_path = PROJECT_PATH / "data" / "processed"
                processed_path.mkdir(exist_ok=True)
                output_file = processed_path / "microct_aligned_corrected.tif"
                
                print(f"\nSaving axis-corrected image (this may take a few minutes)...")
                tifffile.imwrite(output_file, corrected_image)
                print(f"✓ Axis-corrected image saved to: {output_file}")
                if source_image_path is not None:
                    copy_metadata_sidecar(source_image_path, output_file)
                
                # Update the aligned image reference
                aligned_image = corrected_image
            else:
                print("Axis-corrected image not saved. Using original aligned image.")
        else:
            print("Skipping axis verification step.")
        
        # Save final aligned image (if not already saved as corrected)
        if response != 'y' or input("\nAlso save original aligned image? (y/n): ").lower() == 'y':
            processed_path = PROJECT_PATH / "data" / "processed"
            processed_path.mkdir(exist_ok=True)
            output_file = processed_path / "microct_aligned_fullres.tif"
            
            print(f"\nSaving aligned image (this may take a few minutes)...")
            tifffile.imwrite(output_file, aligned_image)
            print(f"✓ Aligned image saved to: {output_file}")
            if source_image_path is not None:
                copy_metadata_sidecar(source_image_path, output_file)
        
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("1. Check the alignment visualization")
        print("2. The green dashed line should align with the brain midline")
        print("3. Verify axis orientation is correct")
        print("4. If alignment looks good, use the saved .tif file for registration")
        print("5. If not satisfied, run again and adjust midline points")
        print("="*60)
    else:
        print("\nNot enough slices marked. Need at least 2 slices with midline points.")
