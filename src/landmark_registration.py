"""
uCT2CCF - Enhanced Landmark-Based Registration Module
Interactive tool for manual landmark selection and registration
Assumes image orientation has been corrected by midline alignment
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import SimpleITK as sitk
from pathlib import Path
import json
import tifffile
from scipy import ndimage
import argparse
from project_paths import resolve_project_path


class LandmarkSelector:
    """Interactive landmark selection tool with performance optimizations"""
    
    def __init__(self, moving_image, fixed_image, moving_name="MicroCT", 
                 fixed_name="Allen CCF"):
        """
        Initialize landmark selector
        
        Parameters:
        -----------
        moving_image : numpy.ndarray
            Moving image (microCT) - 3D array (Z, Y, X)
        fixed_image : numpy.ndarray
            Fixed image (Allen CCF) - 3D array (Z, Y, X)
        moving_name : str
            Name for moving image
        fixed_name : str
            Name for fixed image
        """
        self.moving_image = moving_image
        self.fixed_image = fixed_image
        self.moving_name = moving_name
        self.fixed_name = fixed_name
        
        # Landmarks: list of (z, y, x) coordinates
        self.moving_landmarks = []
        self.fixed_landmarks = []
        self.landmark_names = []
        
        # Store plot objects for efficient updates
        self.moving_plots = []
        self.fixed_plots = []
        
        # Current slice indices
        self.moving_slice = moving_image.shape[0] // 2
        self.fixed_slice = fixed_image.shape[0] // 2
        
        # Current view - use simple axis names
        self.view = 'Z'  # Default to Z-axis slicing
        
        # Figure elements
        self.fig = None
        self.ax_moving = None
        self.ax_fixed = None
        
        # Store image artists to update data directly for performance
        self.im_moving = None
        self.im_fixed = None
        self.zoom_factor = {"moving": 1.0, "fixed": 1.0}
        self.zoom_center = {"moving": None, "fixed": None}
        self.active_panel = "moving"
        self.pan_mode = False
        self._pan_anchor = None

    def get_slice(self, image, slice_idx, view='Z'):
        """Get 2D slice from 3D image based on view
        
        Image convention: (Z, Y, X)
        View indicates which axis we're slicing through
        """
        if view == 'Z':
            # Slice through Z axis, shows Y-X plane
            return image[slice_idx, :, :]
        elif view == 'Y':
            # Slice through Y axis, shows Z-X plane
            return image[:, slice_idx, :]
        elif view == 'X':
            # Slice through X axis, shows Z-Y plane
            return image[:, :, slice_idx]
    
    def get_max_slice(self, image, view):
        """Get maximum slice index for given view"""
        if view == 'Z':
            return image.shape[0] - 1
        elif view == 'Y':
            return image.shape[1] - 1
        elif view == 'X':
            return image.shape[2] - 1

    def get_display_shape(self, image, view):
        """Return displayed slice width/height for a view."""
        sample_slice = self.get_slice(image, 0, view)
        height, width = sample_slice.shape
        return int(width), int(height)

    def set_active_panel_from_axes(self, axes):
        """Track which image panel zoom controls should affect."""
        if axes == self.ax_moving:
            self.active_panel = "moving"
        elif axes == self.ax_fixed:
            self.active_panel = "fixed"

    def default_zoom_center(self, panel):
        """Use the center of the currently displayed slice."""
        image = self.moving_image if panel == "moving" else self.fixed_image
        width, height = self.get_display_shape(image, self.view)
        return (width - 1) / 2.0, (height - 1) / 2.0

    def change_zoom(self, panel, scale, center=None):
        """Zoom one panel for display only; landmark coordinates are unchanged."""
        self.zoom_factor[panel] = float(np.clip(
            self.zoom_factor[panel] * scale, 1.0, 24.0
        ))
        if center is not None:
            self.zoom_center[panel] = center
        elif self.zoom_center[panel] is None:
            self.zoom_center[panel] = self.default_zoom_center(panel)
        self.update_display()

    def reset_zoom(self, panel=None):
        """Reset zoom for one panel or both panels."""
        panels = ("moving", "fixed") if panel is None else (panel,)
        for name in panels:
            self.zoom_factor[name] = 1.0
            self.zoom_center[name] = None
        self.update_display()

    @staticmethod
    def clamp_axis_limits(limits, size):
        """Clamp axes limits to image bounds while preserving direction."""
        first, second = float(limits[0]), float(limits[1])
        reversed_axis = first > second
        lower, upper = (second, first) if reversed_axis else (first, second)
        span = upper - lower

        min_edge = -0.5
        max_edge = float(size) - 0.5
        if span >= max_edge - min_edge:
            lower = min_edge
        else:
            lower = float(np.clip(lower, min_edge, max_edge - span))
        upper = lower + span

        return (upper, lower) if reversed_axis else (lower, upper)

    def toggle_pan(self, enabled=None):
        """Toggle display-only panning for zoomed images."""
        self.pan_mode = (not self.pan_mode) if enabled is None else bool(enabled)
        self._pan_anchor = None
        self.update_display()

    def begin_pan(self, event):
        """Start dragging the current image view."""
        if event.inaxes not in (self.ax_moving, self.ax_fixed):
            return False
        if event.xdata is None or event.ydata is None:
            return False

        panel = "moving" if event.inaxes == self.ax_moving else "fixed"
        self.set_active_panel_from_axes(event.inaxes)
        self._pan_anchor = {
            "panel": panel,
            "axes": event.inaxes,
            "x": float(event.xdata),
            "y": float(event.ydata),
            "xlim": tuple(event.inaxes.get_xlim()),
            "ylim": tuple(event.inaxes.get_ylim()),
        }
        return True

    def drag_pan(self, event):
        """Move the zoomed display window without changing landmarks."""
        if self._pan_anchor is None:
            return
        if event.inaxes != self._pan_anchor["axes"]:
            return
        if event.xdata is None or event.ydata is None:
            return

        panel = self._pan_anchor["panel"]
        image = self.moving_image if panel == "moving" else self.fixed_image
        width, height = self.get_display_shape(image, self.view)

        dx = self._pan_anchor["x"] - float(event.xdata)
        dy = self._pan_anchor["y"] - float(event.ydata)
        xlim = (
            self._pan_anchor["xlim"][0] + dx,
            self._pan_anchor["xlim"][1] + dx,
        )
        ylim = (
            self._pan_anchor["ylim"][0] + dy,
            self._pan_anchor["ylim"][1] + dy,
        )

        event.inaxes.set_xlim(*self.clamp_axis_limits(xlim, width))
        event.inaxes.set_ylim(*self.clamp_axis_limits(ylim, height))

        x0, x1 = event.inaxes.get_xlim()
        y0, y1 = event.inaxes.get_ylim()
        self.zoom_center[panel] = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        self._pan_anchor.update({
            "x": float(event.xdata),
            "y": float(event.ydata),
            "xlim": tuple(event.inaxes.get_xlim()),
            "ylim": tuple(event.inaxes.get_ylim()),
        })
        self.fig.canvas.draw_idle()

    def end_pan(self, event=None):
        """Stop dragging the image view."""
        self._pan_anchor = None

    def apply_zoom_to_axes(self, axes, panel):
        """Apply stored zoom limits to an axes without altering image data."""
        image = self.moving_image if panel == "moving" else self.fixed_image
        width, height = self.get_display_shape(image, self.view)
        if self.zoom_factor[panel] <= 1.001:
            axes.set_xlim(-0.5, width - 0.5)
            axes.set_ylim(height - 0.5, -0.5)
            return

        if self.zoom_center[panel] is None:
            cx, cy = self.default_zoom_center(panel)
        else:
            cx, cy = self.zoom_center[panel]

        half_w = width / (2.0 * self.zoom_factor[panel])
        half_h = height / (2.0 * self.zoom_factor[panel])
        xmin = float(np.clip(cx - half_w, -0.5, max(-0.5, width - 2 * half_w - 0.5)))
        xmax = xmin + 2 * half_w
        ymin = float(np.clip(cy - half_h, -0.5, max(-0.5, height - 2 * half_h - 0.5)))
        ymax = ymin + 2 * half_h
        axes.set_xlim(xmin, xmax)
        axes.set_ylim(ymax, ymin)
    
    def onclick(self, event):
        """Handle mouse clicks for landmark selection"""
        if event.inaxes is None or event.button != 1:
            return
        if event.inaxes not in (self.ax_moving, self.ax_fixed):
            return
        if event.xdata is None or event.ydata is None:
            return
        self.set_active_panel_from_axes(event.inaxes)

        if self.pan_mode:
            self.begin_pan(event)
            return
        
        x, y = int(event.xdata), int(event.ydata)
        
        if event.inaxes == self.ax_moving:
            self.zoom_center["moving"] = (float(x), float(y))
            # Convert 2D click to 3D coordinate based on current view
            if self.view == 'Z':
                landmark = (self.moving_slice, y, x)  # (Z, Y, X)
            elif self.view == 'Y':
                landmark = (y, self.moving_slice, x)  # (Z, Y, X)
            else:  # X
                landmark = (y, x, self.moving_slice)  # (Z, Y, X)
            
            self.moving_landmarks.append(landmark)
            landmark_id = len(self.moving_landmarks)
            print(f"Moving landmark {landmark_id}: {landmark} (Z, Y, X)")
            print(f"  Waiting for corresponding point in {self.fixed_name}...")
            
            # Create and store plot objects
            marker = self.ax_moving.plot(x, y, 'r+', markersize=15, markeredgewidth=2)[0]
            text = self.ax_moving.text(x+5, y-5, str(landmark_id), 
                                       color='red', fontsize=12, fontweight='bold',
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            self.moving_plots.append({'marker': marker, 'text': text})
            
        elif event.inaxes == self.ax_fixed:
            self.zoom_center["fixed"] = (float(x), float(y))
            if len(self.fixed_landmarks) >= len(self.moving_landmarks):
                print("WARNING: Select a point on the moving image first!")
                return
            
            # Convert 2D click to 3D coordinate based on current view
            if self.view == 'Z':
                landmark = (self.fixed_slice, y, x)  # (Z, Y, X)
            elif self.view == 'Y':
                landmark = (y, self.fixed_slice, x)  # (Z, Y, X)
            else:  # X
                landmark = (y, x, self.fixed_slice)  # (Z, Y, X)

            self.fixed_landmarks.append(landmark)
            landmark_id = len(self.fixed_landmarks)
            default_name = f"Point_{landmark_id}"
            self.landmark_names.append(default_name)
            print(f"Fixed landmark {landmark_id}: {landmark} (Z, Y, X) - {default_name}")
            print(f"✓ Landmark pair {landmark_id} complete!\n")

            # Create and store plot objects
            marker = self.ax_fixed.plot(x, y, 'g+', markersize=15, markeredgewidth=2)[0]
            text = self.ax_fixed.text(x+5, y-5, str(landmark_id), 
                                      color='green', fontsize=12, fontweight='bold',
                                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            self.fixed_plots.append({'marker': marker, 'text': text})

        self.fig.canvas.draw_idle()

    def update_display(self):
        """Update the display efficiently by updating data and visibility"""
        # Update image data
        moving_slice_data = self.get_slice(self.moving_image, self.moving_slice, self.view)
        fixed_slice_data = self.get_slice(self.fixed_image, self.fixed_slice, self.view)
        
        self.im_moving.set_data(moving_slice_data)
        self.im_fixed.set_data(fixed_slice_data)
        
        self.im_moving.set_clim(vmin=moving_slice_data.min(), vmax=moving_slice_data.max())
        self.im_fixed.set_clim(vmin=fixed_slice_data.min(), vmax=fixed_slice_data.max())
        
        # Determine plane description
        if self.view == 'Z':
            plane_desc = 'Y-X plane'
        elif self.view == 'Y':
            plane_desc = 'Z-X plane'
        else:  # X
            plane_desc = 'Z-Y plane'
        
        # Update titles
        title_moving = (
            f"{self.moving_name}\n"
            f"{plane_desc} | slice {self.moving_slice}/"
            f"{self.get_max_slice(self.moving_image, self.view)} | "
            f"zoom {self.zoom_factor['moving']:.1f}x | "
            f"pan {'on' if self.pan_mode else 'off'} | "
            f"landmarks {len(self.moving_landmarks)}"
        )
        self.ax_moving.set_title(title_moving, fontsize=9, pad=8)
        
        title_fixed = (
            f"{self.fixed_name}\n"
            f"{plane_desc} | slice {self.fixed_slice}/"
            f"{self.get_max_slice(self.fixed_image, self.view)} | "
            f"zoom {self.zoom_factor['fixed']:.1f}x | "
            f"pan {'on' if self.pan_mode else 'off'} | "
            f"landmarks {len(self.fixed_landmarks)}"
        )
        self.ax_fixed.set_title(title_fixed, fontsize=9, pad=8)

        # Update landmark visibility based on current slice
        for idx, lm in enumerate(self.moving_landmarks):
            visible = False
            if (self.view == 'Z' and lm[0] == self.moving_slice) or \
               (self.view == 'Y' and lm[1] == self.moving_slice) or \
               (self.view == 'X' and lm[2] == self.moving_slice):
                visible = True
            self.moving_plots[idx]['marker'].set_visible(visible)
            self.moving_plots[idx]['text'].set_visible(visible)

        for idx, lm in enumerate(self.fixed_landmarks):
            visible = False
            if (self.view == 'Z' and lm[0] == self.fixed_slice) or \
               (self.view == 'Y' and lm[1] == self.fixed_slice) or \
               (self.view == 'X' and lm[2] == self.fixed_slice):
                visible = True
            self.fixed_plots[idx]['marker'].set_visible(visible)
            self.fixed_plots[idx]['text'].set_visible(visible)

        # Show unpaired landmark warning
        if len(self.moving_landmarks) > len(self.fixed_landmarks):
            self.ax_fixed.text(0.5, 0.98, 
                             'WARNING: Click to pair with moving landmark!',
                             transform=self.ax_fixed.transAxes,
                             color='yellow', fontsize=12, fontweight='bold',
                             ha='center', va='top',
                             bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))

        self.apply_zoom_to_axes(self.ax_moving, "moving")
        self.apply_zoom_to_axes(self.ax_fixed, "fixed")
        self.fig.canvas.draw_idle()

    def on_undo(self, event):
        """Efficiently undo by removing plot objects"""
        if self.moving_landmarks and self.fixed_landmarks:
            self.moving_landmarks.pop()
            self.fixed_landmarks.pop()
            if self.landmark_names:
                self.landmark_names.pop()
            
            plots_mov = self.moving_plots.pop()
            plots_fix = self.fixed_plots.pop()
            plots_mov['marker'].remove()
            plots_mov['text'].remove()
            plots_fix['marker'].remove()
            plots_fix['text'].remove()
            
            print("Removed last landmark pair")
            
        elif self.moving_landmarks:
            self.moving_landmarks.pop()
            plots_mov = self.moving_plots.pop()
            plots_mov['marker'].remove()
            plots_mov['text'].remove()
            print("Removed unpaired moving landmark")
            
        self.update_display()
        
    def select_landmarks(self):
        """Launch interactive landmark selection interface"""
        print("\n" + "="*70)
        print("LANDMARK SELECTION TOOL")
        print("="*70)
        print("\nRecommended Landmarks for Brain Registration:")
        print("  1. Bregma (intersection of coronal and sagittal sutures)")
        print("  2. Lambda (posterior end of sagittal suture)")
        print("  3. Anterior tip of brain")
        print("  4. Posterior tip of brain")
        print("  5. Ventral tip of brain")
        print("  6. Left & right edges of brain (at widest point)")
        print("  7. Ventricle corners (if visible)")
        print("  8. Olfactory bulb tips")
        print("\nInstructions:")
        print("  - Click on MOVING image first, then corresponding point on FIXED")
        print("  - Use scroll wheel or sliders to navigate slices")
        print("  - Use axis buttons (Z/Y/X) to change slicing direction")
        print("  - Select at least 6-8 landmark pairs (more is better)")
        print("  - 'Undo' removes the last pair")
        print("  - 'Done' when finished")
        print("\nKeyboard shortcuts:")
        print("  - 'u': Undo last landmark")
        print("  - 'z': Switch to Z-axis view")
        print("  - 'y': Switch to Y-axis view")
        print("  - 'x': Switch to X-axis view")
        print("  - Use MicroCT/Allen zoom buttons to zoom either image")
        print("  - 'p': Toggle pan mode; drag either zoomed image to move around")
        print("  - '+/-/0': Zoom active image in/out/reset")
        print("  - 'l': List all landmarks")
        print("  - 'Enter': Done")
        print("="*70 + "\n")
        
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.subplots_adjust(left=0.04, right=0.96, top=0.88, bottom=0.08)
        
        # Adjust layout to leave more space at bottom for controls
        # Images sit below the title band and above sliders/buttons.
        self.ax_moving = plt.axes([0.06, 0.34, 0.40, 0.52])
        self.ax_fixed = plt.axes([0.54, 0.34, 0.40, 0.52])

        # Create imshow artists once for performance
        moving_slice_data = self.get_slice(self.moving_image, self.moving_slice, self.view)
        fixed_slice_data = self.get_slice(self.fixed_image, self.fixed_slice, self.view)
        
        self.im_moving = self.ax_moving.imshow(moving_slice_data, cmap='gray')
        self.im_fixed = self.ax_fixed.imshow(fixed_slice_data, cmap='gray')
        
        self.ax_moving.axis('off')
        self.ax_fixed.axis('off')

        # Add sliders - positioned below the images
        ax_slider_moving = plt.axes([0.14, 0.235, 0.27, 0.025])
        ax_slider_fixed = plt.axes([0.64, 0.235, 0.27, 0.025])
        
        max_moving = self.get_max_slice(self.moving_image, self.view)
        max_fixed = self.get_max_slice(self.fixed_image, self.view)
        
        self.slider_moving = Slider(
            ax_slider_moving, 
            f'Moving {self.view}',
            0, 
            max_moving,
            valinit=self.moving_slice,
            valstep=1
        )
        
        self.slider_fixed = Slider(
            ax_slider_fixed,
            f'Fixed {self.view}',
            0,
            max_fixed,
            valinit=self.fixed_slice,
            valstep=1
        )
        
        def update_moving_slice(val):
            self.moving_slice = int(self.slider_moving.val)
            self.update_display()
        
        def update_fixed_slice(val):
            self.fixed_slice = int(self.slider_fixed.val)
            self.update_display()
        
        self.slider_moving.on_changed(update_moving_slice)
        self.slider_fixed.on_changed(update_fixed_slice)
        self.slider_moving.label.set_fontsize(9)
        self.slider_fixed.label.set_fontsize(9)
        self.slider_moving.valtext.set_fontsize(9)
        self.slider_fixed.valtext.set_fontsize(9)
        
        # Add buttons - positioned at the bottom
        # Row 1: View controls
        ax_view_z = plt.axes([0.14, 0.145, 0.075, 0.04])
        ax_view_y = plt.axes([0.225, 0.145, 0.075, 0.04])
        ax_view_x = plt.axes([0.31, 0.145, 0.075, 0.04])
        
        # Row 2: Action buttons
        ax_list = plt.axes([0.54, 0.145, 0.075, 0.04])
        ax_undo = plt.axes([0.625, 0.145, 0.075, 0.04])
        ax_done = plt.axes([0.82, 0.145, 0.10, 0.04])

        # Row 3-4: separate zoom controls for each image panel.
        ax_zoom_moving_in = plt.axes([0.08, 0.085, 0.08, 0.036])
        ax_zoom_moving_out = plt.axes([0.172, 0.085, 0.08, 0.036])
        ax_zoom_moving_reset = plt.axes([0.264, 0.085, 0.08, 0.036])
        ax_zoom_fixed_in = plt.axes([0.08, 0.035, 0.08, 0.036])
        ax_zoom_fixed_out = plt.axes([0.172, 0.035, 0.08, 0.036])
        ax_zoom_fixed_reset = plt.axes([0.264, 0.035, 0.08, 0.036])
        ax_pan = plt.axes([0.38, 0.06, 0.095, 0.04])
        
        btn_view_z = Button(ax_view_z, 'Z-axis')
        btn_view_y = Button(ax_view_y, 'Y-axis')
        btn_view_x = Button(ax_view_x, 'X-axis')
        btn_list = Button(ax_list, 'List All', color='lightblue')
        btn_undo = Button(ax_undo, 'Undo', color='lightyellow')
        btn_done = Button(ax_done, 'Done', color='lightgreen')
        btn_zoom_moving_in = Button(ax_zoom_moving_in, 'MicroCT +')
        btn_zoom_moving_out = Button(ax_zoom_moving_out, 'MicroCT -')
        btn_zoom_moving_reset = Button(ax_zoom_moving_reset, 'MicroCT 1x')
        btn_zoom_fixed_in = Button(ax_zoom_fixed_in, 'Allen +')
        btn_zoom_fixed_out = Button(ax_zoom_fixed_out, 'Allen -')
        btn_zoom_fixed_reset = Button(ax_zoom_fixed_reset, 'Allen 1x')
        btn_pan = Button(ax_pan, 'Pan off')
        for button in (
            btn_zoom_moving_in, btn_zoom_moving_out, btn_zoom_moving_reset,
            btn_zoom_fixed_in, btn_zoom_fixed_out, btn_zoom_fixed_reset,
        ):
            button.label.set_fontsize(9)
        
        def on_done(event):
            if len(self.moving_landmarks) != len(self.fixed_landmarks):
                print("\nWARNING: Unpaired landmarks detected!")
                print(f"  Moving: {len(self.moving_landmarks)}, Fixed: {len(self.fixed_landmarks)}")
                print("  Please complete or undo the last landmark.")
                return
            plt.close(self.fig)
        
        def on_list(event):
            print("\n" + "="*60)
            print("CURRENT LANDMARKS")
            print("="*60)
            for i, (m_lm, f_lm, name) in enumerate(zip(
                self.moving_landmarks, self.fixed_landmarks, self.landmark_names), 1):
                print(f"{i}. {name}")
                print(f"   Moving (Z,Y,X): {m_lm}")
                print(f"   Fixed  (Z,Y,X): {f_lm}")
            if len(self.moving_landmarks) > len(self.fixed_landmarks):
                print(f"\nWARNING: Unpaired moving landmark: {self.moving_landmarks[-1]}")
            print("="*60 + "\n")
        
        def change_view(new_view):
            self.view = new_view
            max_mov = self.get_max_slice(self.moving_image, self.view)
            max_fix = self.get_max_slice(self.fixed_image, self.view)
            
            self.moving_slice = max_mov // 2
            self.fixed_slice = max_fix // 2
            self.zoom_factor = {"moving": 1.0, "fixed": 1.0}
            self.zoom_center = {"moving": None, "fixed": None}
            
            self.slider_moving.valmin = 0
            self.slider_moving.valmax = max_mov
            self.slider_moving.set_val(self.moving_slice)
            self.slider_moving.label.set_text(f'Moving {self.view}')
            
            self.slider_fixed.valmin = 0
            self.slider_fixed.valmax = max_fix
            self.slider_fixed.set_val(self.fixed_slice)
            self.slider_fixed.label.set_text(f'Fixed {self.view}')
            
            print(f"\nSwitched to {self.view}-axis view")
            self.update_display()

        def set_pan_mode(enabled):
            self.toggle_pan(enabled)
            btn_pan.label.set_text('Pan on' if self.pan_mode else 'Pan off')
            print(f"Pan mode {'on' if self.pan_mode else 'off'}")

        def on_pan(event):
            set_pan_mode(not self.pan_mode)
        
        btn_done.on_clicked(on_done)
        btn_undo.on_clicked(self.on_undo)
        btn_list.on_clicked(on_list)
        btn_view_z.on_clicked(lambda e: change_view('Z'))
        btn_view_y.on_clicked(lambda e: change_view('Y'))
        btn_view_x.on_clicked(lambda e: change_view('X'))
        btn_zoom_moving_in.on_clicked(lambda e: self.change_zoom("moving", 1.5))
        btn_zoom_moving_out.on_clicked(lambda e: self.change_zoom("moving", 1/1.5))
        btn_zoom_moving_reset.on_clicked(lambda e: self.reset_zoom("moving"))
        btn_zoom_fixed_in.on_clicked(lambda e: self.change_zoom("fixed", 1.5))
        btn_zoom_fixed_out.on_clicked(lambda e: self.change_zoom("fixed", 1/1.5))
        btn_zoom_fixed_reset.on_clicked(lambda e: self.reset_zoom("fixed"))
        btn_pan.on_clicked(on_pan)
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        def on_motion(event):
            if self._pan_anchor is not None:
                self.drag_pan(event)
                return
            if event.inaxes in (self.ax_moving, self.ax_fixed):
                self.set_active_panel_from_axes(event.inaxes)

        self.fig.canvas.mpl_connect('motion_notify_event', on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.end_pan)
        
        def on_scroll(event):
            if event.inaxes == self.ax_moving:
                self.active_panel = "moving"
                max_slice = self.get_max_slice(self.moving_image, self.view)
                if event.key in ("control", "ctrl", "cmd", "super"):
                    center = None
                    if event.xdata is not None and event.ydata is not None:
                        center = (float(event.xdata), float(event.ydata))
                    self.change_zoom("moving", 1.25 if event.button == "up" else 1/1.25, center)
                    return
                if event.button == 'up':
                    self.moving_slice = min(self.moving_slice + 1, max_slice)
                else:
                    self.moving_slice = max(self.moving_slice - 1, 0)
                self.slider_moving.set_val(self.moving_slice)
            elif event.inaxes == self.ax_fixed:
                self.active_panel = "fixed"
                max_slice = self.get_max_slice(self.fixed_image, self.view)
                if event.key in ("control", "ctrl", "cmd", "super"):
                    center = None
                    if event.xdata is not None and event.ydata is not None:
                        center = (float(event.xdata), float(event.ydata))
                    self.change_zoom("fixed", 1.25 if event.button == "up" else 1/1.25, center)
                    return
                if event.button == 'up':
                    self.fixed_slice = min(self.fixed_slice + 1, max_slice)
                else:
                    self.fixed_slice = max(self.fixed_slice - 1, 0)
                self.slider_fixed.set_val(self.fixed_slice)
            self.update_display()
        
        self.fig.canvas.mpl_connect('scroll_event', on_scroll)
        
        # Add keyboard shortcuts
        def on_key(event):
            if event.key == 'u':
                self.on_undo(None)
            elif event.key == 'enter':
                on_done(None)
            elif event.key == 'z':
                change_view('Z')
            elif event.key == 'y':
                change_view('Y')
            elif event.key == 'x':
                change_view('X')
            elif event.key == 'l':
                on_list(None)
            elif event.key in ('+', '='):
                self.change_zoom(self.active_panel, 1.5)
            elif event.key in ('-', '_'):
                self.change_zoom(self.active_panel, 1/1.5)
            elif event.key == '0':
                self.reset_zoom(self.active_panel)
            elif event.key == 'p':
                set_pan_mode(not self.pan_mode)
        
        self.fig.canvas.mpl_connect('key_press_event', on_key)
        
        # Call update_display once to set initial titles
        self.update_display()
        
        plt.show()
        
        print(f"\n✓ Selected {len(self.fixed_landmarks)} complete landmark pairs")
        if len(self.moving_landmarks) != len(self.fixed_landmarks):
            print(f"WARNING: {len(self.moving_landmarks) - len(self.fixed_landmarks)} unpaired landmarks")
        
        return self.moving_landmarks, self.fixed_landmarks
    
    def save_landmarks(self, filepath):
        """Save landmarks to JSON file"""
        data = {
            'moving_landmarks': [list(lm) for lm in self.moving_landmarks],
            'fixed_landmarks': [list(lm) for lm in self.fixed_landmarks],
            'landmark_names': self.landmark_names,
            'moving_name': self.moving_name,
            'fixed_name': self.fixed_name,
            'num_pairs': len(self.fixed_landmarks),
            'coordinate_convention': 'Z, Y, X'
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Landmarks saved to: {filepath}")
    
    def load_landmarks(self, filepath):
        """Load landmarks from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.moving_landmarks = [tuple(lm) for lm in data['moving_landmarks']]
        self.fixed_landmarks = [tuple(lm) for lm in data['fixed_landmarks']]
        self.landmark_names = data.get('landmark_names', [])
        
        print(f"✓ Loaded {len(self.fixed_landmarks)} landmark pairs from: {filepath}")
        return self.moving_landmarks, self.fixed_landmarks


class LandmarkRegistration:
    """Perform registration based on landmarks with enhanced options"""
    
    def __init__(self, project_path):
        """Initialize registration"""
        self.project_path = Path(project_path)
        self.output_path = self.project_path / "outputs"
        self.output_path.mkdir(exist_ok=True)
    
    def numpy_to_sitk(self, image_array, spacing=(1.0, 1.0, 1.0)):
        """Convert numpy array to SimpleITK image"""
        image_sitk = sitk.GetImageFromArray(image_array.astype(np.float32))
        image_sitk.SetSpacing(spacing)
        return image_sitk
    
    def compute_landmark_error(self, moving_landmarks, fixed_landmarks, transform,
                              moving_spacing, fixed_spacing):
        """Compute registration error for each landmark pair"""
        errors = []

        try:
            moving_to_fixed = transform.GetInverse()
        except RuntimeError:
            moving_to_fixed = None
            print(
                "WARNING: could not invert registration transform for landmark "
                "error calculation. Reporting fixed-to-moving error instead."
            )
        
        for m_lm, f_lm in zip(moving_landmarks, fixed_landmarks):
            # Convert from (Z,Y,X) to physical coordinates (x,y,z)
            m_phys = [m_lm[2] * moving_spacing[0],  # X
                     m_lm[1] * moving_spacing[1],   # Y
                     m_lm[0] * moving_spacing[2]]   # Z
            f_phys = [f_lm[2] * fixed_spacing[0],   # X
                     f_lm[1] * fixed_spacing[1],    # Y
                     f_lm[0] * fixed_spacing[2]]    # Z
            
            if moving_to_fixed is not None:
                # The transform used by Resample maps fixed/output points into
                # moving/input space. Invert it to compare moving landmarks in
                # fixed atlas physical coordinates.
                m_transformed = moving_to_fixed.TransformPoint(m_phys)
                reference = f_phys
            else:
                m_transformed = transform.TransformPoint(f_phys)
                reference = m_phys

            error = np.sqrt(
                sum((m_transformed[i] - reference[i])**2 for i in range(3))
            )
            errors.append(error)
        
        return np.array(errors)
    
    def landmark_registration(self, moving_image, fixed_image, 
                            moving_landmarks, fixed_landmarks,
                            moving_spacing=(0.025, 0.025, 0.025),
                            fixed_spacing=(0.025, 0.025, 0.025),
                            transform_type='affine'):
        """
        Perform landmark-based registration
        
        Parameters:
        -----------
        moving_image : numpy.ndarray
            Moving image (microCT) - shape (Z, Y, X)
        fixed_image : numpy.ndarray
            Fixed image (Allen CCF) - shape (Z, Y, X)
        moving_landmarks : list of tuples
            Landmarks in moving image (Z, Y, X)
        fixed_landmarks : list of tuples
            Corresponding landmarks in fixed image (Z, Y, X)
        moving_spacing : tuple
            Voxel spacing in mm (for X, Y, Z directions)
        fixed_spacing : tuple
            Voxel spacing in mm (for X, Y, Z directions)
        transform_type : str
            'affine', 'similarity', or 'rigid'
            
        Returns:
        --------
        tuple : (registered_image, transform, metrics)
        """
        print("\n" + "="*70)
        print("LANDMARK-BASED REGISTRATION")
        print("="*70)
        print(f"Number of landmark pairs: {len(moving_landmarks)}")
        print(f"Transform type: {transform_type.upper()}")
        print(f"Moving image shape (Z,Y,X): {moving_image.shape}")
        print(f"Fixed image shape (Z,Y,X): {fixed_image.shape}")
        
        min_landmarks = {'rigid': 3, 'similarity': 3, 'affine': 4}
        if len(moving_landmarks) < min_landmarks.get(transform_type, 4):
            raise ValueError(f"Need at least {min_landmarks[transform_type]} "
                           f"landmark pairs for {transform_type} registration")
        
        # Convert to SimpleITK images
        print("\nConverting images to SimpleITK format...")
        moving_sitk = self.numpy_to_sitk(moving_image, moving_spacing)
        fixed_sitk = self.numpy_to_sitk(fixed_image, fixed_spacing)
        
        # Convert landmarks to physical coordinates
        # Note: landmarks are stored as (Z, Y, X) but SimpleITK expects (x, y, z)
        moving_points = []
        fixed_points = []
        
        print("\nConverting landmarks to physical coordinates...")
        for i, (m_lm, f_lm) in enumerate(zip(moving_landmarks, fixed_landmarks), 1):
            # Convert (Z,Y,X) to physical (x,y,z) 
            m_phys = [m_lm[2] * moving_spacing[0],  # X
                     m_lm[1] * moving_spacing[1],   # Y
                     m_lm[0] * moving_spacing[2]]   # Z
            f_phys = [f_lm[2] * fixed_spacing[0],   # X
                     f_lm[1] * fixed_spacing[1],    # Y
                     f_lm[0] * fixed_spacing[2]]    # Z
            
            moving_points.extend(m_phys)
            fixed_points.extend(f_phys)
            
            print(f"  Pair {i}:")
            print(f"    Moving (Z,Y,X): {m_lm} -> Physical (x,y,z): {m_phys}")
            print(f"    Fixed  (Z,Y,X): {f_lm} -> Physical (x,y,z): {f_phys}")
        
        # Initialize transform based on type
        print(f"\nInitializing {transform_type} transform...")
        if transform_type == 'affine':
            transform = sitk.AffineTransform(3)
        elif transform_type == 'similarity':
            transform = sitk.Similarity3DTransform()
        elif transform_type == 'rigid':
            transform = sitk.Euler3DTransform()
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")
        
        # Use landmark correspondence to initialize transform
        landmark_transform = sitk.LandmarkBasedTransformInitializer(
            transform,
            fixed_points,
            moving_points
        )
        
        print("✓ Initial transformation computed")
        print(f"  Transform parameters: {landmark_transform.GetParameters()}")
        
        # Compute registration errors
        errors = self.compute_landmark_error(
            moving_landmarks, fixed_landmarks, landmark_transform,
            moving_spacing, fixed_spacing
        )
        
        print(f"\nLandmark registration errors (mm):")
        print(f"  Mean error: {errors.mean():.3f}")
        print(f"  Std error:  {errors.std():.3f}")
        print(f"  Min error:  {errors.min():.3f}")
        print(f"  Max error:  {errors.max():.3f}")
        
        # Resample moving image to fixed image space
        print("\nResampling moving image to fixed space...")
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_sitk)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(landmark_transform)
        resampler.SetDefaultPixelValue(0)
        
        registered_sitk = resampler.Execute(moving_sitk)
        registered_array = sitk.GetArrayFromImage(registered_sitk)
        
        print("✓ Registration complete")
        print(f"  Registered image shape: {registered_array.shape}")
        print("="*70)
        
        metrics = {
            'errors': errors,
            'mean_error': errors.mean(),
            'std_error': errors.std(),
            'max_error': errors.max(),
            'min_error': errors.min()
        }
        
        return registered_array, landmark_transform, metrics
    
    def refine_registration(self, moving_image, fixed_image, initial_transform,
                          moving_spacing=(0.025, 0.025, 0.025),
                          fixed_spacing=(0.025, 0.025, 0.025),
                          iterations=200):
        """
        Refine registration using intensity-based optimization
        
        Parameters:
        -----------
        moving_image : numpy.ndarray
            Moving image
        fixed_image : numpy.ndarray
            Fixed image
        initial_transform : SimpleITK.Transform
            Initial transform from landmark registration
        moving_spacing : tuple
            Voxel spacing for moving image
        fixed_spacing : tuple
            Voxel spacing for fixed image
        iterations : int
            Number of optimization iterations
            
        Returns:
        --------
        tuple : (refined_image, refined_transform)
        """
        print("\n" + "="*70)
        print("REFINING REGISTRATION WITH INTENSITY-BASED OPTIMIZATION")
        print("="*70)
        
        moving_sitk = self.numpy_to_sitk(moving_image, moving_spacing)
        fixed_sitk = self.numpy_to_sitk(fixed_image, fixed_spacing)
        
        # Setup registration method
        registration = sitk.ImageRegistrationMethod()
        
        # Metric: Mutual Information
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(0.01)
        
        # Optimizer
        registration.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=iterations,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10
        )
        registration.SetOptimizerScalesFromPhysicalShift()
        
        # Interpolator
        registration.SetInterpolator(sitk.sitkLinear)
        
        # Initial transform
        registration.SetInitialTransform(initial_transform, inPlace=False)
        
        # Multi-resolution
        registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        
        # Progress monitoring
        iteration_count = [0]
        def iteration_callback():
            iteration_count[0] += 1
            if iteration_count[0] % 20 == 0:
                print(f"  Iteration {iteration_count[0]}: "
                      f"Metric = {registration.GetMetricValue():.4f}")
        
        registration.AddCommand(sitk.sitkIterationEvent, iteration_callback)
        
        print("\nStarting optimization...")
        refined_transform = registration.Execute(fixed_sitk, moving_sitk)
        
        print(f"\n✓ Optimization complete")
        print(f"  Final metric value: {registration.GetMetricValue():.4f}")
        print(f"  Optimizer stop condition: {registration.GetOptimizerStopConditionDescription()}")
        
        # Apply refined transform
        print("\nApplying refined transform...")
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_sitk)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(refined_transform)
        resampler.SetDefaultPixelValue(0)
        
        refined_sitk = resampler.Execute(moving_sitk)
        refined_array = sitk.GetArrayFromImage(refined_sitk)
        
        print("="*70)
        
        return refined_array, refined_transform
    
    def visualize_registration(self, moving, fixed, registered, 
                              slice_indices=None, save_name='landmark_registration'):
        """
        Visualize registration results across multiple slices
        
        Parameters:
        -----------
        moving : numpy.ndarray
            Original moving image
        fixed : numpy.ndarray
            Fixed image
        registered : numpy.ndarray
            Registered moving image
        slice_indices : list or None
            Slice indices to visualize (defaults to quartiles)
        save_name : str
            Base name for saved figure
        """
        if slice_indices is None:
            # Use quartile slices
            nz = fixed.shape[0]
            slice_indices = [nz//4, nz//2, 3*nz//4]
        
        n_slices = len(slice_indices)
        fig, axes = plt.subplots(n_slices, 4, figsize=(16, 4*n_slices))
        
        if n_slices == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Landmark-Based Registration Results', 
                    fontsize=16, fontweight='bold')
        
        for i, slice_idx in enumerate(slice_indices):
            # Ensure slice is within bounds
            slice_idx = min(slice_idx, fixed.shape[0]-1)
            
            # Get slice from moving (may have different size)
            if slice_idx < moving.shape[0]:
                mov_slice = moving[slice_idx]
            else:
                mov_slice = moving[moving.shape[0]//2]
            
            fix_slice = fixed[slice_idx]
            reg_slice = registered[slice_idx]
            
            # Original moving
            axes[i, 0].imshow(mov_slice, cmap='gray')
            axes[i, 0].set_title(f'Moving (Z-slice {slice_idx})')
            axes[i, 0].axis('off')
            
            # Fixed
            axes[i, 1].imshow(fix_slice, cmap='gray')
            axes[i, 1].set_title(f'Fixed (Z-slice {slice_idx})')
            axes[i, 1].axis('off')
            
            # Registered
            axes[i, 2].imshow(reg_slice, cmap='gray')
            axes[i, 2].set_title(f'Registered')
            axes[i, 2].axis('off')
            
            # Overlay (magenta=fixed, green=registered)
            overlay = np.zeros((*fix_slice.shape, 3))
            fix_norm = (fix_slice - fix_slice.min()) / (fix_slice.max() - fix_slice.min() + 1e-8)
            reg_norm = (reg_slice - reg_slice.min()) / (reg_slice.max() - reg_slice.min() + 1e-8)
            
            overlay[:, :, 0] = fix_norm  # Red channel (for magenta)
            overlay[:, :, 1] = reg_norm  # Green channel
            overlay[:, :, 2] = fix_norm  # Blue channel (for magenta)
            
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title('Overlay (Magenta=Fixed, Green=Reg)')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        
        save_path = self.output_path / f"{save_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {save_path}")
        plt.show()
    
    def visualize_detailed(self, moving, fixed, registered, 
                          slice_idx=None, save_name='registration_detailed'):
        """
        Create detailed visualization with checkerboard and difference maps
        """
        if slice_idx is None:
            slice_idx = fixed.shape[0] // 2
        
        fig = plt.figure(figsize=(16, 12))
        
        # Get slices
        if slice_idx < moving.shape[0]:
            mov_slice = moving[slice_idx]
        else:
            mov_slice = moving[moving.shape[0]//2]
        
        fix_slice = fixed[slice_idx]
        reg_slice = registered[slice_idx]
        
        # Row 1: Original images
        ax1 = plt.subplot(3, 3, 1)
        ax1.imshow(mov_slice, cmap='gray')
        ax1.set_title('Original Moving')
        ax1.axis('off')
        
        ax2 = plt.subplot(3, 3, 2)
        ax2.imshow(fix_slice, cmap='gray')
        ax2.set_title('Fixed (Allen CCF)')
        ax2.axis('off')
        
        ax3 = plt.subplot(3, 3, 3)
        ax3.imshow(reg_slice, cmap='gray')
        ax3.set_title('Registered Moving')
        ax3.axis('off')
        
        # Row 2: Overlays
        ax4 = plt.subplot(3, 3, 4)
        ax4.imshow(fix_slice, cmap='Purples', alpha=0.7)
        ax4.imshow(reg_slice, cmap='Greens', alpha=0.5)
        ax4.set_title('Color Overlay\n(Purple=Fixed, Green=Registered)')
        ax4.axis('off')
        
        ax5 = plt.subplot(3, 3, 5)
        # Blend overlay
        blend = 0.5 * (fix_slice / fix_slice.max()) + 0.5 * (reg_slice / reg_slice.max())
        ax5.imshow(blend, cmap='gray')
        ax5.set_title('Blended Overlay')
        ax5.axis('off')
        
        ax6 = plt.subplot(3, 3, 6)
        checker = self._create_checkerboard(fix_slice, reg_slice, n_tiles=8)
        ax6.imshow(checker, cmap='gray')
        ax6.set_title('Checkerboard')
        ax6.axis('off')
        
        # Row 3: Difference and contours
        ax7 = plt.subplot(3, 3, 7)
        diff = np.abs(fix_slice.astype(float) - reg_slice.astype(float))
        im = ax7.imshow(diff, cmap='hot')
        ax7.set_title('Absolute Difference')
        ax7.axis('off')
        plt.colorbar(im, ax=ax7, fraction=0.046)
        
        ax8 = plt.subplot(3, 3, 8)
        ax8.imshow(fix_slice, cmap='gray')
        # Add contours of registered image
        reg_edges = ndimage.sobel(reg_slice)
        ax8.contour(reg_edges, levels=[reg_edges.mean()], colors='cyan', linewidths=2)
        ax8.set_title('Fixed + Registered Contours')
        ax8.axis('off')
        
        ax9 = plt.subplot(3, 3, 9)
        # Difference histogram
        ax9.hist(diff.flatten(), bins=50, color='red', alpha=0.7)
        ax9.set_xlabel('Absolute Difference')
        ax9.set_ylabel('Frequency')
        ax9.set_title('Difference Histogram')
        ax9.grid(True, alpha=0.3)
        
        plt.suptitle(f'Detailed Registration Analysis (Z-slice {slice_idx})',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_path / f"{save_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Detailed visualization saved to: {save_path}")
        plt.show()
    
    def _create_checkerboard(self, img1, img2, n_tiles=8):
        """Create checkerboard pattern of two images"""
        h, w = img1.shape
        tile_h, tile_w = h // n_tiles, w // n_tiles
        
        # Normalize images
        img1_norm = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-8)
        img2_norm = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-8)
        
        checker = np.zeros_like(img1_norm)
        for i in range(n_tiles):
            for j in range(n_tiles):
                if (i + j) % 2 == 0:
                    checker[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w] = \
                        img1_norm[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
                else:
                    checker[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w] = \
                        img2_norm[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
        
        return checker
    
    def save_transform(self, transform, filepath):
        """Save transform to file"""
        sitk.WriteTransform(transform, str(filepath))
        print(f"✓ Transform saved to: {filepath}")
    
    def load_transform(self, filepath):
        """Load transform from file"""
        transform = sitk.ReadTransform(str(filepath))
        print(f"✓ Transform loaded from: {filepath}")
        return transform


def prompt_spacing_um() -> float:
    """Ask for positive isotropic microCT spacing in micrometers."""
    while True:
        raw = input(
            "\nCould not determine microCT voxel spacing automatically.\n"
            "Enter microCT voxel spacing in micrometers, e.g. 72: "
        ).strip()
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a number, for example 72.")
            continue
        if value <= 0:
            print("Spacing must be positive.")
            continue
        return value


def infer_spacing_um(image_path: Path) -> float | None:
    """Infer isotropic image spacing from a metadata sidecar if available."""
    sidecar_path = image_path.with_suffix(image_path.suffix + ".metadata.json")
    if sidecar_path.exists():
        with open(sidecar_path, "r") as f:
            metadata = json.load(f)

        values = metadata.get("spacing_um_xyz") or metadata.get("spacing_um_zyx")
        spacing = isotropic_spacing_from_values(values, "metadata sidecar")
        if spacing is not None:
            return spacing

    try:
        with tifffile.TiffFile(image_path) as tif:
            imagej_metadata = tif.imagej_metadata or {}
            unit = str(imagej_metadata.get("unit", "")).lower()
            spacing = imagej_metadata.get("spacing")
            x_resolution = tif.pages[0].tags.get("XResolution")
            y_resolution = tif.pages[0].tags.get("YResolution")
            resolution_unit = tif.pages[0].tags.get("ResolutionUnit")
    except Exception:
        return None

    if spacing is not None and unit in {"um", "µm", "micron", "microns"}:
        print(f"✓ Inferred microCT spacing from ImageJ TIFF metadata: {float(spacing):g} um")
        return float(spacing)

    if x_resolution and y_resolution and resolution_unit:
        unit_name = str(resolution_unit.value).upper()
        if unit_name in {"CENTIMETER", "INCH"}:
            x_per_unit = rational_to_float(x_resolution.value)
            y_per_unit = rational_to_float(y_resolution.value)
            unit_um = 10000.0 if unit_name == "CENTIMETER" else 25400.0
            values = [unit_um / x_per_unit, unit_um / y_per_unit]
            return isotropic_spacing_from_values(values, "TIFF resolution tags")

    return None


def rational_to_float(value) -> float:
    try:
        numerator, denominator = value
        return float(numerator) / float(denominator)
    except TypeError:
        return float(value)


def isotropic_spacing_from_values(values, source: str) -> float | None:
    if not values:
        return None

    values = [float(v) for v in values]
    if any(v <= 0 for v in values):
        return None
    if max(values) - min(values) > 1e-6:
        print(
            f"\nWARNING: {source} spacing is anisotropic: {values} um. "
            "This registration entry point currently uses one isotropic value."
        )
        return None

    spacing = values[0]
    print(f"✓ Inferred microCT spacing from {source}: {spacing:g} um")
    return spacing


# Example usage
if __name__ == "__main__":
    from data_loader import DataLoader
    from preprocessing import Preprocessor

    parser = argparse.ArgumentParser(
        description="Register the aligned microCT volume to the Allen CCF atlas.")
    parser.add_argument("project_path", nargs="?",
                        help="Project folder containing data/ and outputs/.")
    parser.add_argument(
        "--microct-spacing", type=float, default=None,
        help="Input microCT voxel spacing in micrometers before resampling. "
             "Overrides metadata detection. If omitted and metadata is missing, "
             "you will be prompted.")
    parser.add_argument(
        "--atlas-spacing", type=float, default=25.0,
        help="Allen CCF atlas spacing in micrometers. Default: 25.")
    parser.add_argument(
        "--resample-moving", action="store_true",
        help="Upsample/downsample the moving microCT to atlas spacing before "
             "landmark selection. By default, registration keeps the moving "
             "image in its native voxel grid and uses physical spacing.")
    args = parser.parse_args()

    PROJECT_PATH = resolve_project_path(args.project_path)
    
    # Load data
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    loader = DataLoader(PROJECT_PATH)
    
    # Load microCT (use aligned and corrected version if available)
    corrected_file = PROJECT_PATH / "data" / "processed" / "microct_aligned_corrected.tif"
    aligned_file = PROJECT_PATH / "data" / "processed" / "microct_aligned_fullres.tif"
    
    if corrected_file.exists():
        print(f"\nLoading axis-corrected microCT from: {corrected_file}")
        microct = tifffile.imread(corrected_file)
        microct_path = corrected_file
        print("✓ Using axis-corrected image")
    elif aligned_file.exists():
        print(f"\nLoading aligned microCT from: {aligned_file}")
        microct = tifffile.imread(aligned_file)
        microct_path = aligned_file
        print("✓ Using midline-aligned image")
    else:
        print("\nWARNING: No aligned microCT found. Loading original...")
        print("It's recommended to run midline alignment first!")
        microct = loader.load_microct()
        microct_path = next(
            iter(sorted(loader.data_path.glob("*.tif")) + sorted(loader.data_path.glob("*.tiff"))),
            None,
        )
    
    # Load Allen CCF
    print(f"\nLoading Allen CCF atlas ({args.atlas_spacing:g}um resolution)...")
    loader.load_allen_ccf(resolution=int(args.atlas_spacing))
    atlas = loader.atlas_image
    
    print(f"\nMicroCT shape (Z,Y,X): {microct.shape}")
    print(f"Atlas shape (Z,Y,X): {atlas.shape}")
    
    # Preprocess microCT
    print("\n" + "="*70)
    print("PREPROCESSING")
    print("="*70)
    preprocessor = Preprocessor(PROJECT_PATH)
    
    current_res = (
        args.microct_spacing
        if args.microct_spacing is not None
        else infer_spacing_um(microct_path) if microct_path is not None else None
    )
    if current_res is None:
        current_res = prompt_spacing_um()
    target_res = args.atlas_spacing
    moving_spacing_um = current_res

    if args.resample_moving:
        print(f"\nResampling microCT from {current_res:g}um "
              f"to {target_res:g}um...")
        microct_for_registration = preprocessor.resample_image(
            microct,
            current_res,
            target_res
        )
        moving_spacing_um = target_res
    else:
        print(f"\nKeeping microCT at native {current_res:g}um spacing.")
        print("Registration will use physical spacing instead of resampling.")
        microct_for_registration = microct
    
    # Normalize intensity
    print("\nNormalizing intensity...")
    microct_normalized = preprocessor.normalize_intensity(
        microct_for_registration,
        method='percentile'
    )
    
    # Normalize atlas
    atlas_normalized = atlas.astype(float) / atlas.max()
    
    print(f"\nPreprocessed microCT shape (Z,Y,X): {microct_normalized.shape}")
    print(f"Note: Image axes should be correctly oriented after midline alignment")
    
    # Interactive landmark selection
    print("\n" + "="*70)
    print("LANDMARK SELECTION")
    print("="*70)
    
    selector = LandmarkSelector(
        microct_normalized, 
        atlas_normalized,
        moving_name="MicroCT",
        fixed_name="Allen CCF"
    )
    
    # Option to load existing landmarks
    landmark_file = PROJECT_PATH / "data" / "landmarks.json"
    if landmark_file.exists():
        response = input(f"\nFound existing landmarks at {landmark_file}.\n"
                        "Load existing landmarks? (y/n): ")
        if response.lower() == 'y':
            moving_lm, fixed_lm = selector.load_landmarks(landmark_file)
        else:
            moving_lm, fixed_lm = selector.select_landmarks()
            selector.save_landmarks(landmark_file)
    else:
        moving_lm, fixed_lm = selector.select_landmarks()
        selector.save_landmarks(landmark_file)
    
    # Perform registration
    if len(moving_lm) >= 4 and len(moving_lm) == len(fixed_lm):
        registrar = LandmarkRegistration(PROJECT_PATH)
        
        # Initial landmark-based registration
        print("\n" + "="*70)
        print("PERFORMING LANDMARK REGISTRATION")
        print("="*70)
        
        registered, transform, metrics = registrar.landmark_registration(
            microct_normalized, 
            atlas_normalized,
            moving_lm, 
            fixed_lm,
            moving_spacing=(moving_spacing_um / 1000.0,) * 3,
            fixed_spacing=(target_res / 1000.0,) * 3,
            transform_type='affine'  # Options: 'affine', 'similarity', 'rigid'
        )

        # Save the landmark transform immediately, before visualization windows
        # or optional refinement can interrupt the run.
        processed_dir = PROJECT_PATH / "data" / "processed"
        processed_dir.mkdir(exist_ok=True, parents=True)
        output_file = processed_dir / "microct_registered.tif"
        transform_file = processed_dir / "transform_landmark.tfm"
        metrics_file = processed_dir / "registration_metrics.json"

        tifffile.imwrite(output_file, (registered * 65535).astype(np.uint16))
        registrar.save_transform(transform, transform_file)

        metrics_data = {
            'mean_error_mm': float(metrics['mean_error']),
            'std_error_mm': float(metrics['std_error']),
            'max_error_mm': float(metrics['max_error']),
            'min_error_mm': float(metrics['min_error']),
            'num_landmarks': len(moving_lm),
            'errors_per_landmark': [float(e) for e in metrics['errors']],
            'coordinate_convention': 'Z, Y, X',
            'moving_image': str(microct_path) if microct_path is not None else None,
            'moving_shape_zyx': list(microct_normalized.shape),
            'fixed_shape_zyx': list(atlas_normalized.shape),
            'moving_spacing_um': moving_spacing_um,
            'fixed_spacing_um': target_res,
            'transform_saved_direction': 'fixed-to-moving',
            'landmark_file': str(landmark_file),
        }
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)

        print(f"\n✓ Registered image saved to: {output_file}")
        print(f"✓ Transform saved to: {transform_file}")
        print(f"✓ Metrics saved to: {metrics_file}")
        
        # Visualize results
        registrar.visualize_registration(
            microct_normalized, 
            atlas_normalized, 
            registered,
            save_name='03_landmark_registration'
        )
        
        registrar.visualize_detailed(
            microct_normalized,
            atlas_normalized,
            registered,
            save_name='03_landmark_registration_detailed'
        )
        
        # Option to refine with intensity-based registration
        refine = input("\nRefine registration with intensity-based optimization? (y/n): ")
        if refine.lower() == 'y':
            refined, refined_transform = registrar.refine_registration(
                microct_normalized,
                atlas_normalized,
                transform,
                moving_spacing=(moving_spacing_um / 1000.0,) * 3,
                fixed_spacing=(target_res / 1000.0,) * 3,
                iterations=200
            )
            
            # Visualize refined results
            registrar.visualize_registration(
                microct_normalized,
                atlas_normalized,
                refined,
                save_name='04_refined_registration'
            )
            
            registrar.visualize_detailed(
                microct_normalized,
                atlas_normalized,
                refined,
                save_name='04_refined_registration_detailed'
            )
            
            # Save refined results
            output_file = PROJECT_PATH / "data" / "processed" / "microct_registered_refined.tif"
            output_file.parent.mkdir(exist_ok=True, parents=True)
            tifffile.imwrite(output_file, (refined * 65535).astype(np.uint16))
            print(f"\n✓ Refined registered image saved to: {output_file}")
            
            # Save refined transform
            transform_file = PROJECT_PATH / "data" / "processed" / "transform_refined.tfm"
            registrar.save_transform(refined_transform, transform_file)
        else:
            refined = registered
            refined_transform = transform
        
        print("\n" + "="*70)
        print("REGISTRATION COMPLETE")
        print("="*70)
        print(f"\nRegistration quality metrics:")
        print(f"  Mean landmark error: {metrics['mean_error']:.3f} mm")
        print(f"  Std landmark error:  {metrics['std_error']:.3f} mm")
        print(f"  Max landmark error:  {metrics['max_error']:.3f} mm")
        print(f"\nOutput files:")
        print(f"  - Registered image: {output_file}")
        print(f"  - Transform: {transform_file}")
        print(f"  - Metrics: {metrics_file}")
        print(f"  - Visualizations: {registrar.output_path}")
        print("="*70)
        
    else:
        print("\nWARNING: Not enough valid landmark pairs selected.")
        print(f"   Moving: {len(moving_lm)}, Fixed: {len(fixed_lm)}")
        print("   Need at least 4 complete pairs for registration.")
