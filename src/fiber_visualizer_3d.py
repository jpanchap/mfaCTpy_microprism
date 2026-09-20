"""
Interactive 3D Fiber Visualization Tool

Displays tracked fibers in 3D space with interactive controls:
- 3D view with rotation
- Slice plane navigation (coronal, sagittal, axial)
- Toggle individual fibers on/off
- Color-coded by brain region
- Option to show microCT or Allen CCF reference

Dependencies:
pip install numpy matplotlib tifffile pandas nrrd
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons, RadioButtons
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
import tifffile
import json
from pathlib import Path
import argparse

try:
    from project_paths import resolve_project_path
except ImportError:
    resolve_project_path = None


class FiberVisualizer3D:
    """Interactive 3D fiber visualization"""
    
    def __init__(self, image_path, fiber_data_path, ccf_path=None):
        """
        Initialize visualizer
        
        Parameters:
        -----------
        image_path : str
            Path to registered microCT image
        fiber_data_path : str
            Path to fiber_data.json
        ccf_path : str, optional
            Path to Allen CCF annotation or template
        """
        print("="*70)
        print("LOADING DATA FOR 3D VISUALIZATION")
        print("="*70)
        
        # Load microCT image
        print(f"\nLoading microCT image: {image_path}")
        self.microct_image = tifffile.imread(image_path)
        self.nz, self.ny, self.nx = self.microct_image.shape
        print(f"✓ MicroCT shape: {self.microct_image.shape}")
        
        # Load CCF if available
        self.ccf_image = None
        if ccf_path and Path(ccf_path).exists():
            print(f"\nLoading Allen CCF: {ccf_path}")
            try:
                if Path(ccf_path).suffix == '.nrrd':
                    import nrrd
                    self.ccf_image, _ = nrrd.read(str(ccf_path))
                else:
                    self.ccf_image = tifffile.imread(ccf_path)
                print(f"✓ CCF shape: {self.ccf_image.shape}")
                
                if self.ccf_image.shape != self.microct_image.shape:
                    print(f"⚠️  Warning: CCF shape doesn't match microCT, will resize")
            except Exception as e:
                print(f"⚠️  Could not load CCF: {e}")
                self.ccf_image = None
        
        # Set current image (default to microCT)
        self.current_image = self.microct_image
        self.image_mode = 'microct'
        
        # Load fiber data
        print(f"\nLoading fiber data: {fiber_data_path}")
        with open(fiber_data_path, 'r') as f:
            fiber_list = json.load(f)
        
        # Parse fibers
        self.fibers = []
        self.fiber_colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for fdata in fiber_list:
            fiber = {
                'id': fdata['fiber_id'],
                'top': np.array([fdata['top_z'], fdata['top_y'], fdata['top_x']]),
                'bottom': np.array([fdata['bottom_z'], fdata['bottom_y'], fdata['bottom_x']]),
                'color': self.fiber_colors[fdata['fiber_id'] % len(self.fiber_colors)],
                'region_name': fdata.get('region_name', 'N/A'),
                'region_acronym': fdata.get('region_acronym', 'N/A'),
                'visible': True  # For toggling
            }
            self.fibers.append(fiber)
        
        print(f"✓ Loaded {len(self.fibers)} fibers")
        
        # Current slice positions
        self.slice_z = self.nz // 2
        self.slice_y = self.ny // 2
        self.slice_x = self.nx // 2
        
        # View state
        self.show_slice = {'coronal': True, 'sagittal': False, 'axial': False}
        self.slice_alpha = 0.3
        
        # Zoom state
        self.zoom_level = 1.0
        
        # Calculate proper aspect ratios for display
        # Assuming 25um isotropic voxels, dimensions are already in correct proportion
        # But we can adjust if needed
        self.aspect_ratio = (self.nx, self.ny, self.nz)
        
        # Figure elements (will be created in start())
        self.fig = None
        self.ax_3d = None
        self.slice_artists = {}
        self.fiber_lines = {}
        
        print("\n✓ Initialization complete!")
        print("="*70)
    
    def create_fiber_line(self, fiber):
        """Create 3D line for a fiber"""
        top = fiber['top']
        bottom = fiber['bottom']
        
        # Create line from top to bottom
        line_x = [top[2], bottom[2]]  # X coordinates
        line_y = [top[1], bottom[1]]  # Y coordinates
        line_z = [top[0], bottom[0]]  # Z coordinates
        
        return line_x, line_y, line_z
    
    def draw_fibers_3d(self):
        """Draw all fibers in 3D view"""
        # Remove existing fiber lines properly
        for line_collection in list(self.fiber_lines.values()):
            try:
                line_collection.remove()
            except:
                pass
        self.fiber_lines.clear()
        
        # Draw each fiber
        for fiber in self.fibers:
            if not fiber['visible']:
                continue
            
            line_x, line_y, line_z = self.create_fiber_line(fiber)
            
            # Draw line - returns a list, take first element
            line = self.ax_3d.plot(line_x, line_y, line_z, 
                                   color=fiber['color'], 
                                   linewidth=3, 
                                   alpha=0.8,
                                   zorder=10)[0]
            self.fiber_lines[fiber['id']] = line
            
            # Add markers at endpoints
            top = fiber['top']
            bottom = fiber['bottom']
            
            # Top marker (entry point)
            self.ax_3d.scatter([top[2]], [top[1]], [top[0]], 
                              color=fiber['color'], s=50, 
                              marker='o', edgecolors='white', linewidths=1.5,
                              zorder=11)
            
            # Bottom marker (tip)
            self.ax_3d.scatter([bottom[2]], [bottom[1]], [bottom[0]], 
                              color=fiber['color'], s=80, 
                              marker='s', edgecolors='white', linewidths=1.5,
                              zorder=11)
    
    def update_slice_coronal(self, slice_z):
        """Update coronal slice (constant Z) - front view, shows XY plane"""
        if 'coronal' in self.slice_artists:
            try:
                self.slice_artists['coronal'].remove()
            except:
                pass
            del self.slice_artists['coronal']
        
        if self.show_slice['coronal']:
            # Coronal slice: constant Z (dorsal-ventral position)
            # Shows X (left-right) vs Y (anterior-posterior) plane
            # This is like looking at the brain from the FRONT
            slice_data = self.current_image[slice_z, :, :]
            
            # Normalize for display
            slice_norm = slice_data.astype(float) / (slice_data.max() + 1e-8)
            
            # Create meshgrid: X and Y vary, Z is constant
            xx, yy = np.meshgrid(range(self.nx), range(self.ny))
            zz = np.full_like(xx, slice_z, dtype=float)
            
            # Plot slice
            surf = self.ax_3d.plot_surface(xx, yy, zz, 
                                          facecolors=plt.cm.gray(slice_norm),
                                          alpha=self.slice_alpha,
                                          shade=False,
                                          zorder=1)
            self.slice_artists['coronal'] = surf
    
    def update_slice_sagittal(self, slice_x):
        """Update sagittal slice (constant X) - side view, shows YZ plane"""
        if 'sagittal' in self.slice_artists:
            try:
                self.slice_artists['sagittal'].remove()
            except:
                pass
            del self.slice_artists['sagittal']
        
        if self.show_slice['sagittal']:
            # Sagittal slice: constant X (left-right position)
            # Shows Y (anterior-posterior) vs Z (dorsal-ventral) plane
            # This is like looking at the brain from the SIDE
            slice_data = self.current_image[:, :, slice_x]
            
            # Normalize for display
            slice_norm = slice_data.astype(float) / (slice_data.max() + 1e-8)
            
            # Create meshgrid: Y and Z vary, X is constant
            yy, zz = np.meshgrid(range(self.ny), range(self.nz))
            xx = np.full_like(yy, slice_x, dtype=float)
            
            # Plot slice
            surf = self.ax_3d.plot_surface(xx, yy, zz,
                                          facecolors=plt.cm.gray(slice_norm),
                                          alpha=self.slice_alpha,
                                          shade=False,
                                          zorder=1)
            self.slice_artists['sagittal'] = surf
    
    def update_slice_axial(self, slice_y):
        """Update axial slice (constant Y) - top view, shows XZ plane"""
        if 'axial' in self.slice_artists:
            try:
                self.slice_artists['axial'].remove()
            except:
                pass
            del self.slice_artists['axial']
        
        if self.show_slice['axial']:
            # Axial slice: constant Y (anterior-posterior position)
            # Shows X (left-right) vs Z (dorsal-ventral) plane
            # This is like looking at the brain from the TOP
            slice_data = self.current_image[:, slice_y, :]
            
            # Normalize for display
            slice_norm = slice_data.astype(float) / (slice_data.max() + 1e-8)
            
            # Create meshgrid: X and Z vary, Y is constant
            xx, zz = np.meshgrid(range(self.nx), range(self.nz))
            yy = np.full_like(xx, slice_y, dtype=float)
            
            # Plot slice
            surf = self.ax_3d.plot_surface(xx, yy, zz,
                                          facecolors=plt.cm.gray(slice_norm),
                                          alpha=self.slice_alpha,
                                          shade=False,
                                          zorder=1)
            self.slice_artists['axial'] = surf
    
    def update_display(self):
        """Update the entire 3D display"""
        # Store current view
        elev = self.ax_3d.elev
        azim = self.ax_3d.azim
        
        # Clear axis content but keep the axis
        for artist in self.ax_3d.lines + self.ax_3d.collections:
            try:
                artist.remove()
            except:
                pass
        
        # Clear slice artists
        for key in list(self.slice_artists.keys()):
            try:
                self.slice_artists[key].remove()
            except:
                pass
        self.slice_artists.clear()
        
        # Redraw fibers
        self.draw_fibers_3d()
        
        # Update slices - NOW CORRECT
        self.update_slice_coronal(self.slice_z)  # Coronal uses Z
        self.update_slice_sagittal(self.slice_x)  # Sagittal uses X
        self.update_slice_axial(self.slice_y)     # Axial uses Y
        
        # Set labels and limits with CORRECTED axes
        self.ax_3d.set_xlabel('X (Left-Right)', fontsize=10)
        self.ax_3d.set_ylabel('Y (Anterior-Posterior)', fontsize=10)
        self.ax_3d.set_zlabel('Z (Dorsal-Ventral)', fontsize=10)
        
        # Apply zoom by adjusting axis limits
        center_x = self.nx / 2
        center_y = self.ny / 2
        center_z = self.nz / 2
        
        range_x = self.nx / (2 * self.zoom_level)
        range_y = self.ny / (2 * self.zoom_level)
        range_z = self.nz / (2 * self.zoom_level)
        
        self.ax_3d.set_xlim(center_x - range_x, center_x + range_x)
        self.ax_3d.set_ylim(center_y - range_y, center_y + range_y)
        self.ax_3d.set_zlim(center_z - range_z, center_z + range_z)
        
        # Set aspect ratio to show proper proportions
        # This makes the brain look anatomically correct
        self.ax_3d.set_box_aspect(self.aspect_ratio)
        
        # Restore view
        self.ax_3d.view_init(elev=elev, azim=azim)
        
        # Update title with visible fiber count and zoom
        visible_count = sum(1 for f in self.fibers if f['visible'])
        self.ax_3d.set_title(f'3D Fiber Visualization ({visible_count}/{len(self.fibers)} fibers visible) - Zoom: {self.zoom_level:.1f}x',
                            fontsize=12, fontweight='bold', pad=20)
        
        self.fig.canvas.draw_idle()
    
    def start(self):
        """Start interactive visualization"""
        print("\n" + "="*70)
        print("STARTING INTERACTIVE 3D VISUALIZATION")
        print("="*70)
        print("\nControls:")
        print("  - Mouse drag: Rotate 3D view")
        print("  - Mouse wheel: Zoom in/out (scroll up/down)")
        print("  - Sliders: Navigate slice planes")
        print("  - View buttons: Toggle slice planes on/off")
        print("  - Fiber checkboxes: Show/hide individual fibers")
        print("  - Preset views: Quick camera angles (Top/Side/Front/3D)")
        print("  - Reset Zoom: Return to default zoom level")
        if self.ccf_image is not None:
            print("  - Image mode: Switch between MicroCT and Allen CCF")
        print("="*70 + "\n")
        
        # Create figure with 3D axis and control panel
        self.fig = plt.figure(figsize=(18, 10))
        
        # 3D visualization (left side - larger)
        self.ax_3d = self.fig.add_subplot(121, projection='3d')
        
        # Initial draw
        self.draw_fibers_3d()
        self.update_slice_coronal(self.slice_z)  # FIXED: Coronal uses Z
        
        self.ax_3d.set_xlabel('X (Left-Right)', fontsize=10)
        self.ax_3d.set_ylabel('Y (Anterior-Posterior)', fontsize=10)
        self.ax_3d.set_zlabel('Z (Dorsal-Ventral)', fontsize=10)
        self.ax_3d.set_title('3D Fiber Visualization', fontsize=12, fontweight='bold')
        
        # Set proper aspect ratio for anatomically correct display
        self.ax_3d.set_box_aspect(self.aspect_ratio)
        
        # Set initial view angle
        self.ax_3d.view_init(elev=20, azim=45)
        
        # Add mouse wheel zoom functionality
        def on_scroll(event):
            """Handle mouse wheel zoom"""
            if event.inaxes == self.ax_3d:
                # Get scroll direction
                if event.button == 'up':
                    # Zoom in
                    self.zoom_level = min(self.zoom_level * 1.1, 5.0)  # Max 5x zoom
                elif event.button == 'down':
                    # Zoom out
                    self.zoom_level = max(self.zoom_level / 1.1, 0.5)  # Min 0.5x zoom
                
                self.update_display()
        
        self.fig.canvas.mpl_connect('scroll_event', on_scroll)
        
        # Control panel (right side)
        # Image mode selector (microCT vs CCF)
        if self.ccf_image is not None:
            ax_radio = plt.axes([0.60, 0.88, 0.15, 0.08])
            radio = RadioButtons(ax_radio, ('MicroCT', 'Allen CCF'), active=0)
            
            def change_image_mode(label):
                if label == 'Allen CCF':
                    self.current_image = self.ccf_image
                    self.image_mode = 'ccf'
                else:
                    self.current_image = self.microct_image
                    self.image_mode = 'microct'
                self.update_display()
            
            radio.on_clicked(change_image_mode)
        
        # Sliders for slice navigation - NOW CORRECT
        ax_slider_z = plt.axes([0.60, 0.82, 0.30, 0.02])
        ax_slider_x = plt.axes([0.60, 0.78, 0.30, 0.02])
        ax_slider_y = plt.axes([0.60, 0.74, 0.30, 0.02])
        
        self.slider_z = Slider(ax_slider_z, 'Coronal (Z)', 0, self.nz-1, 
                               valinit=self.slice_z, valstep=1)
        self.slider_x = Slider(ax_slider_x, 'Sagittal (X)', 0, self.nx-1, 
                               valinit=self.slice_x, valstep=1)
        self.slider_y = Slider(ax_slider_y, 'Axial (Y)', 0, self.ny-1, 
                               valinit=self.slice_y, valstep=1)
        
        def update_z(val):
            self.slice_z = int(val)
            self.update_display()
        
        def update_x(val):
            self.slice_x = int(val)
            self.update_display()
        
        def update_y(val):
            self.slice_y = int(val)
            self.update_display()
        
        self.slider_z.on_changed(update_z)
        self.slider_x.on_changed(update_x)
        self.slider_y.on_changed(update_y)
        
        # View toggle buttons
        ax_btn_coronal = plt.axes([0.60, 0.68, 0.09, 0.04])
        ax_btn_sagittal = plt.axes([0.70, 0.68, 0.09, 0.04])
        ax_btn_axial = plt.axes([0.80, 0.68, 0.09, 0.04])
        
        btn_coronal = Button(ax_btn_coronal, 'Coronal', 
                            color='lightgreen' if self.show_slice['coronal'] else 'lightgray')
        btn_sagittal = Button(ax_btn_sagittal, 'Sagittal',
                             color='lightgreen' if self.show_slice['sagittal'] else 'lightgray')
        btn_axial = Button(ax_btn_axial, 'Axial',
                          color='lightgreen' if self.show_slice['axial'] else 'lightgray')
        
        def toggle_coronal(event):
            self.show_slice['coronal'] = not self.show_slice['coronal']
            btn_coronal.color = 'lightgreen' if self.show_slice['coronal'] else 'lightgray'
            self.update_display()
        
        def toggle_sagittal(event):
            self.show_slice['sagittal'] = not self.show_slice['sagittal']
            btn_sagittal.color = 'lightgreen' if self.show_slice['sagittal'] else 'lightgray'
            self.update_display()
        
        def toggle_axial(event):
            self.show_slice['axial'] = not self.show_slice['axial']
            btn_axial.color = 'lightgreen' if self.show_slice['axial'] else 'lightgray'
            self.update_display()
        
        btn_coronal.on_clicked(toggle_coronal)
        btn_sagittal.on_clicked(toggle_sagittal)
        btn_axial.on_clicked(toggle_axial)
        
        # Fiber visibility checkboxes
        ax_checks = plt.axes([0.60, 0.10, 0.35, 0.55])
        ax_checks.set_title('Fiber Visibility', fontsize=11, fontweight='bold')
        ax_checks.axis('off')
        
        # Create checkbox for each fiber
        fiber_labels = []
        for fiber in self.fibers:
            acronym = fiber['region_acronym']
            if acronym != 'N/A':
                label = f"F{fiber['id']}: {acronym}"
            else:
                region = fiber['region_name'][:25]
                label = f"F{fiber['id']}: {region}"
            fiber_labels.append(label)
        
        # Split into two columns if too many fibers
        max_per_column = 15
        if len(self.fibers) <= max_per_column:
            # Single column
            check_ax = plt.axes([0.62, 0.12, 0.30, 0.50])
            check_ax.axis('off')
            
            checks = CheckButtons(check_ax, fiber_labels, 
                                 [f['visible'] for f in self.fibers])
            
            def toggle_fiber(label):
                # Find fiber index from label
                fiber_id = int(label.split(':')[0].replace('F', ''))
                for fiber in self.fibers:
                    if fiber['id'] == fiber_id:
                        fiber['visible'] = not fiber['visible']
                        break
                self.update_display()
            
            checks.on_clicked(toggle_fiber)
            
        else:
            # Two columns
            mid = (len(self.fibers) + 1) // 2
            
            check_ax1 = plt.axes([0.60, 0.12, 0.17, 0.50])
            check_ax1.axis('off')
            checks1 = CheckButtons(check_ax1, fiber_labels[:mid],
                                  [f['visible'] for f in self.fibers[:mid]])
            
            check_ax2 = plt.axes([0.78, 0.12, 0.17, 0.50])
            check_ax2.axis('off')
            checks2 = CheckButtons(check_ax2, fiber_labels[mid:],
                                  [f['visible'] for f in self.fibers[mid:]])
            
            def toggle_fiber(label):
                fiber_id = int(label.split(':')[0].replace('F', ''))
                for fiber in self.fibers:
                    if fiber['id'] == fiber_id:
                        fiber['visible'] = not fiber['visible']
                        break
                self.update_display()
            
            checks1.on_clicked(toggle_fiber)
            checks2.on_clicked(toggle_fiber)
        
        # View preset buttons with zoom reset
        ax_view_top = plt.axes([0.60, 0.04, 0.07, 0.04])
        ax_view_side = plt.axes([0.68, 0.04, 0.07, 0.04])
        ax_view_front = plt.axes([0.76, 0.04, 0.07, 0.04])
        ax_view_3d = plt.axes([0.84, 0.04, 0.07, 0.04])
        
        btn_view_top = Button(ax_view_top, 'Top')
        btn_view_side = Button(ax_view_side, 'Side')
        btn_view_front = Button(ax_view_front, 'Front')
        btn_view_3d = Button(ax_view_3d, '3D')
        
        def view_top(event):
            self.ax_3d.view_init(elev=90, azim=-90)
            self.fig.canvas.draw_idle()
        
        def view_side(event):
            self.ax_3d.view_init(elev=0, azim=0)
            self.fig.canvas.draw_idle()
        
        def view_front(event):
            self.ax_3d.view_init(elev=0, azim=-90)
            self.fig.canvas.draw_idle()
        
        def view_3d(event):
            self.ax_3d.view_init(elev=20, azim=45)
            self.fig.canvas.draw_idle()
        
        btn_view_top.on_clicked(view_top)
        btn_view_side.on_clicked(view_side)
        btn_view_front.on_clicked(view_front)
        btn_view_3d.on_clicked(view_3d)
        
        # Add zoom reset button
        ax_zoom_reset = plt.axes([0.92, 0.04, 0.06, 0.04])
        btn_zoom_reset = Button(ax_zoom_reset, 'Reset\nZoom')
        
        def reset_zoom(event):
            self.zoom_level = 1.0
            self.update_display()
        
        btn_zoom_reset.on_clicked(reset_zoom)
        
        plt.show()


def first_existing(candidates):
    for candidate in candidates:
        candidate = Path(candidate).expanduser()
        if candidate.exists():
            return candidate.resolve()
    return None


def search_project(project_path, names):
    direct = []
    for name in names:
        direct.extend([
            project_path / "data" / "processed" / name,
            project_path / "outputs" / name,
            project_path / "data" / "ccf" / name,
            project_path / name,
        ])
    found = first_existing(direct)
    if found:
        return found
    for name in names:
        matches = sorted(project_path.rglob(name))
        if matches:
            return matches[0].resolve()
    return None


def resolve_project(project_arg):
    if resolve_project_path is not None:
        return resolve_project_path(project_arg)
    if project_arg:
        project = Path(project_arg).expanduser().resolve()
        print(f"Project path: {project}")
        return project
    project = Path.cwd().expanduser().resolve()
    print(f"Project path: {project} (current folder)")
    return project


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive 3D fiber visualizer.")
    parser.add_argument(
        "project_path",
        nargs="?",
        help="Project folder. Defaults to the current project folder.",
    )
    parser.add_argument(
        "--image",
        help="Path to registered microCT image. Defaults to data/processed/microct_registered.tif.",
    )
    parser.add_argument(
        "--fibers",
        help="Path to fiber_data.json. Defaults to outputs/fiber_data.json.",
    )
    parser.add_argument(
        "--ccf",
        help="Path to Allen CCF template/annotation. Defaults to project data/ccf or shared resources.",
    )
    return parser.parse_args()


def main():
    """Main function"""
    print("="*70)
    print("INTERACTIVE 3D FIBER VISUALIZER")
    print("="*70)

    args = parse_args()
    PROJECT_PATH = resolve_project(args.project_path)
    
    # File paths
    image_path = (
        Path(args.image).expanduser().resolve()
        if args.image else search_project(
            PROJECT_PATH,
            ("microct_registered.tif", "microct_registered.tiff"),
        )
    )
    fiber_data_path = (
        Path(args.fibers).expanduser().resolve()
        if args.fibers else search_project(PROJECT_PATH, ("fiber_data.json",))
    )
    
    # Allen CCF options - try in order of preference
    resource_dir = Path(__file__).resolve().parent.parent / "resources" / "allen_ccf"
    ccf_candidates = [
        ("Allen CCF Template (25um)", PROJECT_PATH / "data" / "ccf" / "average_template_25.nrrd"),
        ("Allen CCF Template (25um shared)", resource_dir / "average_template_25.nrrd"),
        ("Allen CCF Template (10um)", PROJECT_PATH / "data" / "ccf" / "average_template_10.nrrd"),
        ("Allen CCF Annotation (25um)", PROJECT_PATH / "data" / "ccf" / "annotation_25.nrrd"),
        ("Allen CCF Annotation (25um shared)", resource_dir / "annotation_25.nrrd"),
        ("Allen CCF Annotation (10um)", PROJECT_PATH / "data" / "ccf" / "annotation_10.nrrd"),
    ]
    
    if args.ccf:
        ccf_path = Path(args.ccf).expanduser().resolve()
        if not ccf_path.exists():
            print(f"\n❌ ERROR: Allen CCF file not found: {ccf_path}")
            return
        print(f"\n✓ Using Allen CCF: {ccf_path}")
    else:
        ccf_path = None
        for name, path in ccf_candidates:
            if path.exists():
                ccf_path = path.resolve()
                print(f"\n✓ Found {name}: {path}")
                break
    
    if ccf_path is None:
        print(f"\n⚠️  Allen CCF not found (will use microCT only)")
        print(f"\nTo download Allen CCF template (25um, ~200MB):")
        print(f"  URL: http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/average_template/average_template_25.nrrd")
        print(f"  Save to: {PROJECT_PATH / 'data' / 'ccf' / 'average_template_25.nrrd'}")
        print(f"\nThe template (MRI-like) is better for visualization than annotation.")
    
    # Check required files exist
    if image_path is None or not image_path.exists():
        print(f"\n❌ ERROR: Image not found: {image_path}")
        print("Expected a registered microCT image such as:")
        print(f"  {PROJECT_PATH / 'data' / 'processed' / 'microct_registered.tif'}")
        print("Or pass it explicitly with --image.")
        return
    
    if fiber_data_path is None or not fiber_data_path.exists():
        print(f"\n❌ ERROR: Fiber data not found: {fiber_data_path}")
        print("Run fiber_tracker.py first to track fibers, or pass an existing file with --fibers.")
        print(f"Expected: {PROJECT_PATH / 'outputs' / 'fiber_data.json'}")
        return
    
    # Create visualizer
    visualizer = FiberVisualizer3D(
        image_path=str(image_path),
        fiber_data_path=str(fiber_data_path),
        ccf_path=str(ccf_path) if ccf_path else None
    )
    
    # Start interactive session
    visualizer.start()


if __name__ == "__main__":
    main()
