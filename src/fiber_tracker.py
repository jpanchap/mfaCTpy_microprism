"""
MicroCT Fiber Tracking Tool with Allen CCF Integration - FIXED VERSION

Interactive tool for manually tracking optical fibers in registered microCT mouse brain images
with automatic brain region identification using Allen CCF.

Dependencies:
pip install numpy tifffile matplotlib pandas nrrd requests SimpleITK
"""

import sys
from project_paths import resolve_project_path
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import pandas as pd
from pathlib import Path
import json
import requests
import SimpleITK as sitk


class AllenCCFOntology:
    """Handler for Allen CCF structure tree and region lookups"""
    
    def __init__(self, ontology_path=None):
        """Initialize Allen CCF ontology"""
        self.id_to_info = {}
        self.acronym_to_id = {}
        self.name_to_id = {}
        
        if ontology_path and Path(ontology_path).exists():
            self.load_ontology_from_file(ontology_path)
        else:
            self.download_ontology()
    
    def download_ontology(self):
        """Download Allen CCF ontology from Allen Institute API"""
        print("Downloading Allen CCF ontology...")
        url = "http://api.brain-map.org/api/v2/structure_graph_download/1.json"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            self.parse_ontology(data)
            print(">>> Ontology downloaded successfully")
        except Exception as e:
            print(f"Warning: Could not download ontology: {e}")
    
    def load_ontology_from_file(self, path):
        """Load ontology from local JSON file"""
        print(f"Loading ontology from {path}...")
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self.parse_ontology(data)
            print(">>> Ontology loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load ontology: {e}")
    
    def parse_ontology(self, data):
        """Parse ontology JSON data - handles hierarchical structure tree"""
        
        def parse_structure_tree(node):
            """Recursively parse the structure tree"""
            if isinstance(node, dict):
                if 'id' in node:
                    struct_id = node['id']
                    self.id_to_info[struct_id] = {
                        'id': struct_id,
                        'name': node.get('name', f'Region_{struct_id}'),
                        'acronym': node.get('acronym', ''),
                        'parent_id': node.get('parent_structure_id'),
                        'color_hex': node.get('color_hex_triplet', 'FFFFFF'),
                        'depth': node.get('depth', 0)
                    }
                    
                    acronym = node.get('acronym', '')
                    if acronym:
                        self.acronym_to_id[acronym] = struct_id
                    
                    name = node.get('name', '')
                    if name:
                        self.name_to_id[name] = struct_id
                
                # Recursively process children
                if 'children' in node:
                    for child in node['children']:
                        parse_structure_tree(child)
                        
            elif isinstance(node, list):
                for item in node:
                    parse_structure_tree(item)
        
        # Start parsing from the msg field or root
        if 'msg' in data:
            parse_structure_tree(data['msg'])
        else:
            parse_structure_tree(data)
    
    def get_region_info(self, region_id):
        """Get information about a brain region"""
        if region_id == 0:
            return {
                'id': 0,
                'name': 'Outside Brain',
                'acronym': 'OUT',
                'parent_id': None,
                'color_hex': '000000'
            }
        
        return self.id_to_info.get(region_id, {
            'id': region_id,
            'name': f'Unknown_Region_{region_id}',
            'acronym': f'UNK{region_id}',
            'parent_id': None,
            'color_hex': 'FFFFFF'
        })
    
    def get_region_hierarchy(self, region_id, max_depth=10):
        """Get hierarchical path from region to root"""
        hierarchy = []
        current_id = region_id
        depth = 0
        
        while current_id is not None and depth < max_depth:
            info = self.get_region_info(current_id)
            hierarchy.append(info['name'])
            current_id = info.get('parent_id')
            depth += 1
            
            if current_id == 997:  # root
                break
        
        return hierarchy


class FiberTracker:
    """Interactive fiber tracking tool with CCF integration"""
    
    def __init__(self, microct_image_path, ccf_annotation_path=None, 
                 transform_path=None, ontology_path=None, output_dir=None,
                 spacing=0.025, use_registered=False):
        """Initialize fiber tracker
        
        Parameters:
        -----------
        microct_image_path : str
            Path to microCT image (aligned or registered)
        ccf_annotation_path : str
            Path to CCF annotation volume
        transform_path : str
            Path to registration transform file
        ontology_path : str
            Path to CCF ontology JSON
        output_dir : str
            Output directory
        spacing : float
            Voxel spacing in mm (default: 0.025 for 25um)
        use_registered : bool
            True if using microct_registered.tif (already in CCF space, no transform needed)
            False if using microct_aligned.tif (needs transform to map to CCF)
        """
        print("="*70)
        print("FIBER TRACKER - INITIALIZATION")
        print("="*70)
        
        print(f"\nLoading microCT image...")
        self.image = tifffile.imread(microct_image_path)
        print(f">>> Image loaded. Shape: {self.image.shape}")
        
        self.spacing = spacing
        self.use_registered = use_registered
        print(f">>> Voxel spacing: {spacing} mm ({spacing*1000} um)")
        print(f">>> Image type: {'REGISTERED (in CCF space)' if use_registered else 'ALIGNED (original space)'}")
        
        print("\nInitializing Allen CCF ontology...")
        self.ontology = AllenCCFOntology(ontology_path)
        
        self.transform = None
        if transform_path and not use_registered:
            self.load_transform(transform_path)
        elif use_registered:
            print("\n>>> Using registered image - no transform needed")
        
        self.ccf_annotation = None
        if ccf_annotation_path:
            self.load_ccf_annotation(ccf_annotation_path)
        
        if output_dir is None:
            output_dir = Path(registered_image_path).parent.parent / "outputs"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.nz, self.ny, self.nx = self.image.shape
        self.current_slice = self.nz // 2
        
        self.fibers = []
        self.current_fiber = {'top': None, 'bottom': None}
        self.current_points = []
        
        self.fiber_colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        self.fig = None
        self.ax = None
        self.im = None
        self.slice_slider = None
        self.info_text = None
        
        self.check_previous_work()
        
        print("\n>>> Initialization complete!")
        print("="*70)
    
    def load_transform(self, transform_path):
        """Load registration transform (microCT aligned -> CCF)"""
        print(f"\nLoading registration transform from: {transform_path}")
        try:
            self.transform = sitk.ReadTransform(str(transform_path))
            print(f">>> Transform loaded: {self.transform.GetName()}")
            print(">>> This transform maps: microCT_aligned -> CCF")
        except Exception as e:
            print(f"ERROR: Could not load transform: {e}")
            self.transform = None
    
    def load_ccf_annotation(self, ccf_annotation_path):
        """Load Allen CCF annotation volume"""
        print(f"\nLoading CCF annotation from: {ccf_annotation_path}")
        path = Path(ccf_annotation_path)
        
        try:
            if path.suffix == '.nrrd':
                import nrrd
                self.ccf_annotation, header = nrrd.read(str(path))
                print(f">>> CCF annotation loaded. Shape: {self.ccf_annotation.shape}")
                # Check if we need to transpose (Allen CCF is typically in different order)
                print(f"    Header info: {header.get('space', 'unknown')}")
            elif path.suffix in ['.tif', '.tiff']:
                self.ccf_annotation = tifffile.imread(str(path))
                print(f">>> CCF annotation loaded. Shape: {self.ccf_annotation.shape}")
            else:
                raise ValueError(f"Unsupported format: {path.suffix}")
            
        except Exception as e:
            print(f"ERROR: Could not load CCF annotation: {e}")
            self.ccf_annotation = None
    
    def check_previous_work(self):
        """Check if previous tracking work exists and ask user"""
        # Check for both progress file and final outputs
        progress_file = self.output_dir / 'fiber_tracking_progress.json'
        final_data_file = self.output_dir / 'fiber_data.json'
        final_csv_file = self.output_dir / 'fiber_report.csv'
        
        # Determine which file to load from
        load_file = None
        file_type = None
        
        if final_data_file.exists() or final_csv_file.exists():
            # Final output exists - prefer this over progress
            load_file = final_data_file if final_data_file.exists() else None
            file_type = "final"
        elif progress_file.exists():
            # Only progress file exists
            load_file = progress_file
            file_type = "progress"
        
        if load_file is None:
            return  # No previous work
        
        print("\n" + "="*70)
        print("PREVIOUS WORK DETECTED")
        print("="*70)
        
        try:
            with open(load_file, 'r') as f:
                saved_data = json.load(f)
            
            num_fibers = len(saved_data)
            print(f"\nFound {num_fibers} previously tracked fiber(s)")
            print(f"Source: {load_file}")
            
            if file_type == "final":
                print("Type: FINAL OUTPUT (completed session)")
            else:
                print("Type: PROGRESS SAVE (incomplete session)")
            
            # Show summary of previous fibers
            if num_fibers > 0:
                print("\nPrevious fibers:")
                for fiber_dict in saved_data[:5]:  # Show first 5
                    fid = fiber_dict['fiber_id']
                    region = fiber_dict.get('region_name', 'N/A')
                    acronym = fiber_dict.get('region_acronym', 'N/A')
                    if acronym != 'N/A':
                        print(f"  Fiber {fid}: {acronym} - {region}")
                    else:
                        print(f"  Fiber {fid}: {region}")
                if num_fibers > 5:
                    print(f"  ... and {num_fibers - 5} more")
            
            print("\n" + "="*70)
            print("OPTIONS:")
            print("="*70)
            print("  [c] Continue - Load previous fibers and add/edit more")
            print("  [r] Replace - Start fresh (previous work will be overwritten)")
            print("  [q] Quit - Exit without loading")
            print("="*70)
            
            while True:
                response = input("\nYour choice (c/r/q): ").lower()
                if response in ['c', 'r', 'q']:
                    break
                print("Invalid choice. Please enter 'c', 'r', or 'q'")
            
            if response == 'c':
                self.load_previous_work(saved_data)
                print(f"\n>>> Loaded {len(self.fibers)} fiber(s)")
                print(">>> You can continue adding more fibers or use 'Undo Last Fiber' to modify")
            elif response == 'r':
                print("\n>>> Starting fresh")
                print(">>> Previous work will be overwritten when you save")
            else:  # 'q'
                print("\n>>> Exiting...")
                import sys
                sys.exit(0)
                
        except Exception as e:
            print(f"\nERROR: Could not load previous work: {e}")
            print("Starting fresh...")
        
        print("="*70)
    
    def load_previous_work(self, saved_data):
        """Load previously tracked fibers"""
        for fiber_dict in saved_data:
            fiber = {
                'id': fiber_dict['fiber_id'],
                'top': (fiber_dict['top_z'], fiber_dict['top_y'], fiber_dict['top_x']),
                'bottom': (fiber_dict['bottom_z'], fiber_dict['bottom_y'], fiber_dict['bottom_x']),
                'color': self.fiber_colors[fiber_dict['fiber_id'] % len(self.fiber_colors)].tolist(),
                'region_id': fiber_dict.get('region_id'),
                'region_name': fiber_dict.get('region_name', 'N/A'),
                'region_acronym': fiber_dict.get('region_acronym', 'N/A'),
                'region_hierarchy': fiber_dict.get('region_hierarchy', []),
                'ccf_coords': fiber_dict.get('ccf_coords')
            }
            self.fibers.append(fiber)
    
    def transform_point_to_ccf(self, z, y, x):
        """Transform point from microCT space to CCF annotation space
        
        CRITICAL LOGIC:
        - If using microct_registered.tif: Already in CCF space, use coords directly
        - If using microct_aligned.tif: Need to apply transform to map to CCF
        
        Parameters:
        -----------
        z, y, x : int
            Voxel coordinates in microCT image
            
        Returns:
        --------
        tuple : (z_ccf, y_ccf, x_ccf) in CCF annotation space
        """
        
        if self.use_registered:
            # Image is already in CCF space - direct 1:1 mapping
            print(f"  ✓ Direct mapping (registered→CCF): ({z},{y},{x})")
            return (z, y, x)
        
        # Using aligned image - need transform
        if self.transform is None:
            print(f"  ⚠️  WARNING: No transform available, using direct mapping (may be wrong!)")
            return (z, y, x)
        
        try:
            # Convert voxel to physical coordinates
            # SimpleITK uses (x, y, z) order for physical coordinates
            point_physical = [x * self.spacing, y * self.spacing, z * self.spacing]
            
            print(f"  Voxel→Physical: ({point_physical[0]:.3f}, {point_physical[1]:.3f}, {point_physical[2]:.3f}) mm")
            
            # Apply FORWARD transform: microCT_aligned → CCF
            ccf_physical = self.transform.TransformPoint(point_physical)
            
            print(f"  Transform→CCF Physical: ({ccf_physical[0]:.3f}, {ccf_physical[1]:.3f}, {ccf_physical[2]:.3f}) mm")
            
            # Convert back to voxel coordinates
            x_ccf = int(round(ccf_physical[0] / self.spacing))
            y_ccf = int(round(ccf_physical[1] / self.spacing))
            z_ccf = int(round(ccf_physical[2] / self.spacing))
            
            print(f"  ✓ Final CCF voxel: ({z_ccf},{y_ccf},{x_ccf})")
            
            return (z_ccf, y_ccf, x_ccf)
            
        except Exception as e:
            print(f"  ❌ ERROR in transform: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: assume direct correspondence
            return (z, y, x)
    
    def get_region_at_point(self, z, y, x):
        """Get brain region at point in microCT space
        
        This is the critical function that maps fiber coordinates to brain regions.
        """
        if self.ccf_annotation is None:
            return {
                'id': None,
                'name': 'N/A (No CCF annotation loaded)',
                'acronym': 'N/A',
                'hierarchy': [],
                'ccf_coords': None
            }
        
        print(f"\n  === REGION LOOKUP DEBUG ===")
        print(f"  Input (microCT space): ({z}, {y}, {x})")
        print(f"  MicroCT image shape: {self.image.shape}")
        
        # Transform to CCF space
        z_ccf, y_ccf, x_ccf = self.transform_point_to_ccf(z, y, x)
        
        # Validate bounds
        ccf_shape = self.ccf_annotation.shape
        print(f"  CCF annotation shape: {ccf_shape}")
        print(f"  Checking bounds...")
        
        if (z_ccf < 0 or z_ccf >= ccf_shape[0] or 
            y_ccf < 0 or y_ccf >= ccf_shape[1] or 
            x_ccf < 0 or x_ccf >= ccf_shape[2]):
            print(f"  ❌ OUT OF BOUNDS!")
            print(f"     CCF coords: ({z_ccf}, {y_ccf}, {x_ccf})")
            print(f"     CCF shape:  {ccf_shape}")
            return {
                'id': -1,
                'name': 'Out of Bounds',
                'acronym': 'OOB',
                'hierarchy': [],
                'ccf_coords': (z_ccf, y_ccf, x_ccf)
            }
        
        print(f"  ✓ Within bounds")
        
        try:
            # Look up region ID in annotation
            region_id = int(self.ccf_annotation[z_ccf, y_ccf, x_ccf])
            print(f"  Region ID from annotation: {region_id}")
            
            if region_id == 0:
                print(f"  ⚠️  Region ID is 0 (outside brain or background)")
                return {
                    'id': 0,
                    'name': 'Outside Brain',
                    'acronym': 'OUT',
                    'hierarchy': [],
                    'ccf_coords': (z_ccf, y_ccf, x_ccf)
                }
            
            # Get region information from ontology
            region_info = self.ontology.get_region_info(region_id)
            hierarchy = self.ontology.get_region_hierarchy(region_id)
            
            print(f"  ✓ Region found: {region_info['name']} ({region_info['acronym']})")
            print(f"  === END DEBUG ===\n")
            
            return {
                'id': region_id,
                'name': region_info['name'],
                'acronym': region_info['acronym'],
                'hierarchy': hierarchy,
                'color_hex': region_info.get('color_hex', 'FFFFFF'),
                'ccf_coords': (z_ccf, y_ccf, x_ccf)
            }
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            print(f"  === END DEBUG ===\n")
            return {
                'id': -1,
                'name': 'Error',
                'acronym': 'ERR',
                'hierarchy': [],
                'ccf_coords': (z_ccf, y_ccf, x_ccf)
            }
    
    def start(self):
        """Start interactive session"""
        print("\n" + "="*70)
        print("FIBER TRACKING SESSION")
        print("="*70)
        print("\nInstructions:")
        print("1. Navigate slices with slider/scroll")
        print("2. Click to mark fiber TOP (entry)")
        print("3. Navigate to tip")
        print("4. Click to mark fiber BOTTOM (tip)")
        print("5. 'Next Fiber' to save and start next")
        print("6. 'Undo Current' removes current point")
        print("7. 'Undo Last Fiber' removes last saved fiber")
        print("8. 'Save Progress' saves without exiting")
        print("9. 'Finish' when done")
        print("\nKeyboard shortcuts:")
        print("  u: Undo current point")
        print("  U: Undo last fiber (Shift+U)")
        print("  n: Next fiber")
        print("  s: Save progress")
        print("  Enter: Finish")
        print("="*70 + "\n")
        
        self.fig = plt.figure(figsize=(16, 10))
        self.ax = plt.axes([0.1, 0.3, 0.8, 0.6])
        
        self.im = self.ax.imshow(self.image[self.current_slice, :, :], 
                                  cmap='gray', origin='upper')
        self.update_title()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        
        plt.colorbar(self.im, ax=self.ax, fraction=0.046)
        
        ax_slider = plt.axes([0.2, 0.20, 0.6, 0.03])
        self.slice_slider = Slider(ax_slider, 'Slice (Z)', 0, self.nz-1, 
                                   valinit=self.current_slice, valstep=1)
        self.slice_slider.on_changed(self.update_slice)
        
        ax_next = plt.axes([0.10, 0.10, 0.12, 0.05])
        ax_save = plt.axes([0.24, 0.10, 0.12, 0.05])
        ax_undo_current = plt.axes([0.38, 0.10, 0.12, 0.05])
        ax_undo_last = plt.axes([0.52, 0.10, 0.12, 0.05])
        ax_done = plt.axes([0.66, 0.10, 0.12, 0.05])
        
        self.btn_next = Button(ax_next, 'Next Fiber', color='lightblue')
        self.btn_save = Button(ax_save, 'Save Progress', color='lightgreen')
        self.btn_undo_current = Button(ax_undo_current, 'Undo Current', color='lightyellow')
        self.btn_undo_last = Button(ax_undo_last, 'Undo Last Fiber', color='lightcoral')
        self.btn_done = Button(ax_done, 'Finish', color='lightgray')
        
        self.btn_next.on_clicked(self.next_fiber)
        self.btn_save.on_clicked(self.save_progress)
        self.btn_undo_current.on_clicked(self.undo_current)
        self.btn_undo_last.on_clicked(self.undo_last_fiber)
        self.btn_done.on_clicked(self.finish_tracking)
        
        ax_info = plt.axes([0.1, 0.01, 0.8, 0.06])
        ax_info.axis('off')
        self.info_text = ax_info.text(0.5, 0.5, self.get_info_text(), 
                                     ha='center', va='center', fontsize=10,
                                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        self.redraw_all_fibers()
        plt.show()
    
    def update_title(self):
        """Update title"""
        status = " | TOP marked" if self.current_fiber['top'] else ""
        self.ax.set_title(f'Slice: {self.current_slice}/{self.nz-1} | '
                         f'Fibers: {len(self.fibers)}{status}', fontweight='bold')
    
    def get_info_text(self):
        """Get info text"""
        if self.current_fiber['top'] is None:
            return f"Fibers: {len(self.fibers)} | Click for FIBER TOP"
        else:
            return f"Fibers: {len(self.fibers)} | Click for FIBER BOTTOM (tip)"
    
    def update_slice(self, val):
        """Update slice"""
        self.current_slice = int(self.slice_slider.val)
        self.im.set_data(self.image[self.current_slice, :, :])
        
        slice_data = self.image[self.current_slice, :, :]
        if slice_data.max() > 0:
            vmin, vmax = np.percentile(slice_data[slice_data > 0], [1, 99])
            self.im.set_clim(vmin, vmax)
        
        self.update_title()
        self.redraw_all_fibers()
        self.fig.canvas.draw_idle()
    
    def on_scroll(self, event):
        """Handle scroll"""
        if event.inaxes != self.ax:
            return
        
        if event.button == 'up':
            new_slice = min(self.current_slice + 1, self.nz - 1)
        elif event.button == 'down':
            new_slice = max(self.current_slice - 1, 0)
        else:
            return
        
        if new_slice != self.current_slice:
            self.slice_slider.set_val(new_slice)
    
    def on_click(self, event):
        """Handle click"""
        if event.inaxes != self.ax or event.xdata is None:
            return
        
        x, y = int(round(event.xdata)), int(round(event.ydata))
        z = self.current_slice
        
        if x < 0 or x >= self.nx or y < 0 or y >= self.ny:
            return
        
        if self.current_fiber['top'] is None:
            self.current_fiber['top'] = (z, y, x)
            print(f"\nFiber {len(self.fibers)}: TOP at ({z},{y},{x})")
            
            marker = self.ax.plot(x, y, 'r+', markersize=20, markeredgewidth=3)[0]
            self.current_points.append(marker)
            
            self.update_title()
            self.info_text.set_text(self.get_info_text())
            self.fig.canvas.draw_idle()
            
        elif self.current_fiber['bottom'] is None:
            self.current_fiber['bottom'] = (z, y, x)
            print(f"Fiber {len(self.fibers)}: BOTTOM at ({z},{y},{x})")
            
            # Get region information with detailed debugging
            print(f"\nDEBUG: Getting region for point ({z},{y},{x})...")
            region_info = self.get_region_at_point(z, y, x)
            self.current_fiber['region_info'] = region_info
            
            print(f">>> Region: {region_info['name']} ({region_info['acronym']})")
            if region_info.get('id'):
                print(f"    ID: {region_info['id']}")
            if region_info.get('ccf_coords'):
                print(f"    CCF coords: {region_info['ccf_coords']}")
            if region_info['hierarchy']:
                print(f"    Path: {' > '.join(region_info['hierarchy'][:3])}")
            
            marker = self.ax.plot(x, y, 'bx', markersize=20, markeredgewidth=3)[0]
            self.current_points.append(marker)
            
            top = self.current_fiber['top']
            fiber_id = len(self.fibers)
            color = self.fiber_colors[fiber_id % len(self.fiber_colors)]
            
            if top[0] == z:
                line = self.ax.plot([top[2], x], [top[1], y], 
                           color=color, linewidth=3, alpha=0.7)[0]
                self.current_points.append(line)
            
            self.update_title()
            self.info_text.set_text(self.get_info_text())
            self.fig.canvas.draw_idle()
            
            print(f">>> Click 'Next Fiber' to save\n")
    
    def undo_current(self, event):
        """Undo current fiber being labeled (in progress)"""
        if self.current_fiber['bottom']:
            self.current_fiber['bottom'] = None
            self.current_fiber.pop('region_info', None)
            print("UNDO CURRENT: Bottom point removed")
        elif self.current_fiber['top']:
            self.current_fiber['top'] = None
            print("UNDO CURRENT: Top point removed")
        else:
            print("Nothing to undo - no current fiber in progress")
            return
        
        for marker in self.current_points:
            marker.remove()
        self.current_points.clear()
        
        if self.current_fiber['top']:
            z, y, x = self.current_fiber['top']
            if z == self.current_slice:
                marker = self.ax.plot(x, y, 'r+', markersize=20, markeredgewidth=3)[0]
                self.current_points.append(marker)
        
        self.update_title()
        self.info_text.set_text(self.get_info_text())
        self.fig.canvas.draw_idle()
    
    def undo_last_fiber(self, event):
        """Undo the last saved fiber"""
        if len(self.fibers) == 0:
            print("No saved fibers to undo")
            return
        
        removed = self.fibers.pop()
        print(f"UNDO LAST FIBER: Removed Fiber {removed['id']} ({removed.get('region_name', 'N/A')})")
        
        self.redraw_all_fibers()
        self.update_title()
        self.fig.canvas.draw_idle()
    
    def next_fiber(self, event):
        """Save fiber"""
        if not self.current_fiber['top'] or not self.current_fiber['bottom']:
            print("ERROR: Mark both TOP and BOTTOM first")
            return
        
        region_info = self.current_fiber.get('region_info', {
            'id': None, 'name': 'N/A', 'acronym': 'N/A', 
            'hierarchy': [], 'ccf_coords': None
        })
        
        fiber_data = {
            'id': len(self.fibers),
            'top': self.current_fiber['top'],
            'bottom': self.current_fiber['bottom'],
            'color': self.fiber_colors[len(self.fibers) % len(self.fiber_colors)].tolist(),
            'region_id': region_info['id'],
            'region_name': region_info['name'],
            'region_acronym': region_info['acronym'],
            'region_hierarchy': region_info.get('hierarchy', []),
            'ccf_coords': region_info.get('ccf_coords')
        }
        
        self.fibers.append(fiber_data)
        print(f">>> Fiber {fiber_data['id']} SAVED: {region_info['name']}\n")
        
        for marker in self.current_points:
            marker.remove()
        self.current_points.clear()
        
        self.current_fiber = {'top': None, 'bottom': None}
        self.redraw_all_fibers()
        self.update_title()
        self.info_text.set_text(self.get_info_text())
        self.fig.canvas.draw_idle()
    
    def redraw_all_fibers(self):
        """Redraw fibers"""
        for artist in self.ax.lines[:]:
            if artist not in self.current_points:
                artist.remove()
        for artist in self.ax.texts[:]:
            artist.remove()
        
        for fiber in self.fibers:
            color = fiber['color']
            top, bottom = fiber['top'], fiber['bottom']
            
            if top[0] == self.current_slice:
                self.ax.plot(top[2], top[1], 'o', color=color, markersize=12,
                           markeredgecolor='white', markeredgewidth=2)
            
            if bottom[0] == self.current_slice:
                self.ax.plot(bottom[2], bottom[1], 's', color=color, markersize=12,
                           markeredgecolor='white', markeredgewidth=2)
                
                acronym = fiber.get('region_acronym', '')
                label = f"F{fiber['id']}"
                if acronym and acronym != 'N/A':
                    label += f"\n{acronym}"
                
                self.ax.text(bottom[2]+5, bottom[1]+5, label, 
                           color='white', fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
            
            if top[0] == self.current_slice and bottom[0] == self.current_slice:
                self.ax.plot([top[2], bottom[2]], [top[1], bottom[1]], 
                           color=color, linewidth=3, alpha=0.7)
    
    def save_progress(self, event):
        """Save progress"""
        if len(self.fibers) == 0:
            print("No fibers yet")
            return
        
        progress_file = self.output_dir / 'fiber_tracking_progress.json'
        self.save_fiber_data(progress_file)
        print(f">>> Progress saved ({len(self.fibers)} fibers)")
    
    def finish_tracking(self, event):
        """Finish tracking"""
        if self.current_fiber['top'] and self.current_fiber['bottom']:
            response = input("Save unsaved fiber? (y/n): ")
            if response.lower() == 'y':
                self.next_fiber(None)
        
        if len(self.fibers) == 0:
            print("No fibers tracked")
            plt.close(self.fig)
            return
        
        print("\n" + "="*70)
        print("GENERATING OUTPUTS...")
        print("="*70)
        
        self.save_fiber_data(self.output_dir / 'fiber_data.json')
        self.generate_fiber_report(self.output_dir / 'fiber_report.csv')
        self.generate_summary_report(self.output_dir / 'fiber_summary.txt')
        self.generate_horizontal_view(self.output_dir / 'fiber_horizontal_view.png')
        
        if self.ccf_annotation is not None:
            self.generate_ccf_visualization(self.output_dir / 'fiber_ccf_overlay.png')
        
        print("\n" + "="*70)
        print(f"COMPLETE! {len(self.fibers)} fibers tracked")
        print(f"Outputs: {self.output_dir}")
        print("="*70)
        
        plt.close(self.fig)
    
    def save_fiber_data(self, output_path):
        """Save to JSON"""
        fiber_list = []
        for fiber in self.fibers:
            fiber_list.append({
                'fiber_id': fiber['id'],
                'top_z': fiber['top'][0],
                'top_y': fiber['top'][1],
                'top_x': fiber['top'][2],
                'bottom_z': fiber['bottom'][0],
                'bottom_y': fiber['bottom'][1],
                'bottom_x': fiber['bottom'][2],
                'region_id': fiber.get('region_id'),
                'region_name': fiber.get('region_name', 'N/A'),
                'region_acronym': fiber.get('region_acronym', 'N/A'),
                'region_hierarchy': fiber.get('region_hierarchy', []),
                'ccf_coords': fiber.get('ccf_coords')
            })
        
        with open(output_path, 'w') as f:
            json.dump(fiber_list, f, indent=2)
        print(f">>> Saved: {output_path}")
    
    def generate_fiber_report(self, output_path):
        """Generate CSV"""
        data = []
        for fiber in self.fibers:
            hierarchy = fiber.get('region_hierarchy', [])
            ccf_coords = fiber.get('ccf_coords')
            
            data.append({
                'Fiber_ID': fiber['id'],
                'Top_Z': fiber['top'][0],
                'Top_Y': fiber['top'][1],
                'Top_X': fiber['top'][2],
                'Bottom_Z': fiber['bottom'][0],
                'Bottom_Y': fiber['bottom'][1],
                'Bottom_X': fiber['bottom'][2],
                'CCF_Z': ccf_coords[0] if ccf_coords else 'N/A',
                'CCF_Y': ccf_coords[1] if ccf_coords else 'N/A',
                'CCF_X': ccf_coords[2] if ccf_coords else 'N/A',
                'Region_ID': fiber.get('region_id', 'N/A'),
                'Region_Name': fiber.get('region_name', 'N/A'),
                'Region_Acronym': fiber.get('region_acronym', 'N/A'),
                'Parent_Region': hierarchy[1] if len(hierarchy) > 1 else 'N/A',
                'Grandparent_Region': hierarchy[2] if len(hierarchy) > 2 else 'N/A'
            })
        
        pd.DataFrame(data).to_csv(output_path, index=False)
        print(f">>> Saved: {output_path}")
    
    def generate_summary_report(self, output_path):
        """Generate summary"""
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("FIBER TRACKING SUMMARY\n")
            f.write("="*70 + "\n\n")
            f.write(f"Total Fibers: {len(self.fibers)}\n\n")
            
            region_groups = {}
            for fiber in self.fibers:
                region = fiber.get('region_name', 'N/A')
                region_groups.setdefault(region, []).append(fiber)
            
            f.write("FIBERS BY REGION:\n" + "-"*70 + "\n")
            for region, fibers in sorted(region_groups.items()):
                f.write(f"{region}: {len(fibers)} fiber(s)\n")
                for fiber in fibers:
                    f.write(f"  - Fiber {fiber['id']}\n")
                f.write("\n")
            
            f.write("DETAILS:\n" + "-"*70 + "\n")
            for fiber in self.fibers:
                f.write(f"\nFiber {fiber['id']}:\n")
                f.write(f"  TOP: {fiber['top']}\n")
                f.write(f"  BOTTOM: {fiber['bottom']}\n")
                if fiber.get('ccf_coords'):
                    f.write(f"  CCF: {fiber['ccf_coords']}\n")
                f.write(f"  Region: {fiber.get('region_name', 'N/A')}\n")
                f.write(f"  Acronym: {fiber.get('region_acronym', 'N/A')}\n")
                hierarchy = fiber.get('region_hierarchy', [])
                if hierarchy:
                    f.write(f"  Hierarchy: {' > '.join(hierarchy[:5])}\n")
        
        print(f">>> Saved: {output_path}")
    
    def generate_horizontal_view(self, output_path):
        """Generate horizontal view - FIXED: Balanced aspect ratio, markers only"""
        if len(self.fibers) == 0:
            return
        
        # Find middle Y coordinate based on fiber locations
        all_y_coords = []
        for fiber in self.fibers:
            all_y_coords.append(fiber['top'][1])
            all_y_coords.append(fiber['bottom'][1])
        
        min_y = min(all_y_coords)
        max_y = max(all_y_coords)
        middle_y = (min_y + max_y) // 2
        
        print(f"Horizontal view: Y={middle_y} (range {min_y}-{max_y})")
        
        # Get coronal slice (constant Y, showing X-Z plane)
        coronal_slice = self.image[:, middle_y, :]
        
        # Calculate proper figure size to maintain aspect ratio
        # Image is (Z, X), so height/width ratio
        z_size, x_size = coronal_slice.shape
        aspect_ratio = z_size / x_size
        
        # Base width, calculate height to maintain aspect
        fig_width = 12
        fig_height = fig_width * aspect_ratio * 0.8  # 0.8 factor for nice display
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Display with correct aspect ratio
        ax.imshow(coronal_slice, cmap='gray', origin='lower', aspect='equal')
        ax.set_title(f'Fiber Locations - Coronal View (Y={middle_y})', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('X (Medial-Lateral)', fontsize=12)
        ax.set_ylabel('Z (Dorsal-Ventral)', fontsize=12)
        
        # Collect unique regions for legend
        region_summary = {}
        
        for fiber in self.fibers:
            color = fiber['color']
            top = fiber['top']
            bottom = fiber['bottom']
            
            y_min = min(top[1], bottom[1])
            y_max = max(top[1], bottom[1])
            
            # Check if fiber passes through this Y slice
            if y_min <= middle_y <= y_max:
                # Interpolate position at this Y slice
                if bottom[1] != top[1]:
                    t = (middle_y - top[1]) / (bottom[1] - top[1])
                else:
                    t = 0
                
                z = int(top[0] + t * (bottom[0] - top[0]))
                x = int(top[2] + t * (bottom[2] - top[2]))
                
                # Plot marker ONLY - no labels near markers
                ax.plot(x, z, 'o', color=color, markersize=8, 
                       markeredgecolor='white', markeredgewidth=1.5, zorder=10)
                
                # Track regions for summary
                region_name = fiber.get('region_name', 'N/A')
                acronym = fiber.get('region_acronym', 'N/A')
                if region_name not in region_summary:
                    region_summary[region_name] = {
                        'acronym': acronym,
                        'fibers': []
                    }
                region_summary[region_name]['fibers'].append(fiber['id'])
        
        # Add region summary in top-right corner
        if region_summary:
            summary_text = "Brain Regions:\n" + "-"*30 + "\n"
            for region, info in sorted(region_summary.items()):
                fiber_ids = ', '.join([f"F{fid}" for fid in info['fibers']])
                acronym = info['acronym']
                if acronym != 'N/A':
                    summary_text += f"{acronym}: {fiber_ids}\n"
                else:
                    summary_text += f"{region}: {fiber_ids}\n"
            
            ax.text(0.98, 0.98, summary_text, 
                   transform=ax.transAxes,
                   fontsize=10, 
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', 
                            alpha=0.9, edgecolor='black', linewidth=1.5),
                   family='monospace')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f">>> Saved: {output_path}")
        plt.close()
    
    def generate_ccf_visualization(self, output_path):
        """Generate CCF overlay - FIXED: Markers only, no labels"""
        slice_fibers = {}
        for fiber in self.fibers:
            z = fiber['bottom'][0]
            slice_fibers.setdefault(z, []).append(fiber)
        
        num_slices = len(slice_fibers)
        if num_slices == 0:
            return
        
        ncols = min(3, num_slices)
        nrows = (num_slices + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 7*nrows))
        if num_slices == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (slice_z, fibers_in_slice) in enumerate(sorted(slice_fibers.items())):
            ax = axes[idx]
            
            # Overlay CCF annotation and microCT
            ax.imshow(self.ccf_annotation[slice_z, :, :], cmap='nipy_spectral', 
                     alpha=0.5, origin='upper')
            ax.imshow(self.image[slice_z, :, :], cmap='gray', alpha=0.5, origin='upper')
            
            # Collect region information for this slice
            region_summary = {}
            
            for fiber in fibers_in_slice:
                y, x = fiber['bottom'][1], fiber['bottom'][2]
                color = fiber['color']
                region_name = fiber.get('region_name', 'N/A')
                acronym = fiber.get('region_acronym', 'N/A')
                
                # Plot marker ONLY - no labels
                ax.plot(x, y, 'o', color=color, markersize=10, 
                       markeredgecolor='white', markeredgewidth=2, zorder=10)
                
                # Track for summary
                if region_name not in region_summary:
                    region_summary[region_name] = {
                        'acronym': acronym,
                        'fibers': []
                    }
                region_summary[region_name]['fibers'].append(fiber['id'])
            
            # Add region summary in top-right
            if region_summary:
                summary_text = "Regions (Slice {}):\n".format(slice_z) + "-"*25 + "\n"
                for region, info in sorted(region_summary.items()):
                    fiber_ids = ', '.join([f"F{fid}" for fid in info['fibers']])
                    acronym = info['acronym']
                    if acronym != 'N/A':
                        summary_text += f"{acronym}: {fiber_ids}\n"
                    else:
                        region_short = region[:20] + '...' if len(region) > 20 else region
                        summary_text += f"{region_short}: {fiber_ids}\n"
                
                ax.text(0.98, 0.98, summary_text,
                       transform=ax.transAxes,
                       fontsize=9,
                       verticalalignment='top',
                       horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white',
                                alpha=0.95, edgecolor='black', linewidth=1.5),
                       family='monospace')
            
            ax.set_title(f"Fiber Tips - Slice Z={slice_z}", 
                        fontsize=12, fontweight='bold', pad=10)
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(num_slices, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f">>> Saved: {output_path}")
        plt.close()


def download_ccf_annotation(output_dir, resolution=25):
    """Download Allen CCF annotation"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    output_file = output_dir / f"annotation_{resolution}.nrrd"
    
    if output_file.exists():
        print(f"Annotation exists: {output_file}")
        return str(output_file)
    
    print(f"\nDownloading CCF annotation ({resolution}um)...")
    print("This may take several minutes...")
    
    url = f"http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_{resolution}.nrrd"
    
    try:
        import urllib.request
        urllib.request.urlretrieve(url, output_file)
        print(f">>> Complete: {output_file}")
        return str(output_file)
    except Exception as e:
        print(f"ERROR: {e}")
        print(f"\nManual download: {url}")
        print(f"Save to: {output_file}")
        return None


def main():
    """Main function"""
    print("="*70)
    print("MICROCT FIBER TRACKER WITH CCF")
    print("="*70)
    
    PROJECT_PATH = resolve_project_path(sys.argv[1] if len(sys.argv) > 1 else None)
    
    # File paths
    aligned_image = PROJECT_PATH / "data" / "processed" / "microct_aligned.tif"
    registered_image = PROJECT_PATH / "data" / "processed" / "microct_registered.tif"
    transform_file = PROJECT_PATH / "data" / "processed" / "transform_landmark.tfm"
    ccf_annotation = PROJECT_PATH / "data" / "ccf" / "annotation_25.nrrd"
    # ccf_annotation = PROJECT_PATH / "data" / "ccf" / "average_template_25.nrrd"
    ontology_file = PROJECT_PATH / "data" / "ccf" / "structure_tree.json"
    output_dir = PROJECT_PATH / "outputs"
    
    # Strategy selection
    print("\n" + "="*70)
    print("IMAGE SELECTION STRATEGY")
    print("="*70)
    
    use_registered = False
    microct_image = None
    
    # RECOMMENDED: Use registered image with direct mapping
    if registered_image.exists():
        print("\n✓ Found microct_registered.tif")
        print("\nRECOMMENDED APPROACH:")
        print("  Use microct_registered.tif with DIRECT coordinate mapping")
        print("  Why?")
        print("    - Already transformed into CCF space during registration")
        print("    - Coordinates should match CCF annotation 1:1")
        print("    - No additional transform needed")
        print("    - Simpler and more accurate")
        
        if aligned_image.exists():
            print("\nAlternative: microct_aligned.tif + transform")
            print("  (More complex, requires correct transform application)")
            
            response = input("\nUse microct_registered.tif (recommended)? (y/n): ")
            if response.lower() != 'n':
                microct_image = registered_image
                use_registered = True
            else:
                microct_image = aligned_image
                use_registered = False
        else:
            microct_image = registered_image
            use_registered = True
            print("\n>>> Using microct_registered.tif")
            
    elif aligned_image.exists():
        print("\n✓ Found microct_aligned.tif (will use with transform)")
        microct_image = aligned_image
        use_registered = False
        
    else:
        print("\n❌ ERROR: No microCT image found!")
        print(f"  Expected: {aligned_image}")
        print(f"       or: {registered_image}")
        return
    
    print(f"\n>>> Selected: {microct_image.name}")
    print(f">>> Mode: {'REGISTERED (direct coords)' if use_registered else 'ALIGNED (with transform)'}")
    
    # Check transform (only needed for aligned image)
    if not use_registered:
        if not transform_file.exists():
            print(f"\n❌ ERROR: Transform not found: {transform_file}")
            refined = PROJECT_PATH / "data" / "processed" / "transform_refined.tfm"
            if refined.exists():
                transform_file = refined
                print(f"✓ Using refined transform: {transform_file}")
            else:
                print("\n❌ Cannot proceed without transform for aligned image")
                response = input("\nSwitch to registered image instead? (y/n): ")
                if response.lower() == 'y' and registered_image.exists():
                    microct_image = registered_image
                    use_registered = True
                    transform_file = None
                    print("✓ Switched to registered image mode")
                else:
                    return
    else:
        print("\n✓ Using registered image - no transform needed")
        transform_file = None
    
    # Check CCF annotation
    if not ccf_annotation.exists():
        print(f"\n⚠️  CCF annotation not found: {ccf_annotation}")
        print("\nThe annotation volume contains region IDs for each voxel.")
        print("Without it, we cannot identify brain regions.")
        
        response = input("\nDownload CCF annotation now? (y/n): ")
        
        if response.lower() == 'y':
            ccf_dir = PROJECT_PATH / "data" / "ccf"
            downloaded = download_ccf_annotation(ccf_dir, resolution=25)
            if downloaded:
                ccf_annotation = Path(downloaded)
            else:
                print("\n⚠️  Continuing without annotation (no region IDs)")
                ccf_annotation = None
        else:
            print("\n⚠️  Continuing without annotation (no region IDs)")
            ccf_annotation = None
    else:
        print(f"\n✓ CCF annotation found: {ccf_annotation}")
        
        # Verify we can read it
        try:
            import nrrd
            print("✓ nrrd library available")
        except ImportError:
            print("\n❌ ERROR: nrrd library not installed!")
            print("Install with: pip install pynrrd")
            response = input("\nContinue without region identification? (y/n): ")
            if response.lower() != 'y':
                return
            ccf_annotation = None
    
    # Check ontology
    if ccf_annotation and not ontology_file.exists():
        print(f"\n⚠️  Ontology not found: {ontology_file}")
        print("Will download from Allen Institute API during runtime")
        ontology_file = None
    
    # Final summary
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY")
    print("="*70)
    print(f"MicroCT Image:  {microct_image.name}")
    print(f"Mode:           {'REGISTERED' if use_registered else 'ALIGNED+TRANSFORM'}")
    if not use_registered and transform_file:
        print(f"Transform:      {transform_file.name}")
    print(f"CCF Annotation: {'✓ Available' if ccf_annotation else '❌ Not available'}")
    print(f"Ontology:       {'✓ Available' if ontology_file and ontology_file.exists() else 'Will download'}")
    print("="*70)
    
    input("\nPress Enter to start fiber tracking...")
    
    print("\n" + "="*70)
    print("STARTING TRACKER")
    print("="*70)
    
    tracker = FiberTracker(
        microct_image_path=str(microct_image),
        ccf_annotation_path=str(ccf_annotation) if ccf_annotation else None,
        transform_path=str(transform_file) if transform_file and not use_registered else None,
        ontology_path=str(ontology_file) if ontology_file and ontology_file.exists() else None,
        output_dir=str(output_dir),
        spacing=0.025,  # 25 micrometers
        use_registered=use_registered
    )
    
    tracker.start()


if __name__ == "__main__":
    main()
