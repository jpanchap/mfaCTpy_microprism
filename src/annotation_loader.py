import nrrd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox, RadioButtons
from matplotlib.colors import ListedColormap, Normalize
import json

def load_annotation(filepath):
    """Load the NRRD annotation file"""
    data, header = nrrd.read(filepath)
    return data, header

def load_structure_tree(filepath):
    """Load the structure tree JSON file and create ID to name mapping"""
    with open(filepath, 'r', encoding="utf-8") as f:
        data = json.load(f)

    # Create a mapping from structure ID to structure info
    id_to_info = {}

    def traverse_tree(node):
        """Recursively traverse the structure tree"""
        structure_id = node.get('id')
        name = node.get('name', 'Unknown')
        acronym = node.get('acronym', '')

        if structure_id is not None:
            id_to_info[structure_id] = {
                'name': name,
                'acronym': acronym,
                'color_hex': node.get('color_hex_triplet', 'FFFFFF')
            }

        # Process children
        if 'children' in node:
            for child in node['children']:
                traverse_tree(child)

    # Handle different JSON structures
    if isinstance(data, dict):
        # Check if there's a 'msg' key (common in Allen Institute files)
        if 'msg' in data:
            root_nodes = data['msg']
        else:
            root_nodes = [data]
    elif isinstance(data, list):
        root_nodes = data
    else:
        root_nodes = [data]

    # Traverse all root nodes
    for root in root_nodes:
        traverse_tree(root)

    return id_to_info

def create_color_mapped_image(slice_data, structure_info):
    """Create RGB image from structure IDs using their defined colors"""
    # Create RGB image
    height, width = slice_data.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            struct_id = int(slice_data[i, j])
            if struct_id in structure_info and struct_id != 0:
                hex_color = structure_info[struct_id]['color_hex']
                # Convert hex to RGB
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                rgb_image[i, j] = [r, g, b]
            else:
                # Black for background or unknown
                rgb_image[i, j] = [0, 0, 0]

    return rgb_image

def interactive_viewer_advanced(annotation_data, structure_info):
    """
    Advanced interactive viewer with:
    - Slice navigation (slider + mouse wheel)
    - Colorbar
    - Adjustable color range (slider + manual input)
    - Axis switching
    - Rotation
    - Zoom and Pan
    - Structure name display
    - Color mode selection (ID vs Structure colors)
    """
    # Create figure and axis
    fig = plt.figure(figsize=(16, 10))

    # Main image axis
    ax_img = plt.axes([0.1, 0.30, 0.62, 0.60])

    # Colorbar axis
    ax_cbar = plt.axes([0.74, 0.30, 0.02, 0.60])

    # Info panel axis (for displaying structure details)
    ax_info = plt.axes([0.78, 0.65, 0.20, 0.25])
    ax_info.axis('off')

    # Radio buttons for color mode
    ax_radio = plt.axes([0.78, 0.92, 0.15, 0.07])

    # Slider axes
    ax_slice = plt.axes([0.1, 0.20, 0.62, 0.03])
    ax_vmin = plt.axes([0.1, 0.15, 0.62, 0.03])
    ax_vmax = plt.axes([0.1, 0.10, 0.62, 0.03])

    # Text box axes for manual input
    ax_text_vmin = plt.axes([0.1, 0.05, 0.08, 0.03])
    ax_text_vmax = plt.axes([0.22, 0.05, 0.08, 0.03])

    # Button axes for switching views
    ax_btn_coronal = plt.axes([0.78, 0.57, 0.08, 0.04])
    ax_btn_sagittal = plt.axes([0.87, 0.57, 0.08, 0.04])
    ax_btn_axial = plt.axes([0.78, 0.52, 0.08, 0.04])

    # Button axes for rotation
    ax_btn_rot_ccw = plt.axes([0.78, 0.45, 0.08, 0.04])
    ax_btn_rot_cw = plt.axes([0.87, 0.45, 0.08, 0.04])

    # Button axes for zoom
    ax_btn_zoom_in = plt.axes([0.78, 0.38, 0.08, 0.04])
    ax_btn_zoom_out = plt.axes([0.87, 0.38, 0.08, 0.04])

    # Other buttons
    ax_btn_reset = plt.axes([0.78, 0.31, 0.08, 0.04])
    ax_btn_apply = plt.axes([0.34, 0.05, 0.06, 0.03])
    ax_btn_reset_view = plt.axes([0.87, 0.31, 0.08, 0.04])

    # Initialize parameters
    current_axis = [0]  # 0=coronal, 1=sagittal, 2=axial
    rotation_angle = [0]  # Current rotation angle in degrees
    zoom_level = [1.0]  # Current zoom level
    current_slice_idx = [annotation_data.shape[0] // 2]  # Store current slice index
    color_mode = ['by_id']  # 'by_id' or 'by_structure'

    # Get initial min/max values
    data_min = float(np.min(annotation_data))
    data_max = float(np.max(annotation_data))

    axis_names = ['Coronal', 'Sagittal', 'Axial']

    # Create function to get slice data
    def get_slice_data(slice_idx=None, axis=None, rotation=None):
        """Get current slice data based on axis and rotation"""
        if slice_idx is None:
            slice_idx = current_slice_idx[0]
        if axis is None:
            axis = current_axis[0]
        if rotation is None:
            rotation = rotation_angle[0]

        slice_idx = int(slice_idx)

        if axis == 0:
            slice_data = annotation_data[slice_idx, :, :].T
        elif axis == 1:
            slice_data = annotation_data[:, slice_idx, :].T
        else:
            slice_data = annotation_data[:, :, slice_idx].T

        # Apply rotation
        k = rotation // 90  # Number of 90-degree rotations
        if k != 0:
            slice_data = np.rot90(slice_data, k)

        return slice_data

    # Create initial image
    initial_slice = get_slice_data()

    # Display image (initially by ID)
    img = ax_img.imshow(initial_slice, cmap='nipy_spectral', origin='lower',
                        vmin=data_min, vmax=data_max)
    ax_img.set_title(f'{axis_names[current_axis[0]]} Slice {current_slice_idx[0]} (Rotation: {rotation_angle[0]}°) - Color by ID',
                     fontsize=14, fontweight='bold')
    ax_img.axis('on')

    # Add colorbar
    cbar = plt.colorbar(img, cax=ax_cbar)
    cbar.set_label('Structure ID', rotation=270, labelpad=20, fontsize=12)

    # Info panel text
    info_text = ax_info.text(0.05, 0.95, 'Hover over image\nfor structure info',
                            transform=ax_info.transAxes,
                            fontsize=10, verticalalignment='top',
                            family='monospace',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Create radio buttons for color mode
    radio = RadioButtons(ax_radio, ('Color by ID', 'Color by Structure'))

    # Create sliders
    slider_slice = Slider(ax_slice, 'Slice', 0, annotation_data.shape[current_axis[0]]-1,
                          valinit=current_slice_idx[0], valstep=1)
    slider_vmin = Slider(ax_vmin, 'Min ID', data_min, data_max,
                         valinit=data_min, valstep=1)
    slider_vmax = Slider(ax_vmax, 'Max ID', data_min, data_max,
                         valinit=data_max, valstep=1)

    # Create text boxes for manual input
    text_box_vmin = TextBox(ax_text_vmin, 'Min:', initial=str(int(data_min)))
    text_box_vmax = TextBox(ax_text_vmax, 'Max:', initial=str(int(data_max)))

    # Create buttons
    btn_coronal = Button(ax_btn_coronal, 'Coronal')
    btn_sagittal = Button(ax_btn_sagittal, 'Sagittal')
    btn_axial = Button(ax_btn_axial, 'Axial')
    btn_rot_ccw = Button(ax_btn_rot_ccw, '↺ 90°')
    btn_rot_cw = Button(ax_btn_rot_cw, '↻ 90°')
    btn_zoom_in = Button(ax_btn_zoom_in, 'Zoom +')
    btn_zoom_out = Button(ax_btn_zoom_out, 'Zoom -')
    btn_reset = Button(ax_btn_reset, 'Reset Range')
    btn_apply = Button(ax_btn_apply, 'Apply')
    btn_reset_view = Button(ax_btn_reset_view, 'Reset View')

    # Text for displaying current structure ID under cursor
    text_info = ax_img.text(0.02, 0.98, '', transform=ax_img.transAxes,
                           fontsize=9, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    def update_image():
        """Update the displayed image"""
        current_slice_idx[0] = int(slider_slice.val)
        slice_data = get_slice_data()

        if color_mode[0] == 'by_structure':
            # Create RGB image using structure colors
            rgb_image = create_color_mapped_image(slice_data, structure_info)
            img.set_data(rgb_image)
            img.set_cmap(None)  # Remove colormap for RGB
            img.set_clim(None, None)
            # Hide colorbar for structure color mode
            cbar.ax.set_visible(False)
            title_suffix = "Color by Structure"
            # Disable range sliders
            slider_vmin.ax.set_visible(False)
            slider_vmax.ax.set_visible(False)
            text_box_vmin.ax.set_visible(False)
            text_box_vmax.ax.set_visible(False)
            btn_apply.ax.set_visible(False)
            btn_reset.ax.set_visible(False)
        else:
            # Use ID-based coloring
            img.set_data(slice_data)
            img.set_cmap('nipy_spectral')
            vmin = slider_vmin.val
            vmax = slider_vmax.val
            img.set_clim(vmin, vmax)
            # Show colorbar for ID mode
            cbar.ax.set_visible(True)
            title_suffix = "Color by ID"
            # Enable range sliders
            slider_vmin.ax.set_visible(True)
            slider_vmax.ax.set_visible(True)
            text_box_vmin.ax.set_visible(True)
            text_box_vmax.ax.set_visible(True)
            btn_apply.ax.set_visible(True)
            btn_reset.ax.set_visible(True)

        img.set_extent([0, slice_data.shape[1], 0, slice_data.shape[0]])
        ax_img.set_title(f'{axis_names[current_axis[0]]} Slice {current_slice_idx[0]} (Rotation: {rotation_angle[0]}°) - {title_suffix}',
                         fontsize=14, fontweight='bold')
        fig.canvas.draw_idle()

    def update_slice(val):
        """Update the displayed slice"""
        update_image()

    def on_scroll(event):
        """Handle mouse wheel scroll to navigate slices"""
        if event.inaxes == ax_img:
            # Get current slice and max slice
            current_slice = int(slider_slice.val)
            max_slice = annotation_data.shape[current_axis[0]] - 1

            # Scroll up = next slice, scroll down = previous slice
            if event.button == 'up':
                new_slice = min(current_slice + 1, max_slice)
            elif event.button == 'down':
                new_slice = max(current_slice - 1, 0)
            else:
                return

            # Update slider (this will trigger update_image through the slider callback)
            slider_slice.set_val(new_slice)

    def update_vmin(val):
        """Update minimum color value"""
        if color_mode[0] == 'by_id':
            vmin = slider_vmin.val
            vmax = slider_vmax.val
            if vmin < vmax:
                img.set_clim(vmin=vmin)
                text_box_vmin.set_val(str(int(vmin)))
                fig.canvas.draw_idle()
            else:
                slider_vmin.set_val(vmax - 1)

    def update_vmax(val):
        """Update maximum color value"""
        if color_mode[0] == 'by_id':
            vmin = slider_vmin.val
            vmax = slider_vmax.val
            if vmax > vmin:
                img.set_clim(vmax=vmax)
                text_box_vmax.set_val(str(int(vmax)))
                fig.canvas.draw_idle()
            else:
                slider_vmax.set_val(vmin + 1)

    def apply_manual_range(event):
        """Apply manually entered min/max values"""
        if color_mode[0] == 'by_id':
            try:
                vmin = float(text_box_vmin.text)
                vmax = float(text_box_vmax.text)

                if vmin >= vmax:
                    print("Error: Min ID must be less than Max ID")
                    return

                if vmin < data_min or vmax > data_max:
                    print(f"Warning: Values outside data range [{data_min}, {data_max}]")

                slider_vmin.set_val(vmin)
                slider_vmax.set_val(vmax)
                img.set_clim(vmin=vmin, vmax=vmax)
                fig.canvas.draw_idle()
                print(f"Applied range: [{vmin}, {vmax}]")

            except ValueError:
                print("Error: Please enter valid numbers")

    def submit_vmin(text):
        """Handle text box submit for vmin"""
        apply_manual_range(None)

    def submit_vmax(text):
        """Handle text box submit for vmax"""
        apply_manual_range(None)

    def change_color_mode(label):
        """Change between ID and structure color modes"""
        if label == 'Color by ID':
            color_mode[0] = 'by_id'
        else:
            color_mode[0] = 'by_structure'
        update_image()

    def rotate_ccw(event):
        """Rotate counter-clockwise by 90 degrees"""
        rotation_angle[0] = (rotation_angle[0] - 90) % 360
        update_image()

    def rotate_cw(event):
        """Rotate clockwise by 90 degrees"""
        rotation_angle[0] = (rotation_angle[0] + 90) % 360
        update_image()

    def zoom_in(event):
        """Zoom in by reducing view limits"""
        zoom_level[0] *= 1.3
        xlim = ax_img.get_xlim()
        ylim = ax_img.get_ylim()

        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        x_range = (xlim[1] - xlim[0]) / 1.3
        y_range = (ylim[1] - ylim[0]) / 1.3

        ax_img.set_xlim(x_center - x_range/2, x_center + x_range/2)
        ax_img.set_ylim(y_center - y_range/2, y_center + y_range/2)
        fig.canvas.draw_idle()

    def zoom_out(event):
        """Zoom out by expanding view limits"""
        zoom_level[0] /= 1.3
        xlim = ax_img.get_xlim()
        ylim = ax_img.get_ylim()

        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        x_range = (xlim[1] - xlim[0]) * 1.3
        y_range = (ylim[1] - ylim[0]) * 1.3

        # Don't zoom out beyond initial limits
        slice_data = get_slice_data()
        max_x = slice_data.shape[1]
        max_y = slice_data.shape[0]

        new_xlim = [max(0, x_center - x_range/2), min(max_x, x_center + x_range/2)]
        new_ylim = [max(0, y_center - y_range/2), min(max_y, y_center + y_range/2)]

        ax_img.set_xlim(new_xlim)
        ax_img.set_ylim(new_ylim)
        fig.canvas.draw_idle()

    def reset_view(event):
        """Reset zoom and pan to original view"""
        zoom_level[0] = 1.0
        slice_data = get_slice_data()
        ax_img.set_xlim(0, slice_data.shape[1])
        ax_img.set_ylim(0, slice_data.shape[0])
        fig.canvas.draw_idle()

    def switch_axis(new_axis):
        """Switch viewing axis"""
        current_axis[0] = new_axis
        rotation_angle[0] = 0  # Reset rotation when switching axis

        # Update slice slider range and value
        current_slice_idx[0] = annotation_data.shape[new_axis] // 2
        slider_slice.valmax = annotation_data.shape[new_axis] - 1
        slider_slice.ax.set_xlim(0, annotation_data.shape[new_axis] - 1)
        slider_slice.set_val(current_slice_idx[0])

        # Reset view
        reset_view(None)

    def reset_range(event):
        """Reset color range to full data range"""
        slider_vmin.set_val(data_min)
        slider_vmax.set_val(data_max)
        text_box_vmin.set_val(str(int(data_min)))
        text_box_vmax.set_val(str(int(data_max)))

    def on_mouse_move(event):
        """Display structure ID and name under cursor"""
        if event.inaxes == ax_img and event.xdata is not None and event.ydata is not None:
            # Get current slice data to determine dimensions
            slice_data = get_slice_data()
            x, y = int(event.xdata), int(event.ydata)

            if 0 <= x < slice_data.shape[1] and 0 <= y < slice_data.shape[0]:
                structure_id = int(slice_data[y, x])

                # Get structure information
                if structure_id in structure_info:
                    info = structure_info[structure_id]
                    name = info['name']
                    acronym = info['acronym']

                    # Update small overlay text
                    text_info.set_text(f'Pos: ({x}, {y})\nID: {structure_id}\n{acronym}')

                    # Update info panel with word wrapping for long names
                    info_display = f"Structure ID: {structure_id}\n\n"
                    info_display += f"Acronym: {acronym}\n\n"
                    info_display += f"Name:\n{name}\n\n"
                    info_display += f"Position: ({x}, {y})"

                    info_text.set_text(info_display)
                else:
                    text_info.set_text(f'Pos: ({x}, {y})\nID: {structure_id}\n(Unknown)')
                    info_text.set_text(f"Structure ID: {structure_id}\n\nName: Unknown\n\nPosition: ({x}, {y})")

                fig.canvas.draw_idle()

    # Connect callbacks
    slider_slice.on_changed(update_slice)
    slider_vmin.on_changed(update_vmin)
    slider_vmax.on_changed(update_vmax)
    text_box_vmin.on_submit(submit_vmin)
    text_box_vmax.on_submit(submit_vmax)
    radio.on_clicked(change_color_mode)
    btn_coronal.on_clicked(lambda event: switch_axis(0))
    btn_sagittal.on_clicked(lambda event: switch_axis(1))
    btn_axial.on_clicked(lambda event: switch_axis(2))
    btn_rot_ccw.on_clicked(rotate_ccw)
    btn_rot_cw.on_clicked(rotate_cw)
    btn_zoom_in.on_clicked(zoom_in)
    btn_zoom_out.on_clicked(zoom_out)
    btn_reset.on_clicked(reset_range)
    btn_apply.on_clicked(apply_manual_range)
    btn_reset_view.on_clicked(reset_view)
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    fig.canvas.mpl_connect('scroll_event', on_scroll)  # Add scroll event

    plt.show()

# Main execution
if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from project_paths import resolve_project_path

    parser = argparse.ArgumentParser(description="Inspect Allen CCF annotations.")
    parser.add_argument("project_path", nargs="?", help="Project containing data/.")
    parser.add_argument("--annotation", help="Explicit annotation NRRD path.")
    parser.add_argument("--ontology", help="Explicit structure-tree JSON path.")
    args = parser.parse_args()
    project = resolve_project_path(args.project_path)
    annotation_filepath = Path(args.annotation).expanduser() if args.annotation else project / "data/ccf/annotation_25.nrrd"
    structure_tree_filepath = Path(args.ontology).expanduser() if args.ontology else project / "data/ccf/structure_tree.json"

    print("Loading annotation file...")
    annotation_data, header = load_annotation(annotation_filepath)

    print(f"Annotation shape: {annotation_data.shape}")
    print(f"Unique structure IDs: {len(np.unique(annotation_data))}")
    print(f"Data type: {annotation_data.dtype}")
    print(f"Min ID: {np.min(annotation_data)}, Max ID: {np.max(annotation_data)}")

    print("\nLoading structure tree...")
    structure_info = load_structure_tree(structure_tree_filepath)
    print(f"Loaded {len(structure_info)} brain structures")

    # Print first few structures to verify loading
    if len(structure_info) > 0:
        print("\nSample structures:")
        for i, (struct_id, info) in enumerate(list(structure_info.items())[:5]):
            print(f"  ID {struct_id}: {info['acronym']} - {info['name']} (Color: #{info['color_hex']})")

    print("\nLaunching interactive viewer...")
    print("- Use radio buttons to switch between 'Color by ID' and 'Color by Structure'")
    print("- Use 'Slice' slider to navigate through slices")
    print("- Scroll mouse wheel over image to move through slices (up=next, down=previous)")
    print("- In 'Color by ID' mode: Use sliders/text boxes to adjust color range")
    print("- In 'Color by Structure' mode: Each region shows its anatomical color")
    print("- Click '↺ 90°' or '↻ 90°' to rotate the view")
    print("- Click 'Zoom +' or 'Zoom -' to zoom in/out")
    print("- Click 'Reset View' to reset zoom and rotation")
    print("- Hover mouse over image to see structure name and ID")

    interactive_viewer_advanced(annotation_data, structure_info)
