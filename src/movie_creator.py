"""
Movie Creator for MicroCT Brain Volumes
Creates MP4 movies from 3D volumes with visual plane selection
"""

import numpy as np
from pathlib import Path
from tifffile import imread
import cv2
from tqdm import tqdm
from typing import Tuple, Optional
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """
    Normalize volume to [0, 1] range for consistent contrast.

    Parameters:
    -----------
    volume : np.ndarray
        Input volume

    Returns:
    --------
    normalized : np.ndarray
        Normalized volume
    """
    vol_min = np.min(volume)
    vol_max = np.max(volume)

    if vol_max > vol_min:
        normalized = (volume - vol_min) / (vol_max - vol_min)
    else:
        normalized = volume

    return normalized


def create_movie_along_axis(
    volume: np.ndarray,
    output_path: str,
    slice_axis: int,
    frame_rate: int = 20,
    flip_vertical: bool = False,
    flip_horizontal: bool = False,
    compression_quality: int = 5
):
    """
    Create movie by slicing along specified axis.

    Parameters:
    -----------
    volume : np.ndarray
        3D volume of shape (dim0, dim1, dim2)
    output_path : str
        Output video file path
    slice_axis : int
        Axis to slice through (0, 1, or 2)
    frame_rate : int
        Frames per second
    flip_vertical : bool
        Flip frames vertically
    flip_horizontal : bool
        Flip frames horizontally
    compression_quality : int
        0 selects lossless HFYU/AVI; positive values select MPEG-4/MP4.
        Positive levels currently use the same codec settings.
    """
    output_path = Path(output_path)
    if compression_quality == 0:
        output_path = output_path.with_suffix(".avi")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    axis_names = {0: "Axis 0", 1: "Axis 1", 2: "Axis 2"}
    print(f"\nCreating movie along {axis_names[slice_axis]}: {output_path.name}")
    print(f"Input volume shape: {volume.shape}")

    # Normalize volume
    normalized = normalize_volume(volume)

    # Configure slicing based on axis
    num_frames = normalized.shape[slice_axis]

    # Get frame dimensions by extracting a sample slice
    if slice_axis == 0:
        sample_frame = normalized[0, :, :]
    elif slice_axis == 1:
        sample_frame = normalized[:, 0, :]
    else:  # slice_axis == 2
        sample_frame = normalized[:, :, 0]

    frame_height, frame_width = sample_frame.shape

    print(f"Output dimensions: {frame_width} x {frame_height}")
    print(f"Number of frames: {num_frames}")
    print(f"Video duration: {num_frames / frame_rate:.2f} seconds")
    print(f"Compression quality: {compression_quality} {'(uncompressed)' if compression_quality == 0 else ''}")

    # Setup video writer based on compression quality
    if compression_quality == 0:
        # Uncompressed
        fourcc = cv2.VideoWriter_fourcc(*'HFYU')  # Huffman lossless codec
        print("Using uncompressed (lossless) codec")
    else:
        # MPEG-4 Part 2 (mp4v), supported by standard OpenCV wheels
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        frame_rate,
        (frame_width, frame_height),
        isColor=False
    )

    if not out.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    # Set quality parameter if using compression
    if compression_quality > 0:
        # Note: cv2.VideoWriter doesn't have direct quality control for mp4v
        # Quality is implicitly controlled by the codec choice
        pass

    # Write frames
    for i in tqdm(range(num_frames), desc=f"Writing frames (axis {slice_axis})"):
        # Extract slice based on axis
        if slice_axis == 0:
            frame = normalized[i, :, :]
        elif slice_axis == 1:
            frame = normalized[:, i, :]
        else:  # slice_axis == 2
            frame = normalized[:, :, i]

        # Apply flips
        if flip_vertical:
            frame = np.flipud(frame)
        if flip_horizontal:
            frame = np.fliplr(frame)

        # Convert to 8-bit
        frame_uint8 = (frame * 255).astype(np.uint8)
        out.write(frame_uint8)

    out.release()
    print(f"✓ Movie saved: {output_path}")


class MovieCreatorGUI:
    """GUI for selecting movie creation options with visual plane preview."""

    def __init__(self, volume_path: str):
        """
        Initialize GUI.

        Parameters:
        -----------
        volume_path : str
            Path to input TIF volume
        """
        self.volume_path = Path(volume_path)
        self.volume = None
        self.frame_rate = 20

        # Store selections as simple attributes
        self.selected_axis0 = False
        self.selected_axis1 = False
        self.selected_axis2 = False
        self.flip_v = False
        self.flip_h = False

        # Store axis names
        self.axis0_name = "axis0"
        self.axis1_name = "axis1"
        self.axis2_name = "axis2"

        # Store compression quality
        self.compression_quality = 5

        # Load volume
        self.load_volume()

        # Create main window
        self.root = tk.Tk()
        self.root.title("MicroCT Movie Creator - Visual Plane Selection")
        self.root.geometry("900x850")
        self.root.resizable(True, True)

        self.create_widgets()

    def load_volume(self):
        """Load and prepare volume."""
        print(f"\nLoading volume from: {self.volume_path}")
        self.volume = imread(str(self.volume_path))

        # Handle different data types
        if self.volume.dtype == np.uint16:
            self.volume = self.volume.astype(np.float64) / 65535.0
        elif self.volume.dtype == np.uint8:
            self.volume = self.volume.astype(np.float64) / 255.0
        else:
            self.volume = self.volume.astype(np.float64)
            if self.volume.max() > 1.0:
                self.volume = self.volume / self.volume.max()

        print(f"Volume shape: {self.volume.shape}")
        print(f"  Axis 0: {self.volume.shape[0]} slices")
        print(f"  Axis 1: {self.volume.shape[1]} slices")
        print(f"  Axis 2: {self.volume.shape[2]} slices")

    def create_widgets(self):
        """Create GUI widgets."""

        # Title
        title_frame = tk.Frame(self.root, bg="#2196F3")
        title_frame.pack(fill="x")

        title = tk.Label(
            title_frame,
            text="MicroCT Movie Creator - Visual Plane Selection",
            font=("Arial", 14, "bold"),
            bg="#2196F3",
            fg="white",
            pady=10
        )
        title.pack()

        # Main container
        main_container = tk.Frame(self.root)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Left side: Preview images
        left_frame = tk.LabelFrame(main_container, text="Visual Plane Preview", padx=10, pady=10)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

        self.create_preview_panel(left_frame)

        # Right side: Controls
        right_frame = tk.Frame(main_container)
        right_frame.pack(side="right", fill="y", padx=(5, 0))

        self.create_control_panel(right_frame)

    def create_preview_panel(self, parent):
        """Create preview images panel."""

        info_label = tk.Label(
            parent,
            text="Select which plane(s) to use for movie creation.\nEach plane shows a middle slice through that axis.",
            justify="left",
            wraplength=500,
            font=("Arial", 8)
        )
        info_label.pack(pady=(0, 5))

        # Create matplotlib figure with 3 subplots - reduced size
        self.fig = Figure(figsize=(7, 6), dpi=75)

        # Get middle slices for each axis
        mid0 = self.volume.shape[0] // 2
        mid1 = self.volume.shape[1] // 2
        mid2 = self.volume.shape[2] // 2

        # Axis 0: slice along first dimension
        ax0 = self.fig.add_subplot(3, 1, 1)
        ax0.imshow(self.volume[mid0, :, :], cmap='gray')
        ax0.set_title(f'Axis 0 Preview ({self.volume.shape[0]} frames)', fontsize=9, fontweight='bold')
        ax0.axis('off')

        # Axis 1: slice along second dimension
        ax1 = self.fig.add_subplot(3, 1, 2)
        ax1.imshow(self.volume[:, mid1, :], cmap='gray')
        ax1.set_title(f'Axis 1 Preview ({self.volume.shape[1]} frames)', fontsize=9, fontweight='bold')
        ax1.axis('off')

        # Axis 2: slice along third dimension
        ax2 = self.fig.add_subplot(3, 1, 3)
        ax2.imshow(self.volume[:, :, mid2], cmap='gray')
        ax2.set_title(f'Axis 2 Preview ({self.volume.shape[2]} frames)', fontsize=9, fontweight='bold')
        ax2.axis('off')

        self.fig.tight_layout(pad=0.5)

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(self.fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def create_control_panel(self, parent):
        """Create control panel."""

        # File info
        file_frame = tk.LabelFrame(parent, text="Input File", padx=10, pady=10)
        file_frame.pack(fill="x", pady=(0, 10))

        file_label = tk.Label(
            file_frame,
            text=f"{self.volume_path.name}",
            wraplength=250,
            justify="left",
            font=("Arial", 9)
        )
        file_label.pack()

        shape_label = tk.Label(
            file_frame,
            text=f"Shape: {self.volume.shape}",
            font=("Arial", 9),
            fg="gray"
        )
        shape_label.pack()

        # Axis selection with naming
        axis_frame = tk.LabelFrame(parent, text="Select Axis/Axes & Names", padx=10, pady=10)
        axis_frame.pack(fill="x", pady=(0, 10))

        tk.Label(
            axis_frame,
            text="Check plane(s) and name them:",
            font=("Arial", 9, "bold")
        ).pack(anchor="w", pady=(0, 5))

        # Axis 0
        axis0_frame = tk.Frame(axis_frame)
        axis0_frame.pack(fill="x", pady=3)

        self.cb0 = tk.Checkbutton(
            axis0_frame,
            text=f"Axis 0 ({self.volume.shape[0]}fr)",
            font=("Arial", 9),
            width=18,
            anchor="w"
        )
        self.cb0.pack(side="left")
        self.cb0.config(command=lambda: self.toggle_axis(0))

        tk.Label(axis0_frame, text="→", font=("Arial", 9)).pack(side="left", padx=2)
        self.axis0_name_var = tk.StringVar(value="axis0")
        axis0_combo = ttk.Combobox(
            axis0_frame,
            textvariable=self.axis0_name_var,
            values=["axis0", "coronal", "horizontal", "sagittal"],
            width=10,
            state="readonly",
            font=("Arial", 9)
        )
        axis0_combo.pack(side="left")
        axis0_combo.bind('<<ComboboxSelected>>', lambda e: self.update_axis_name(0))

        # Axis 1
        axis1_frame = tk.Frame(axis_frame)
        axis1_frame.pack(fill="x", pady=3)

        self.cb1 = tk.Checkbutton(
            axis1_frame,
            text=f"Axis 1 ({self.volume.shape[1]}fr)",
            font=("Arial", 9),
            width=18,
            anchor="w"
        )
        self.cb1.pack(side="left")
        self.cb1.config(command=lambda: self.toggle_axis(1))

        tk.Label(axis1_frame, text="→", font=("Arial", 9)).pack(side="left", padx=2)
        self.axis1_name_var = tk.StringVar(value="axis1")
        axis1_combo = ttk.Combobox(
            axis1_frame,
            textvariable=self.axis1_name_var,
            values=["axis1", "coronal", "horizontal", "sagittal"],
            width=10,
            state="readonly",
            font=("Arial", 9)
        )
        axis1_combo.pack(side="left")
        axis1_combo.bind('<<ComboboxSelected>>', lambda e: self.update_axis_name(1))

        # Axis 2
        axis2_frame = tk.Frame(axis_frame)
        axis2_frame.pack(fill="x", pady=3)

        self.cb2 = tk.Checkbutton(
            axis2_frame,
            text=f"Axis 2 ({self.volume.shape[2]}fr)",
            font=("Arial", 9),
            width=18,
            anchor="w"
        )
        self.cb2.pack(side="left")
        self.cb2.config(command=lambda: self.toggle_axis(2))

        tk.Label(axis2_frame, text="→", font=("Arial", 9)).pack(side="left", padx=2)
        self.axis2_name_var = tk.StringVar(value="axis2")
        axis2_combo = ttk.Combobox(
            axis2_frame,
            textvariable=self.axis2_name_var,
            values=["axis2", "coronal", "horizontal", "sagittal"],
            width=10,
            state="readonly",
            font=("Arial", 9)
        )
        axis2_combo.pack(side="left")
        axis2_combo.bind('<<ComboboxSelected>>', lambda e: self.update_axis_name(2))

        # Frame rate
        fr_frame = tk.LabelFrame(parent, text="Frame Rate", padx=10, pady=10)
        fr_frame.pack(fill="x", pady=(0, 10))

        fr_control = tk.Frame(fr_frame)
        fr_control.pack(fill="x")

        tk.Label(fr_control, text="FPS:", font=("Arial", 9)).pack(side="left", padx=(0, 5))

        self.fr_var = tk.IntVar(value=20)
        fr_spinbox = tk.Spinbox(
            fr_control,
            from_=1,
            to=60,
            textvariable=self.fr_var,
            width=8,
            command=self.update_duration
        )
        fr_spinbox.pack(side="left")

        # Bind multiple events to ensure duration updates
        fr_spinbox.bind('<KeyRelease>', lambda e: self.update_duration())
        fr_spinbox.bind('<ButtonRelease-1>', lambda e: self.update_duration())
        fr_spinbox.bind('<MouseWheel>', lambda e: self.update_duration())
        fr_spinbox.bind('<<Increment>>', lambda e: self.update_duration())
        fr_spinbox.bind('<<Decrement>>', lambda e: self.update_duration())

        # Duration display
        self.duration_label = tk.Label(
            fr_frame,
            text="",
            font=("Arial", 9),
            fg="blue",
            wraplength=250,
            justify="left"
        )
        self.duration_label.pack(pady=(10, 0))

        self.update_duration()

        # Compression quality
        comp_frame = tk.LabelFrame(parent, text="Video Quality", padx=10, pady=10)
        comp_frame.pack(fill="x", pady=(0, 10))

        tk.Label(
            comp_frame,
            text="Select compression (0=uncompressed/lossless):",
            font=("Arial", 9)
        ).pack(anchor="w", pady=(0, 5))

        comp_control = tk.Frame(comp_frame)
        comp_control.pack(fill="x")

        self.comp_var = tk.IntVar(value=5)
        comp_scale = tk.Scale(
            comp_control,
            from_=0,
            to=10,
            orient=tk.HORIZONTAL,
            variable=self.comp_var,
            length=200,
            command=self.update_compression_label
        )
        comp_scale.pack(side="left", padx=(0, 10))

        self.comp_label = tk.Label(comp_control, text="Quality: 5 (Medium)", font=("Arial", 9))
        self.comp_label.pack(side="left")

        tk.Label(
            comp_frame,
            text="Note: 0=largest file/best quality, 10=smallest file",
            font=("Arial", 8),
            fg="gray"
        ).pack(anchor="w", pady=(5, 0))

        # Flip options
        flip_frame = tk.LabelFrame(parent, text="Orientation (Optional)", padx=10, pady=10)
        flip_frame.pack(fill="x", pady=(0, 10))

        tk.Label(
            flip_frame,
            text="If orientation is wrong:",
            font=("Arial", 9)
        ).pack(anchor="w", pady=(0, 5))

        self.cbv = tk.Checkbutton(
            flip_frame,
            text="Flip vertically",
            font=("Arial", 9)
        )
        self.cbv.pack(anchor="w", pady=2)
        self.cbv.config(command=lambda: self.toggle_flip('v'))

        self.cbh = tk.Checkbutton(
            flip_frame,
            text="Flip horizontally",
            font=("Arial", 9)
        )
        self.cbh.pack(anchor="w", pady=2)
        self.cbh.config(command=lambda: self.toggle_flip('h'))

        # Buttons
        button_frame = tk.Frame(parent)
        button_frame.pack(side="bottom", pady=20)

        create_btn = tk.Button(
            button_frame,
            text="Create Movies",
            command=self.create_movies,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 11, "bold"),
            padx=15,
            pady=8,
            width=14
        )
        create_btn.pack(pady=5)

        cancel_btn = tk.Button(
            button_frame,
            text="Cancel",
            command=self.root.quit,
            bg="#f44336",
            fg="white",
            font=("Arial", 11),
            padx=15,
            pady=8,
            width=14
        )
        cancel_btn.pack(pady=5)

    def update_axis_name(self, axis):
        """Update axis name when combobox changes."""
        if axis == 0:
            self.axis0_name = self.axis0_name_var.get()
            print(f"Axis 0 named as: {self.axis0_name}")
        elif axis == 1:
            self.axis1_name = self.axis1_name_var.get()
            print(f"Axis 1 named as: {self.axis1_name}")
        elif axis == 2:
            self.axis2_name = self.axis2_name_var.get()
            print(f"Axis 2 named as: {self.axis2_name}")

        # Update duration display to show new names
        self.update_duration()

    def update_compression_label(self, value):
        """Update compression quality label."""
        quality = int(float(value))
        self.compression_quality = quality

        if quality == 0:
            label_text = "Quality: Uncompressed (Lossless)"
        elif quality <= 3:
            label_text = f"Quality: {quality} (High)"
        elif quality <= 7:
            label_text = f"Quality: {quality} (Medium)"
        else:
            label_text = f"Quality: {quality} (Low)"

        self.comp_label.config(text=label_text)

    def toggle_axis(self, axis):
        """Toggle axis selection - manual state management."""
        # Read the actual checkbox state to synchronize
        if axis == 0:
            # Toggle the stored state
            self.selected_axis0 = not self.selected_axis0
            # Update checkbox to match
            if self.selected_axis0:
                self.cb0.select()
            else:
                self.cb0.deselect()
            print(f"Toggled Axis 0: now {self.selected_axis0}")
        elif axis == 1:
            self.selected_axis1 = not self.selected_axis1
            if self.selected_axis1:
                self.cb1.select()
            else:
                self.cb1.deselect()
            print(f"Toggled Axis 1: now {self.selected_axis1}")
        elif axis == 2:
            self.selected_axis2 = not self.selected_axis2
            if self.selected_axis2:
                self.cb2.select()
            else:
                self.cb2.deselect()
            print(f"Toggled Axis 2: now {self.selected_axis2}")

        print(f"All axes: axis0={self.selected_axis0}, axis1={self.selected_axis1}, axis2={self.selected_axis2}")
        self.update_duration()

    def toggle_flip(self, direction):
        """Toggle flip option."""
        if direction == 'v':
            self.flip_v = not self.flip_v
            print(f"Toggled flip vertical: now {self.flip_v}")
        elif direction == 'h':
            self.flip_h = not self.flip_h
            print(f"Toggled flip horizontal: now {self.flip_h}")

    def on_axis_change(self, axis):
        """Handle axis checkbox change."""
        # Get current values directly from the IntVars
        val0 = self.axis0_var.get()
        val1 = self.axis1_var.get()
        val2 = self.axis2_var.get()

        # Update stored boolean values
        self.selected_axis0 = (val0 == 1)
        self.selected_axis1 = (val1 == 1)
        self.selected_axis2 = (val2 == 1)

        print(f"Axis {axis} clicked")
        print(f"IntVar values: axis0={val0}, axis1={val1}, axis2={val2}")
        print(f"Stored booleans: axis0={self.selected_axis0}, axis1={self.selected_axis1}, axis2={self.selected_axis2}")

        self.update_duration()

    def on_flip_change(self):
        """Handle flip checkbox change."""
        self.flip_v = bool(self.flip_vertical_var.get())
        self.flip_h = bool(self.flip_horizontal_var.get())
        print(f"Flip changed: vertical={self.flip_v}, horizontal={self.flip_h}")

    def update_duration(self, *args):
        """Update estimated video duration."""
        try:
            frame_rate = self.fr_var.get()
            if frame_rate <= 0:
                frame_rate = 1
        except:
            frame_rate = 20

        durations = []

        # Show duration with the user-selected names
        if self.selected_axis0:
            dur = self.volume.shape[0] / frame_rate
            name = self.axis0_name_var.get() if hasattr(self, 'axis0_name_var') else self.axis0_name
            durations.append(f"{name}: {dur:.1f}s ({self.volume.shape[0]} frames)")
        if self.selected_axis1:
            dur = self.volume.shape[1] / frame_rate
            name = self.axis1_name_var.get() if hasattr(self, 'axis1_name_var') else self.axis1_name
            durations.append(f"{name}: {dur:.1f}s ({self.volume.shape[1]} frames)")
        if self.selected_axis2:
            dur = self.volume.shape[2] / frame_rate
            name = self.axis2_name_var.get() if hasattr(self, 'axis2_name_var') else self.axis2_name
            durations.append(f"{name}: {dur:.1f}s ({self.volume.shape[2]} frames)")

        if durations:
            duration_text = f"Est. duration @ {frame_rate} fps:\n" + "\n".join(durations)
        else:
            duration_text = "Select at least one axis\nto see duration"

        self.duration_label.config(text=duration_text)

        # Force update
        self.duration_label.update_idletasks()

    def create_movies(self):
        """Create selected movies."""

        print("\n" + "="*60)
        print("CREATE MOVIES - Reading selections...")
        print(f"self.selected_axis0 = {self.selected_axis0}")
        print(f"self.selected_axis1 = {self.selected_axis1}")
        print(f"self.selected_axis2 = {self.selected_axis2}")
        print("="*60)

        # Get selected axes from stored attributes
        selected_axes = []
        axis_names = []

        if self.selected_axis0:
            selected_axes.append(0)
            # Get the current name from the combobox
            name = self.axis0_name_var.get()
            axis_names.append(name)
            print(f"Adding Axis 0 with name: {name}")
        if self.selected_axis1:
            selected_axes.append(1)
            name = self.axis1_name_var.get()
            axis_names.append(name)
            print(f"Adding Axis 1 with name: {name}")
        if self.selected_axis2:
            selected_axes.append(2)
            name = self.axis2_name_var.get()
            axis_names.append(name)
            print(f"Adding Axis 2 with name: {name}")

        print(f"\nFinal: Selected axes = {selected_axes}")
        print(f"Final: Axis names = {axis_names}")

        if not selected_axes:
            print("ERROR: No axes selected!")
            messagebox.showwarning("No Selection", "Please select at least one axis.")
            return

        # Get parameters - READ DIRECTLY FROM GUI ELEMENTS
        try:
            frame_rate = int(self.fr_var.get())
            if frame_rate <= 0:
                frame_rate = 20
        except:
            frame_rate = 20

        try:
            compression_quality = int(self.comp_var.get())
        except:
            compression_quality = 5

        flip_v = self.flip_v
        flip_h = self.flip_h

        print(f"Debug: Frame rate = {frame_rate}")
        print(f"Debug: Compression quality = {compression_quality}")
        print(f"Debug: Flip vertical = {flip_v}, Flip horizontal = {flip_h}")

        # Confirm before closing GUI
        msg = f"Creating {len(selected_axes)} movie(s):\n\n"
        for i, axis in enumerate(selected_axes):
            dur = self.volume.shape[axis] / frame_rate
            msg += f"  {axis_names[i]}: {self.volume.shape[axis]} frames, {dur:.1f}s\n"
        msg += f"\nFrame rate: {frame_rate} fps\n"
        msg += f"Compression: {compression_quality} {'(Uncompressed)' if compression_quality == 0 else ''}\n"
        msg += "\nOutput files will be named:\n"
        base_name = self.volume_path.stem
        extension = ".avi" if compression_quality == 0 else ".mp4"
        for name in axis_names:
            msg += f"  {base_name}_{name}_movie{extension}\n"
        msg += "\nProceed?"

        if not messagebox.askyesno("Confirm", msg):
            return

        # Close GUI
        self.root.destroy()

        # Create movies for each selected axis
        output_dir = self.volume_path.parent
        base_name = self.volume_path.stem

        print("\n" + "="*60)
        print("Starting movie creation...")
        print("="*60)

        for i, axis in enumerate(selected_axes):
            # Use the axis name directly from axis_names list
            movie_name = axis_names[i]
            output_path = output_dir / f"{base_name}_{movie_name}_movie{extension}"

            print(f"\nCreating movie {i+1}/{len(selected_axes)}:")
            print(f"  Axis: {axis}")
            print(f"  Name: {movie_name}")
            print(f"  Output: {output_path}")
            print(f"  Frame rate: {frame_rate}")
            print(f"  Compression: {compression_quality}")

            try:
                create_movie_along_axis(
                    self.volume,
                    str(output_path),
                    axis,
                    frame_rate,
                    flip_v,
                    flip_h,
                    compression_quality
                )
            except Exception as e:
                print(f"Error creating movie for axis {axis}: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "="*60)
        print("✓ All movies created successfully!")
        print("="*60)
        for i, axis in enumerate(selected_axes):
            print(f"  {axis_names[i]}: {base_name}_{axis_names[i]}_movie{extension}")
        print("="*60)

    def run(self):
        """Run the GUI."""
        self.root.mainloop()


def create_movies_from_volume_gui(volume_path: str):
    """
    Launch GUI for movie creation.

    Parameters:
    -----------
    volume_path : str
        Path to input TIF volume
    """
    gui = MovieCreatorGUI(volume_path)
    gui.run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create movies from a TIFF volume.")
    parser.add_argument("volume_path", nargs="?", help="TIFF volume; opens a file picker when omitted.")
    args = parser.parse_args()
    volume_path = args.volume_path
    if not volume_path:
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        try:
            volume_path = filedialog.askopenfilename(
                title="Select TIF volume to create movies",
                filetypes=[("TIFF files", "*.tif *.tiff")])
        finally:
            root.destroy()
    if volume_path:
        create_movies_from_volume_gui(str(Path(volume_path).expanduser()))
    else:
        print("No file selected. Exiting.")
