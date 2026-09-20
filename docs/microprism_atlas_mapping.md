# Microprism Atlas Mapping Logic

## Goal

`microprism_atlas_mapper.py` runs after `microprism_corner_tracker.py`. It:

1. identifies the Allen CCF region at each of the four imaging-face corners;
2. constructs the oblique atlas plane corresponding to that face;
3. displays the complete atlas slice through that plane;
4. outlines the actual prism face inside the larger slice;
5. reports region coverage inside the prism face.

## Inputs

The tracker JSON provides four corners in `(x, y, z)` voxel coordinates:

- `P1`: bottom-left shared corner
- `P2`: bottom-right shared corner
- `derived_left`: top-left imaging-face corner
- `derived_right`: top-right imaging-face corner

The mapper also loads:

- `annotation_25.nrrd`: one Allen structure ID per atlas voxel
- `structure_tree.json`: structure names, acronyms, and colors
- optionally `transform_landmark.tfm`: mapping between aligned microCT and CCF

The tracker records source voxel size as `spacing_mm_xyz` in `(X, Y, Z)` order.
This is important when the microCT pixels are not physically isotropic. The
mapper uses those per-axis dimensions when converting tracked voxel coordinates
into physical millimeters before applying the registration transform.

## How mfaCT Places the MicroCT Image in the Atlas

The existing registration workflow is in `src/landmark_registration.py`.

1. The user chooses matching landmarks in the microCT volume and Allen atlas.
2. Landmark coordinates are stored as NumPy `(z, y, x)`.
3. They are reordered to SimpleITK physical `(x, y, z)` and multiplied by the
   image spacing for each axis.
4. `LandmarkBasedTransformInitializer` estimates an affine, similarity, or
   rigid transform from the paired landmarks.
5. `ResampleImageFilter` creates an output image using the Allen atlas as its
   reference grid.
6. For every Allen output voxel, SimpleITK uses the saved transform to find
   where that location came from in the moving microCT image.
7. The result is saved as `microct_registered.tif`. Its voxel grid is therefore
   intended to correspond directly to the Allen atlas grid.

Transform direction can vary with how a registration pipeline constructs and
saves its SimpleITK transform. The mapper therefore tries both the saved
transform and its inverse, then chooses the mapping that places the most prism
corners inside the CCF annotation bounds. The selected direction is printed and
recorded in the output JSON.

## Corner Region Lookup

Pinpoint's `ReferenceAtlas.GetAnnotationIdx` performs this operation:

1. receive an atlas index coordinate;
2. round each component to the nearest voxel;
3. reject coordinates outside the annotation dimensions;
4. return the integer annotation ID at that voxel;
5. use the ontology to convert the ID into an acronym, name, and color.

The mapper reproduces that behavior:

```text
x_i, y_i, z_i = round(x), round(y), round(z)
region_id = annotation[z_i, y_i, x_i]
region = ontology[region_id]
```

NumPy volumes are indexed `(z, y, x)`, even though points are stored and
reported as `(x, y, z)`.

For the Allen 25 um coronal volume used here, those array axes correspond to
`[AP, DV, ML]`. This matches the convention documented by the lab's
FOV-atlas-registration project: `annotation[ap, dv, ml]`.

## Fitting the Imaging Plane

Let the four CCF-space points be:

```text
BL = P1
BR = P2
TL = derived_left
TR = derived_right
```

The plane origin is the mean of all four corners:

```text
O = (BL + BR + TL + TR) / 4
```

The width direction averages the bottom and top edges:

```text
w_hint = ((BR - BL) + (TR - TL)) / 2
w = w_hint / ||w_hint||
```

The initial length direction averages the left and right edges:

```text
l_hint = ((TL - BL) + (TR - BR)) / 2
```

Any component parallel to the width direction is removed:

```text
l_perp = l_hint - dot(l_hint, w) * w
l = l_perp / ||l_perp||
```

This is Gram-Schmidt orthogonalization. It creates perpendicular display axes
without discarding the measured direction of the prism.

The plane normal is:

```text
n = normalize(cross(w, l))
```

Each corner's signed distance from the fitted plane is also saved. Large
distances indicate that the four mapped points are not sufficiently coplanar.

## Displaying the Complete Corresponding Plane

The fitted plane is infinite, but the annotation volume is a rectangular box.
The code tests the plane against all 12 box edges. Wherever an edge crosses the
plane, it computes the intersection:

```text
t = d1 / (d1 - d2)
intersection = edge_start + t * (edge_end - edge_start)
```

Here `d1` and `d2` are the signed distances from the two edge endpoints to the
plane. The intersection points determine the full visible atlas-plane extent.

For each display pixel with 2D plane coordinate `(u, v)`, its 3D atlas point is:

```text
P(u, v) = O + u*w + v*l
```

That point is rounded and looked up in the annotation volume. This is the CPU
equivalent of the oriented annotation-texture sampling used by Pinpoint's
in-plane slice.

## Prism Outline and Face Coverage

The four prism corners are projected into plane coordinates:

```text
u = dot(P - O, w)
v = dot(P - O, l)
```

Those four projected points form the yellow quadrilateral in the visual. A
polygon mask determines which full-plane pixels lie inside the prism face.
Region percentages are calculated separately for:

- the complete atlas plane;
- only the outlined prism face.

Because an affine atlas registration can shear or unequally scale the four
mapped corners, the viewer fits the closest true rectangle in this plane. Its
height-to-width ratio is taken from the prism length and width entered in the
tracker. This keeps the displayed imaging face physically rectangular instead
of displaying registration skew as part of the prism.

The final figure contains only this atlas slice. The screen coordinate system
is fixed to the prism: prism width runs left-to-right and prism length runs
bottom-to-top, so an arbitrarily oriented implant is not shown as a tilted
canvas. `Coronal`, `Sagittal`, and `Axial` controls show all three standard
Allen sections through the face center and project the four face corners onto
each one. These views preserve atlas anatomy, so their projection is not
necessarily rectangular. Click-drag, use the directional buttons, or use the
arrow keys to pan. Scroll or use `+`/`-` to zoom, use `Full`/`F`/`0` for the
complete current cross-section, and use `R`/`Home` to reset the current view.
When available, the Allen average anatomical template is shown beneath
translucent annotation colors and boundaries. The prism face remains
translucent so underlying anatomy is readable.

## Outputs

- `microprism_atlas_plane.png`: close prism view, outline, and corner regions
- `microprism_atlas_plane_full.png`: complete corresponding atlas slice
- `microprism_atlas_plane_coronal.png`: standard coronal section
- `microprism_atlas_plane_sagittal.png`: standard sagittal section
- `microprism_atlas_plane_axial.png`: standard axial section
- `microprism_atlas_regions.csv`: full-plane and prism-face region coverage
- `microprism_atlas_regions.json`: coordinates, corner regions, plane vectors,
  fit errors, settings, and summaries
- `microprism_atlas_plane_labels.npy`: raw region-ID image
- `microprism_atlas_face_mask.npy`: pixels belonging to the prism face
