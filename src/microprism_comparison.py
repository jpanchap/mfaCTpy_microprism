"""Independent atlas mappings with synchronized anatomical inspection and selection."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.widgets import Button, Slider
from microprism_geometry import slice_intersection
from project_paths import resolve_project_path

METHODS = ('fixed', 'original')
ORDER = ('bottom_left', 'bottom_right', 'top_right', 'top_left')


def compare_methods(args, run_mapper):
    # Import here to avoid a circular import when launched through mapper.main.
    from microprism_atlas_mapper import load_face_corners, load_annotation, resolve_input
    project = resolve_project_path(args.project_path)
    source = resolve_input(args.corners, project, ('microprism_corners.json',), 'tracker output')
    for method in METHODS:
        load_face_corners(source, method, args.fixed_branch)  # preflight before any expensive maps
    root = (Path(args.output).expanduser().resolve() if args.output else project / 'outputs') / 'derivation_comparison'
    root.mkdir(parents=True, exist_ok=True)
    source_snapshot = root / 'tracker_input.json'
    source_snapshot.write_bytes(source.read_bytes())
    results = {}
    direction = args.saved_transform_maps
    for method in METHODS:
        child = copy.copy(args)
        child.compare_methods = False
        child.method = method
        child.no_show = True
        child.output = str(root / method)
        child.corners = str(source_snapshot)
        child.saved_transform_maps = direction
        run_mapper(child)
        results[method] = json.loads((root / method / 'microprism_atlas_regions.json').read_text(encoding="utf-8"))
        # An auto-direction heuristic must not select a different transform for each face.
        selected = results[method]['selected_transform_direction']
        if selected in ('direct', 'inverse'):
            direction = 'moving-to-fixed' if selected == 'direct' else 'fixed-to-moving'
    manifest = {'tracker_snapshot': str(source_snapshot), 'tracker_output': str(source), 'tracker_sha256': hashlib.sha256(source_snapshot.read_bytes()).hexdigest(),
                'results': {m: str(root / m / 'microprism_atlas_regions.json') for m in METHODS},
                'registration_direction': direction, 'preferred_method': None,
                'fixed_branch': results['fixed'].get('derivation_branch'),
                'note': 'Visual preference is not independent validation of anatomical accuracy.'}
    manifest_path = root / 'comparison.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f'Both independent mappings saved under {root}')
    if args.no_show:
        return {'output_dir': str(root)}
    reference = results['fixed']
    image_path = reference.get('template') or reference['annotation']
    volume = load_annotation(Path(image_path))
    show_comparison(results, root, volume, manifest, manifest_path)
    return {'output_dir': str(root)}


def show_comparison(results, root, volume, manifest, manifest_path):
    """Same anatomical slice and limits on both sides; separate in-plane samples."""
    spacing = np.array(results['fixed']['atlas_spacing_mm_xyz'])
    faces = {m: np.array([results[m]['ccf_corners_xyz_voxel'][k] for k in ORDER]) for m in METHODS}
    colors = {'fixed': '#00cbbc', 'original': '#ff9d39'}
    labels = {m: np.load(root / m / 'microprism_atlas_plane_labels.npy') for m in METHODS}
    masks = {m: np.load(root / m / 'microprism_atlas_face_mask.npy') for m in METHODS}
    fig, axes = plt.subplots(1, 2, figsize=(15, 9))
    fig.subplots_adjust(bottom=.31, top=.89, wspace=.08)
    state = {'view': 'coronal', 'updating': False, 'labels': True}
    # axis dropped, displayed horizontal, displayed vertical
    convention = {'coronal': (2,0,1), 'sagittal': (0,2,1), 'axial': (1,2,0)}
    center = np.concatenate(list(faces.values())).mean(0)
    size_xyz = np.array(volume.shape[::-1])
    slider = Slider(fig.add_axes([.22,.245,.56,.025]), 'Slice', 0, volume.shape[0]-1,
                    valinit=int(np.clip(round(center[2]),0,volume.shape[0]-1)), valstep=1)
    status = fig.text(.5,.015,'Choose a preferred method after inspecting the overlays.',ha='center')
    ids = sorted(set(np.unique(labels['fixed'])) | set(np.unique(labels['original'])))
    region_colors = {r['region_id']: '#' + r['color_hex'] for m in METHODS
                     for r in results[m]['full_plane_region_summary']}
    palette = np.array([to_rgb(region_colors.get(i, '#000000')) for i in ids])
    summaries = []
    for i,m in enumerate(METHODS):
        rows = results[m]['prism_face_region_summary']
        text = '\n'.join('  |  '.join(f"{r['acronym']}: {r['percent_of_plane']:.1f}%" for r in rows[start:start+3]) for start in (0,3) if rows[start:start+3])
        diag = results[m].get('derivation_diagnostics', {})
        fit = f"\nSource fit RMS: {diag['rms_error_mm']:.4f} mm" if 'rms_error_mm' in diag else ''
        summaries.append(fig.text(.27 + .46*i,.19,text+fit,ha='center',fontsize=9,wrap=True))

    def redraw(_=None):
        view = state['view']
        for m, ax in zip(METHODS, axes):
            ax.clear()
            face = faces[m]
            if view == 'in-plane':
                # Same Allen colors across methods; sample grids remain independent.
                image = labels[m]
                display = np.searchsorted(ids, image)
                extent = np.array(results[m]['plane_extent_uv_voxels']) * spacing[0]
                ax.imshow(palette[display], origin='lower', extent=extent, interpolation='nearest')
                origin = np.array(results[m]['plane_origin_xyz_voxel'])
                u = np.array(results[m]['plane_width_unit_xyz'])
                v = np.array(results[m]['plane_length_unit_xyz'])
                xy = np.column_stack(((face-origin)@u,(face-origin)@v))*spacing[0]
                ax.contour(masks[m], levels=[.5], origin='lower', extent=extent, colors=[colors[m]], linewidths=1)
                ax.set_xlim(xy[:,0].min()-.3,xy[:,0].max()+.3)
                ax.set_ylim(xy[:,1].min()-.3,xy[:,1].max()+.3)
            else:
                dropped,h,v = convention[view]
                index = int(slider.val)
                if view == 'coronal': image = volume[index,:,:]
                elif view == 'sagittal': image = volume[:,:,index].T
                else: image = volume[:,index,:].T
                extent = [0,(size_xyz[h]-1)*spacing[h],-(size_xyz[v]-1)*spacing[v],0]
                ax.imshow(image,cmap='gray',origin='upper',extent=extent,interpolation='nearest')
                xy = np.column_stack((face[:,h]*spacing[h],-face[:,v]*spacing[v]))
                all_points = np.concatenate(list(faces.values()))
                horizontal=all_points[:,h]*spacing[h]; vertical=-all_points[:,v]*spacing[v]
                ax.set_xlim(horizontal.min()-.6,horizontal.max()+.6)
                ax.set_ylim(vertical.min()-.6,vertical.max()+.6)
                # Only draw a slice-intersection line for genuinely planar faces.
                n = np.cross(face[1]-face[0],face[3]-face[0])
                if np.linalg.norm(n)>0 and abs((face[2]-face[0])@n/np.linalg.norm(n))<1e-5:
                    hit = slice_intersection(face,dropped,index)
                    if len(hit)>=2:
                        if len(hit)>2: hit=np.vstack([hit,hit[0]])
                        ax.plot(hit[:,h]*spacing[h],-hit[:,v]*spacing[v],color=colors[m],lw=3)
            closed=np.vstack([xy,xy[0]])
            ax.plot(closed[:,0],closed[:,1], '--' if view!='in-plane' else '-',color=colors[m],lw=2)
            for label, point in zip(('P1','P2','D2','D1'),xy):
                if state['labels']:
                    ax.text(*point,label,color=colors[m],fontsize=10)
            ax.set_aspect('equal')
            ax.set_title(('Fixed dimensions' if m=='fixed' else 'Original triangles')+' — '+view)
        fig.suptitle('Independent imaging-face mappings\nDashed: full projection • Solid: slice intersection • Region summaries use each face’s own plane')
        fig.canvas.draw_idle()

    def change_view(view):
        state['view']=view
        slider.ax.set_visible(view!='in-plane')
        if view!='in-plane':
            dropped,_,_=convention[view]
            state['updating']=True
            slider.valmax=int(size_xyz[dropped]-1)
            slider.ax.set_xlim(0,slider.valmax)
            slider.set_val(int(np.clip(round(center[dropped]),0,slider.valmax)))
            state['updating']=False
        redraw()

    def choose(method):
        # Export a complete selected face for downstream mapper runs, preserving both alternatives.
        source_data = json.loads(Path(manifest['tracker_snapshot']).read_text(encoding="utf-8"))
        entry = source_data['derivation_methods'][method]
        if method == 'fixed':
            branch = manifest['fixed_branch']
            face = entry['candidates'][branch]
            entry['selected_branch'] = branch
            entry['imaging_face_points'] = face
        else:
            face = entry['imaging_face_points']
        source_data['selected_method'] = method
        source_data['imaging_face_points'] = face
        source_data['derived_points'] = {k: face[k] for k in ('derived_left', 'derived_right')}
        source_data['imaging_face_diagnostics'] = entry.get('diagnostics', {})
        source_data['imaging_face_math'] = ('Rigid fixed-dimension right-prism fit with explicit orientation.'
                                          if method == 'fixed' else 'Original independent right-triangle solves.')
        preferred_path = root / 'preferred_corners.json'
        preferred_path.write_text(json.dumps(source_data, indent=2), encoding="utf-8")
        manifest['preferred_corners'] = str(preferred_path)
        manifest['preferred_method']=method
        manifest_path.write_text(json.dumps(manifest,indent=2), encoding="utf-8")
        status.set_text(f'Preference saved: {method}. Both mapping outputs are retained.')
        fig.canvas.draw_idle()

    slider.on_changed(lambda value: None if state['updating'] else redraw())
    buttons=[]
    for i,view in enumerate(('coronal','sagittal','axial','in-plane')):
        button=Button(fig.add_axes([.16+i*.17,.105,.15,.045]),view.title())
        button.on_clicked(lambda e,v=view:change_view(v));buttons.append(button)
    for i,m in enumerate(METHODS):
        button=Button(fig.add_axes([.22+i*.34,.045,.23,.045]),'Prefer '+m)
        button.on_clicked(lambda e,method=m:choose(method));buttons.append(button)
    from prism_ui_help import COMPARISON_HELP, guide_location

    def show_help(_=None):
        print(COMPARISON_HELP)
        print(f"Complete controls and visual reference: {guide_location()}")

    def toggle_labels(_=None):
        state['labels'] = not state['labels']
        label_button.label.set_text('Labels ' + ('on' if state['labels'] else 'off') + ' (L)')
        redraw()

    label_button = Button(fig.add_axes([.835,.105,.15,.045]), 'Labels on (L)')
    label_button.on_clicked(toggle_labels)
    help_button = Button(fig.add_axes([.015,.105,.13,.045]), 'Help (?)')
    help_button.on_clicked(show_help)
    buttons.extend([label_button, help_button])

    def on_key(event):
        key = (event.key or '').lower()
        if key == 'l': toggle_labels()
        elif key in ('?', 'h', 'f1'): show_help()
        elif state['view'] != 'in-plane' and key.split('+')[-1] in ('left','right','up','down'):
            delta = (10 if key.startswith('shift+') else 1) * (-1 if key.split('+')[-1] in ('left','down') else 1)
            slider.set_val(int(np.clip(slider.val+delta,0,slider.valmax)))

    handler = getattr(fig.canvas.manager, 'key_press_handler_id', None)
    if handler is not None: fig.canvas.mpl_disconnect(handler)
    fig.canvas.mpl_connect('key_press_event', on_key)
    show_help()
    fig._prism_comparison = {'change_view': change_view, 'choose': choose, 'slider': slider, 'buttons': buttons, 'toggle_labels': toggle_labels, 'on_key': on_key}
    redraw()
    fig.savefig(root/'comparison.png',dpi=140)
    plt.show()
    return fig
