# Allen CCF Resources

The microprism atlas mapper stores its shared Allen Mouse CCF files here:

- `annotation_25.nrrd`
- `average_template_25.nrrd`
- `structure_tree.json`

They are downloaded automatically from the Allen Institute the first time the
mapper runs. To download them before running an analysis:

```bash
python src/allen_ccf_resources.py
```

The annotation and template volumes are not committed to Git because they are
external data. One cached copy is reused for every microCT project analyzed
with this repository.
