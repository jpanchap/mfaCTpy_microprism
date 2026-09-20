"""Download and locate shared Allen Mouse CCF resources for mfaCT."""

from __future__ import annotations

import json
import shutil
import sys
import urllib.request
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESOURCE_DIR = REPOSITORY_ROOT / "resources" / "allen_ccf"

ANNOTATION_NAME = "annotation_25.nrrd"
TEMPLATE_NAME = "average_template_25.nrrd"
ONTOLOGY_NAME = "structure_tree.json"

ANNOTATION_URL = (
    "https://download.alleninstitute.org/informatics-archive/current-release/"
    "mouse_ccf/annotation/ccf_2017/annotation_25.nrrd"
)
ONTOLOGY_URL = (
    "https://api.brain-map.org/api/v2/structure_graph_download/1.json"
)
TEMPLATE_URL = (
    "https://download.alleninstitute.org/informatics-archive/current-release/"
    "mouse_ccf/average_template/average_template_25.nrrd"
)


def download_file(url: str, destination: Path, description: str) -> Path:
    """Download atomically so interrupted downloads are not mistaken for data."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {description}...")
    print(f"  From: {url}")
    print(f"  To:   {destination}")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            with temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Downloaded {description}.")
    return destination


def validate_annotation(path: Path) -> None:
    """Perform inexpensive checks before accepting a cached annotation file."""
    if not path.exists() or path.stat().st_size < 100_000:
        raise RuntimeError(f"Allen annotation download looks incomplete: {path}")
    with path.open("rb") as stream:
        header = stream.read(8)
    if not header.startswith(b"NRRD"):
        raise RuntimeError(f"Allen annotation is not a valid NRRD file: {path}")


def validate_template(path: Path) -> None:
    if not path.exists() or path.stat().st_size < 1_000_000:
        raise RuntimeError(f"Allen template download looks incomplete: {path}")
    with path.open("rb") as stream:
        header = stream.read(8)
    if not header.startswith(b"NRRD"):
        raise RuntimeError(f"Allen template is not a valid NRRD file: {path}")


def validate_ontology(path: Path) -> None:
    if not path.exists() or path.stat().st_size < 1_000:
        raise RuntimeError(f"Allen structure tree download looks incomplete: {path}")
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or not data.get("msg"):
        raise RuntimeError(f"Allen structure tree has an unexpected format: {path}")


def ensure_allen_ccf_resources(
    resource_dir: Path | None = None,
    allow_download: bool = True,
) -> tuple[Path, Path]:
    """Return annotation/ontology paths, downloading missing files once."""
    destination = (resource_dir or DEFAULT_RESOURCE_DIR).expanduser().resolve()
    annotation = destination / ANNOTATION_NAME
    ontology = destination / ONTOLOGY_NAME

    if not annotation.exists():
        if not allow_download:
            raise FileNotFoundError(f"Missing Allen annotation: {annotation}")
        download_file(ANNOTATION_URL, annotation, "Allen CCF annotation (25 um)")
    if not ontology.exists():
        if not allow_download:
            raise FileNotFoundError(f"Missing Allen structure tree: {ontology}")
        download_file(ONTOLOGY_URL, ontology, "Allen mouse structure tree")

    validate_annotation(annotation)
    validate_ontology(ontology)
    return annotation, ontology


def ensure_allen_template(
    resource_dir: Path | None = None,
    allow_download: bool = True,
) -> Path | None:
    """Return the Allen average template, downloading it when permitted."""
    destination = (resource_dir or DEFAULT_RESOURCE_DIR).expanduser().resolve()
    template = destination / TEMPLATE_NAME
    if not template.exists():
        if not allow_download:
            return None
        download_file(TEMPLATE_URL, template, "Allen average template (25 um)")
    validate_template(template)
    return template


def main() -> None:
    try:
        annotation, ontology = ensure_allen_ccf_resources()
        template = ensure_allen_template()
    except Exception as exc:
        print(f"\nCould not prepare Allen CCF resources: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print("\nAllen CCF resources are ready:")
    print(f"  Annotation: {annotation}")
    print(f"  Template:   {template}")
    print(f"  Ontology:   {ontology}")


if __name__ == "__main__":
    main()
