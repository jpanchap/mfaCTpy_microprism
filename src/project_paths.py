"""Shared project path helpers for command-line scripts."""

from pathlib import Path


def resolve_project_path(project_path: str | Path | None = None) -> Path:
    """Resolve the project folder, defaulting to the current working directory.

    A project folder is expected to contain a ``data`` directory. When the
    current working directory is inside a project, such as ``data`` or
    ``data/processed``, this walks upward and returns the project root.
    """
    if project_path:
        resolved = Path(project_path).expanduser().resolve()
        print(f"Project path: {resolved}")
        return resolved

    start = Path.cwd().expanduser().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "data").is_dir():
            print(f"Project path: {candidate} (current folder)")
            return candidate

    raise FileNotFoundError(
        "Could not find a project folder from the current location.\n"
        "Please cd into the project folder, for example:\n"
        "  cd /Users/jojop/microPrismRegistration/stuber-lab/testA\n"
        "Then run the script again."
    )
