"""PEP 517 adapter: archive the same vetted snapshot as the standalone exporter.

Hatchling owns metadata, reproducibility and archive construction. The project
owns only its public source selection and descriptor-backed input validation.
Build frontends invoke this backend in a dedicated process. Editable installs
intentionally refer to the developer checkout and are not release artifacts.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
import tomllib

from hatchling import build as hatch
from export_standalone import export_snapshot


@contextmanager
def _snapshot():
    source = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="capability-anatomy-build-") as work:
        destination = Path(work).resolve() / "source"
        manifest = export_snapshot(source, destination)
        _validate_builder_paths(destination, {entry["path"] for entry in manifest["files"]})
        try:
            os.chdir(destination)
            yield
        finally:
            os.chdir(source)


def _validate_builder_paths(source, selected):
    """Do not let secondary Hatchling selection escape the vetted snapshot."""
    document = tomllib.loads((source / "pyproject.toml").read_text())
    project = document.get("project", {})
    readme = project.get("readme")
    readme_file = readme.get("file") if isinstance(readme, dict) else readme
    if readme_file is not None and readme_file not in selected:
        raise ValueError("release builder readme is outside the reviewed selection")
    if any(path not in selected for path in project.get("license-files", [])):
        raise ValueError("release builder license source is outside the reviewed selection")
    if project.get("dynamic"):
        raise ValueError("release builder requires static reviewed metadata")
    hatch_config = document.get("tool", {}).get("hatch", {})
    if set(hatch_config) - {"build"}:
        raise ValueError("release builder does not execute metadata hooks")
    build = hatch_config.get("build", {})
    if set(build) - {"targets"} or set(build.get("targets", {})) - {"wheel"}:
        raise ValueError("release builder supports only the reviewed wheel projection")
    wheel = build.get("targets", {}).get("wheel", {})
    if set(wheel) - {"packages", "force-include"} or wheel.get("packages") != ["src/capability_anatomy"]:
        raise ValueError("release builder package selection changed")
    for origin, target in wheel.get("force-include", {}).items():
        if origin not in selected or Path(origin).is_absolute() or ".." in Path(origin).parts:
            raise ValueError("release builder source is outside the reviewed selection")
        if Path(target).is_absolute() or ".." in Path(target).parts or not target.startswith("capability_anatomy/_schemas/"):
            raise ValueError("release builder destination is outside the package schema directory")


def build_sdist(sdist_directory, config_settings=None):
    destination = str(Path(sdist_directory).absolute())
    with _snapshot():
        return hatch.build_sdist(destination, config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    destination = str(Path(wheel_directory).absolute())
    metadata = str(Path(metadata_directory).absolute()) if metadata_directory else None
    with _snapshot():
        return hatch.build_wheel(destination, config_settings, metadata)


def get_requires_for_build_sdist(config_settings=None):
    with _snapshot():
        return hatch.get_requires_for_build_sdist(config_settings)


def get_requires_for_build_wheel(config_settings=None):
    with _snapshot():
        return hatch.get_requires_for_build_wheel(config_settings)


build_editable = hatch.build_editable
get_requires_for_build_editable = hatch.get_requires_for_build_editable
