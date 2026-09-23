#!/usr/bin/env python3
"""Verify built wheel lifecycle and the complete extracted-sdist test suite."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import stat
import sys
import tarfile
import zipfile


def run(command, *, cwd, env, log):
    command = [str(item) for item in command]
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    log.write_text("command: " + json.dumps(command) + "\n" + result.stdout + result.stderr)
    print(log.name, "exit", result.returncode, flush=True)
    if result.returncode:
        raise RuntimeError(f"release check failed: {log.name} (exit {result.returncode})")
    return result.stdout


def hashes(root):
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def verify_export(source):
    manifest = json.loads((source / "EXPORT-MANIFEST.json").read_text())
    actual = set()
    for path in source.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("source distribution contains a link or special file")
        actual.add(path.relative_to(source).as_posix())
    expected = {item["path"] for item in manifest["files"]} | {"EXPORT-MANIFEST.json", "PKG-INFO"}
    if actual != expected or len(manifest["files"]) != len(expected) - 2:
        raise ValueError("source distribution differs from the reviewed file selection")
    for item in manifest["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid source manifest path")
        path = source / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("source manifest file is unavailable")
        payload = path.read_bytes()
        if len(payload) != item["bytes"] or hashlib.sha256(payload).hexdigest() != item["sha256"]:
            raise ValueError("source manifest digest mismatch")
    digest = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != manifest["source_tree_sha256"]:
        raise ValueError("source manifest tree digest mismatch")
    return manifest


def verify(artifacts: Path, work: Path) -> dict:
    artifacts = artifacts.resolve(strict=True)
    wheels = list(artifacts.glob("*.whl"))
    sources = list(artifacts.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("exactly one wheel and one source distribution are required")
    if work.exists() or work.is_symlink():
        raise ValueError("verification work directory must be new")
    work.mkdir(mode=0o700)
    work = work.resolve()
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"}}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONNOUSERSITE="1")
    extracted = work / "sdist"
    extracted.mkdir()
    with tarfile.open(sources[0], "r:gz") as archive:
        names = set()
        for member in archive.getmembers():
            path = Path(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts or member.name in names:
                raise ValueError("source archive requires unique relative regular files")
            names.add(member.name)
        archive.extractall(extracted, filter="data")
    roots = list(extracted.iterdir())
    if len(roots) != 1 or not roots[0].is_dir() or roots[0].is_symlink():
        raise ValueError("source distribution must have one ordinary root directory")
    source = roots[0]
    manifest = verify_export(source)
    with zipfile.ZipFile(wheels[0]) as wheel:
        for item in manifest["files"]:
            if item["path"].startswith("src/capability_anatomy/"):
                wheel_path = item["path"].removeprefix("src/")
                if hashlib.sha256(wheel.read(wheel_path)).hexdigest() != item["sha256"]:
                    raise ValueError("wheel source differs from the source snapshot")
    environment = work / "installed-wheel"
    run([sys.executable, "-m", "venv", environment], cwd=work, env=env, log=work / "01-venv.log")
    environment_identity = (environment.stat().st_dev, environment.stat().st_ino)
    scripts = environment / ("Scripts" if os.name == "nt" else "bin")
    python = scripts / ("python.exe" if os.name == "nt" else "python")
    cli = scripts / ("capability-anatomy.exe" if os.name == "nt" else "capability-anatomy")
    run([python, "-m", "pip", "install", wheels[0]], cwd=work, env=env, log=work / "02-install.log")
    run([python, "-I", "-c", "import pathlib,sys,capability_anatomy; p=pathlib.Path(capability_anatomy.__file__).resolve(); print(p); assert p.is_relative_to(pathlib.Path(sys.prefix).resolve())"], cwd=work, env=env, log=work / "03-import-origin.log")
    doctor = json.loads(run([cli, "doctor"], cwd=work, env=env, log=work / "04-doctor.log"))
    if doctor.get("status") != "ok":
        raise ValueError("installed core doctor did not succeed")
    example = work / "example"
    created = json.loads(run([cli, "example", "--output", example], cwd=work, env=env, log=work / "05-example.log"))
    run([cli, "run", "--config", created["configuration"]], cwd=work, env=env, log=work / "06-run.log")
    run([cli, "verify", "--evidence", example / "run"], cwd=work, env=env, log=work / "07-verify.log")
    run([cli, "report", "--evidence", example / "run", "--format", "json"], cwd=work, env=env, log=work / "08-report.log")
    task_hashes = hashes(example / "run/tasks")
    if not task_hashes:
        raise ValueError("synthetic run did not produce resumable tasks")
    run([cli, "run", "--config", created["configuration"]], cwd=work, env=env, log=work / "08a-resume.log")
    if hashes(example / "run/tasks") != task_hashes:
        raise ValueError("compatible resume changed completed task bytes")
    run([cli, "verify", "--evidence", example / "run"], cwd=work, env=env, log=work / "08b-resumed-verify.log")
    original = hashes(example)
    run([python, "-m", "pip", "uninstall", "--yes", "capability-anatomy"], cwd=work, env=env, log=work / "09-uninstall.log")
    run([python, "-I", "-c", "import importlib.util; assert importlib.util.find_spec('capability_anatomy') is None"], cwd=work, env=env, log=work / "10-uninstalled-import.log")
    if cli.exists() or hashes(example) != original:
        raise ValueError("uninstall left the entry point or modified experiment evidence")
    run([python, "-m", "pip", "install", "--no-deps", wheels[0]], cwd=work, env=env, log=work / "11-reinstall.log")
    run([cli, "verify", "--evidence", example / "run"], cwd=work, env=env, log=work / "12-reinstalled-verify.log")
    if hashes(example) != original:
        raise ValueError("reinstallation or verification modified experiment evidence")
    run(["uv", "sync", "--frozen", "--extra", "dev"], cwd=source, env=env, log=work / "13-sdist-sync.log")
    run(["uv", "run", "--frozen", "pytest"], cwd=source, env=env, log=work / "14-sdist-tests.log")
    reverse = "import pytest\nclass Reverse:\n def pytest_collection_modifyitems(self, items): items.reverse()\nraise SystemExit(pytest.main(['tests'],plugins=[Reverse()]))\n"
    run(["uv", "run", "--frozen", "python", "-c", reverse], cwd=source, env=env, log=work / "15-sdist-reverse-tests.log")
    current = environment.lstat()
    if environment.is_symlink() or (current.st_dev, current.st_ino) != environment_identity:
        raise ValueError("verification environment ownership changed")
    shutil.rmtree(environment)
    if environment.exists() or hashes(example) != original:
        raise ValueError("complete environment removal failed or changed experiment evidence")
    result = {"status": "passed", "source_tree_sha256": manifest["source_tree_sha256"],
              "artifacts": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in [*wheels, *sources]},
              "experiment_files_preserved": len(original), "compatible_task_files_preserved": len(task_hashes),
              "environment_removed": True, "included_test_files": manifest["tests"]["included_files"],
              "license_present": manifest["license_present"], "release_status": manifest["release_status"]}
    (work / "verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(verify(args.artifacts, args.work_dir), sort_keys=True))
    except (OSError, ValueError, RuntimeError, tarfile.TarError, zipfile.BadZipFile) as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
