#!/usr/bin/env python3
"""Canonical Linux/macOS source, artifact and real-collector acceptance gate."""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import urllib.request

COLLECTORS = {
    ('Linux', 'x86_64'): ('linux_amd64', '5415b8daf782f17cc463c3e46816abf181a68a04b7bcf98c273c3c204096c743'),
    ('Darwin', 'arm64'): ('darwin_arm64', 'a56143a40a2c205cdd63da4af0249f2534691785c14cdd2a0c532becb4335521'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    artifacts = args.artifacts.resolve()
    artifacts.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.pop('VIRTUAL_ENV', None)
    env.pop('UV_PROJECT_ENVIRONMENT', None)

    def run(name, command, cwd=source):
        with (work / (name + '.log')).open('w') as log:
            result = subprocess.run(list(map(str, command)), cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(name, 'exit', result.returncode, flush=True)
        if result.returncode:
            print((work / (name + '.log')).read_text(), file=sys.stderr)
            raise SystemExit(result.returncode)

    run('source-sync', ['uv', 'sync', '--frozen', '--extra', 'dev'])
    run('source-tests', ['uv', 'run', '--frozen', 'pytest'])
    reverse = "import pytest\nclass Reverse:\n def pytest_collection_modifyitems(self, items): items.reverse()\nraise SystemExit(pytest.main(['tests'],plugins=[Reverse()]))"
    run('source-reverse', ['uv', 'run', '--frozen', 'python', '-c', reverse])
    snapshot = work / 'source'
    run('export', ['uv', 'run', '--frozen', 'python', 'scripts/export_standalone.py', snapshot])
    run('build', ['uv', 'build', '--out-dir', artifacts], cwd=snapshot)
    run('release', [sys.executable, source / 'scripts/verify_release.py', '--artifacts', artifacts, '--work-dir', work / 'lifecycle'])
    target, expected = COLLECTORS[(platform.system(), platform.machine())]
    archive = work / 'collector.tar.gz'
    url = f'https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v0.160.0/otelcol_0.160.0_{target}.tar.gz'
    with urllib.request.urlopen(url, timeout=60) as response, archive.open('wb') as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise ValueError('collector archive digest mismatch')
    collector_dir = work / 'collector-bin'
    collector_dir.mkdir()
    with tarfile.open(archive) as package:
        package.extract('otelcol', collector_dir, filter='data')
    environment = work / 'collector-env'
    run('collector-env', ['uv', 'venv', environment, '--python', '3.12'])
    python = environment / 'bin/python'
    wheels = list(artifacts.glob('*.whl'))
    if len(wheels) != 1:
        raise ValueError('exactly one wheel is required')
    run('collector-install', ['uv', 'pip', 'install', '--python', python, wheels[0]])
    run('collector-readback', [python, source / 'scripts/verify_observability.py', '--collector', collector_dir / 'otelcol', '--work-dir', work / 'collector-evidence'])


if __name__ == '__main__':
    main()
