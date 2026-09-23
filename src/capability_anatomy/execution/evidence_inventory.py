"""One descriptor-bound, streaming inventory for both evidence formats."""
from __future__ import annotations

import os
from pathlib import Path
import stat
from typing import Callable

from ..errors import InvalidEvidenceError
from . import secure_fs


def inventory_paths(output: Path, *, max_entries: int, max_depth: int,
                    error: Callable[[str], InvalidEvidenceError],
                    reclaim_temporaries: bool = False) -> list[str]:
    paths = []
    entries = 0
    linked = []
    staged = set()

    def walk(descriptor, prefix="", depth=0):
        nonlocal entries
        if depth > max_depth:
            raise error("evidence_directory_depth_limit")
        with os.scandir(descriptor) as directory:
            for entry in directory:
                entries += 1
                if entries > max_entries:
                    raise error("evidence_entry_limit")
                name = entry.name
                if name in (".", "..") or "/" in name or "\\" in name:
                    raise error("evidence_path_invalid")
                item = entry.stat(follow_symlinks=False)
                relative = prefix + name
                if secure_fs.temporary_target(name) is not None:
                    if reclaim_temporaries:
                        secure_fs.reclaim_temporary(descriptor, name)
                    else:
                        secure_fs.validate_temporary(descriptor, name)
                        staged.add((item.st_dev, item.st_ino))
                    continue
                if stat.S_ISDIR(item.st_mode):
                    child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                    try:
                        walk(child, relative + "/", depth + 1)
                    finally:
                        os.close(child)
                else:
                    # Recovery visits every entry, but a published create_once
                    # target can still have its second link until we reach the
                    # staging alias. Validate authoritative leaves on read.
                    if not reclaim_temporaries:
                        if stat.S_ISREG(item.st_mode) and item.st_nlink == 2:
                            linked.append(item)
                        else:
                            secure_fs._regular(item)
                    paths.append(relative)

    descriptor = secure_fs._open_directory(output)
    try:
        walk(descriptor)
        for item in linked:
            if (item.st_dev, item.st_ino) in staged:
                raise secure_fs.StorageError("storage_recovery_required")
            secure_fs._regular(item)
    finally:
        os.close(descriptor)
    return sorted(paths)
