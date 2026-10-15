"""Artifact integrity: hashing a checkpoint directory and reading the code version.

No dependencies: used by the DAG (when registering) and by the service (when loading the
champion, so that weights tampered with in MinIO are never served).
"""

import hashlib
import os

# The DAG writes metadata.json and .lock under the version prefix: they are not part of the checkpoint
EXCLUDE = ("metadata.json", ".lock")


def sha256_dir(path, exclude=EXCLUDE):
    """SHA-256 of the names and contents of every file under `path`, in a fixed order."""
    h = hashlib.sha256()
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(files):
            if name in exclude:
                continue
            h.update(name.encode())
            with open(os.path.join(root, name), "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
    return h.hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(git_dir):
    """HEAD commit, read straight from `.git` (no git binary needed); None if unavailable."""
    try:
        with open(os.path.join(git_dir, "HEAD")) as f:
            head = f.read().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        ref_path = os.path.join(git_dir, ref)
        if os.path.exists(ref_path):
            with open(ref_path) as f:
                return f.read().strip()
        with open(os.path.join(git_dir, "packed-refs")) as f:
            for line in f:
                if line.rstrip().endswith(" " + ref):
                    return line.split()[0]
    except OSError:
        pass
    return None
