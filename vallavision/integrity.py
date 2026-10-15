"""Integridad de artefactos: hash de un checkpoint (directorio) y del código.

Sin dependencias: lo usan el DAG (al registrar), el servicio (al cargar el champion,
para no servir pesos alterados en MinIO).
"""

import hashlib
import os

# metadata.json y .lock los escribe el DAG en el prefijo de la versión: no son parte del checkpoint
EXCLUDE = ("metadata.json", ".lock")


def sha256_dir(path, exclude=EXCLUDE):
    """SHA-256 de los nombres y contenidos de todos los archivos bajo `path`, en orden."""
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
    """Commit de HEAD leyendo `.git` a mano (sin el binario de git); None si no se puede."""
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
