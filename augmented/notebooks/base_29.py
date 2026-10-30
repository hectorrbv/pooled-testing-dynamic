"""Protect a hand-edited base of notebook 29 from automatic rebuilds.

Héctor trimmed the presentation notebook by hand on 2026-10-20 and declared it the
base. The notebook carries ``metadata['manual_base']`` with that stamp; every
build script asks ``manual_base(path)`` before writing and skips the write unless
the environment variable ``REBUILD_29=1`` is set.
"""
import os
from pathlib import Path

import nbformat as nb

STAMP = '2026-10-20T18:44 manual base (Héctor)'


def manual_base(path):
    """Return the stamp if the notebook at ``path`` is a protected base, else None."""
    if os.environ.get('REBUILD_29') == '1':
        return None
    path = Path(path)
    if not path.exists():
        return None
    try:
        return nb.read(path, as_version=4).metadata.get('manual_base')
    except Exception:
        return None


def write_unless_base(book, path):
    stamp = manual_base(path)
    if stamp:
        print(f'skip {Path(path).name}: protected base ({stamp}); set REBUILD_29=1 to overwrite')
        return False
    nb.write(book, path)
    return True


def mark(path, stamp=STAMP):
    path = Path(path)
    book = nb.read(path, as_version=4)
    book.metadata['manual_base'] = stamp
    nb.write(book, path)
    return stamp
