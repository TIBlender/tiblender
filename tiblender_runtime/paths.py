"""Host-independent path rendering for generated output files.

Run artifacts (``cycle_summary.json``, ``cycle_record.json``, ...) are
committed as expected output and compared across machines and between the
host and the container, so they must never embed an absolute host path such
as ``/home/alice/checkout/output/cycle_001``. Every path they record is
rendered relative to the run's own output root instead.
"""

from __future__ import annotations

import os
from pathlib import Path


def relative_path(path: Path | str, root: Path | str) -> str:
    """Render ``path`` relative to ``root`` as a POSIX string.

    Falls back to ``os.path.relpath`` (which may produce ``../``) when
    ``path`` lies outside ``root`` -- e.g. an explicitly configured
    ``TIBLENDER_HISTORY_DIR`` pointing elsewhere -- and finally to the bare
    file name if even that is impossible. The result never contains an
    absolute host path.
    """
    path = Path(path)
    root = Path(root)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        pass
    try:
        return Path(os.path.relpath(path.resolve(), root.resolve())).as_posix()
    except (OSError, ValueError):  # pragma: no cover - only on exotic filesystems
        return path.name
