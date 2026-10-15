"""
Ensure omnivoice package is importable from source
"""

import sys
from pathlib import Path

def ensure_omnivoice_importable(backend_dir: Path, logger=None) -> None:
    """
    Ensure the omnivoice package is importable by adding its source to sys.path.
    This handles the case where the editable install is missing/broken.
    """
    omnivoice_source = backend_dir.parent / "omnivoice"
    if omnivoice_source.exists() and str(omnivoice_source) not in sys.path:
        sys.path.insert(0, str(omnivoice_source))
        if logger:
            logger.info(f"Added omnivoice source to sys.path: {omnivoice_source}")
        else:
            print(f"[omnivoice_path] Added {omnivoice_source} to sys.path")
