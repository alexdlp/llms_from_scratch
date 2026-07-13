import shutil
from pathlib import Path
from ..logger import logger

def clean_up(
    dirs_to_remove=("outputs", "multirun", "mlruns", "logs"),
    dry_run: bool = False
) -> None:
    """
    Removes project runtime directories such as outputs, logs, and mlruns.

    Args:
        dirs_to_remove (tuple): Directory names to remove from the project root.
        dry_run (bool): If True, only logs what would be removed.
    """
    for name in dirs_to_remove:
        for path in Path(".").glob(name):
            if path.exists() and path.is_dir():
                if dry_run:
                    logger.info(f"[clean_up] Would remove: {path}")
                else:
                    shutil.rmtree(path)
                    logger.info(f"[clean_up] Removed: {path}")
            else:
                logger.debug(f"[clean_up] Skipped (not found): {path}")
