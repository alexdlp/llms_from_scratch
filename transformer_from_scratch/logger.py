import logging
from pathlib import Path
from datetime import datetime

# Nombre del logger global
LOGGER_NAME = "llms_from_scratch"

# Obtener el logger
logger = logging.getLogger(LOGGER_NAME)

# Configura solo consola por defecto
if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    logger.propagate = False  # No propagar a root

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(
        "[%(asctime)s] - %(levelname)s (%(filename)s:%(lineno)d) - %(message)s"
    ))

    logger.addHandler(stream_handler)

def add_file_handler(log_dir: Path) -> None:
    """
    Adds a FileHandler to the logger that logs into the given directory.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        "[%(asctime)s] - %(levelname)s (%(filename)s:%(lineno)d) - %(message)s"
    ))

    logger.addHandler(file_handler)
