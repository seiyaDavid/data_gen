from loguru import logger
import sys
import os
from ..src.config import Config


def configure_logging():
    config = Config.from_environment()
    log_path = os.path.join(config.log_dir, "app.log")

    logger.remove()
    logger.add(sys.stderr, level=config.log_level)
    logger.add(log_path, level=config.log_level, rotation="10 MB", retention="10 days")

    return logger


logger = configure_logging()
