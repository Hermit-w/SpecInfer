import logging
from typing import Literal, Optional

LOGGING_LEVEL = Literal['info', 'debug', 'critical', 'error', 'fatal', 'warning']  # noqa: E501


def setup_logger(log_file: Optional[str] = None, level: LOGGING_LEVEL = "info"):
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        filename=log_file,
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
