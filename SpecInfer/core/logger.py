import logging
import os
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from _typeshed import OpenTextMode

LOGGING_LEVEL = Literal["info", "debug", "critical", "error", "fatal", "warning"]


def setup_logger(
    log_path: Optional[str] = None,
    level: LOGGING_LEVEL = "info",
    *,
    rank: Optional[int] = None,
    file_mode: "OpenTextMode" = "a",
    force_console: bool = False
):
    """
    setup logger that is compatiable with distributed system
    Args:
        log_path: directory to save log file. If None, only log to console.
        level: logging level, default to "info".
            Can be one of ["info", "debug", "critical", "error", "fatal", "warning"].
        rank: rank of the current process. If None, will read from environment variable "RANK".
        force_console: if True, log to console while logging to file when log_path is provided.
    """
    rank = rank if rank is not None else int(os.environ.get("RANK", 0))
    log_level = getattr(logging, level.upper(), logging.INFO)

    # basic formatter
    formatter = logging.Formatter(
        f"[RANK{rank}] %(asctime)s [%(levelname)s] %(name)s: %(lineno)d - %(message)s",
        datefmt="%y-%m-%d %H:%M:%S"
    )

    if log_path:
        os.makedirs(log_path, exist_ok=True)

    handlers: list[logging.Handler] = []

    # file handler
    if log_path:
        file_handler = logging.FileHandler(
            os.path.join(log_path, f"log_file_{rank}.log"), mode=file_mode, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
        if force_console:
            # console handler
            console = logging.StreamHandler()
            console.setFormatter(formatter)

            handlers.append(console)
    else:
        # console handler
        console = logging.StreamHandler()
        console.setFormatter(formatter)

        handlers.append(console)

    # configure root logger
    logging.basicConfig(
        handlers=handlers,
        level=log_level,
        force=True,
    )
