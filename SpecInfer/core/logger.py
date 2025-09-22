import logging
import os
from typing import Optional, Literal

LOGGING_LEVEL = Literal["info", "debug", "critical", "error", "fatal", "warning"]

def setup_logger(log_path: Optional[str] = None,
                 level: LOGGING_LEVEL = "info",
                 *,
                 rank: Optional[int] = None,
                 force_console: bool = False):
    """
    分布式安全日志
    :param log_file:  基础文件名，例如 'train.log'；实际写入 '{rank}_train.log'
    :param level:     日志级别
    :param rank:      当前全局 rank；None 时自动从环境变量 RANK 读取
    :param force_console: 为 True 时强制输出到控制台（多机调试方便）
    """
    rank = rank if rank is not None else int(os.environ.get("RANK", 0))
    log_level = getattr(logging, level.upper(), logging.INFO)

    # 1. 日志格式：强制带上 [RANKx]
    formatter = logging.Formatter(
        f"[RANK{rank}] %(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S"
    )

    # 2. 控制台 handler（每个进程都有，方便实时看）
    console = logging.StreamHandler()
    console.setFormatter(formatter)

    handlers: list[logging.Handler] = [console]

    # 3. 文件 handler：只有 RANK0 写文件，避免冲突
    if log_path and not force_console:
        # 如果想让每个 RANK 各自写文件，把下面 if 判断去掉即可
        file_handler = logging.FileHandler(os.path.join(log_path,f"log_file_{rank}"), mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # 4. 一次性配置，避免重复 addHandler
    logging.basicConfig(
        handlers=handlers,
        level=log_level,
        force=True,
    )