"""TuneBench 日志工具。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


_DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_RESULT_LOGGER_NAME = "tunebench.result"
_LOG_DIRNAME = ".log"
_APP_LOG_FILENAME = "tunebench.log"
_RESULT_LOG_FILENAME = "result.log"


def _get_project_root() -> Path:
    """获取项目根目录。"""
    return Path(__file__).resolve().parents[3]


def _ensure_log_dir() -> Path:
    """确保项目根目录下的 .log 文件夹存在。"""
    log_dir = _get_project_root() / _LOG_DIRNAME
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _has_file_handler(logger: logging.Logger, file_path: Path) -> bool:
    """判断日志器是否已绑定到指定文件。"""
    target = str(file_path.resolve())
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == target:
            return True
    return False


def setup_logging(level: int = logging.INFO) -> None:
    """初始化 TuneBench 默认日志配置。"""
    log_dir = _ensure_log_dir()
    app_log_path = log_dir / _APP_LOG_FILENAME
    result_log_path = log_dir / _RESULT_LOG_FILENAME

    logger = logging.getLogger("tunebench")
    if logger.handlers:
        logger.setLevel(level)
    else:
        logger.setLevel(level)
        logger.propagate = False

        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(_DEFAULT_LOG_FORMAT, _DEFAULT_DATE_FORMAT))
        logger.addHandler(handler)

    if not _has_file_handler(logger, app_log_path):
        file_handler = logging.FileHandler(app_log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(_DEFAULT_LOG_FORMAT, _DEFAULT_DATE_FORMAT))
        logger.addHandler(file_handler)

    result_logger = logging.getLogger(_RESULT_LOGGER_NAME)
    if not result_logger.handlers:
        result_logger.setLevel(level)
        result_logger.propagate = False
        result_handler = logging.StreamHandler(sys.stdout)
        result_handler.setLevel(level)
        result_handler.setFormatter(logging.Formatter("%(message)s"))
        result_logger.addHandler(result_handler)
    else:
        result_logger.setLevel(level)

    if not _has_file_handler(result_logger, result_log_path):
        result_file_handler = logging.FileHandler(result_log_path, encoding="utf-8")
        result_file_handler.setLevel(level)
        result_file_handler.setFormatter(logging.Formatter("%(message)s"))
        result_logger.addHandler(result_file_handler)


def get_logger(name: str) -> logging.Logger:
    """获取 TuneBench 子日志器。"""
    return logging.getLogger(f"tunebench.{name}")


def get_result_logger() -> logging.Logger:
    """获取用于输出命令结果的日志器。"""
    return logging.getLogger(_RESULT_LOGGER_NAME)