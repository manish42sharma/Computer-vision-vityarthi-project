"""
utils.py
--------
Shared, cross-cutting helpers: logging setup, error-handling decorators,
and small validation utilities used across every module.

Centralising these concerns here is what implements the project's
"logging/monitoring" and "error handling strategy" non-functional
requirements in one auditable place instead of scattering ad-hoc
print statements through the codebase.
"""

import functools
import logging
import os
import sys
import time

from src import config


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-level logger that writes to both stdout and a
    rotating log file under logs/. Safe to call repeatedly - handlers
    are only attached once per logger name.
    """
    config.ensure_directories()
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, config.LOGGING_LEVEL, logging.INFO))
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_path = os.path.join(config.LOG_DIR, "sign_language_system.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def timeit(logger: logging.Logger = None):
    """Decorator that logs how long a function took (performance monitoring)."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            log = logger or get_logger(func.__module__)
            log.info("%s completed in %.3f seconds", func.__name__, elapsed)
            return result

        return wrapper

    return decorator


def safe_run(logger: logging.Logger = None):
    """
    Decorator implementing the project's error-handling strategy:
    catch, log with context, and fail with a clean, non-zero exit code
    instead of an unhandled stack trace reaching the terminal user.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log = logger or get_logger(func.__module__)
            try:
                return func(*args, **kwargs)
            except FileNotFoundError as exc:
                log.error("Required file not found: %s", exc)
                sys.exit(2)
            except ValueError as exc:
                log.error("Invalid value / bad input: %s", exc)
                sys.exit(3)
            except Exception as exc:  # pragma: no cover - last line of defence
                log.exception("Unexpected error in %s: %s", func.__name__, exc)
                sys.exit(1)

        return wrapper

    return decorator


def validate_path_exists(path: str, description: str = "Path") -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{description} does not exist: {path}")


def validate_non_empty(value, description: str = "Value") -> None:
    if value is None or (hasattr(value, "__len__") and len(value) == 0):
        raise ValueError(f"{description} must not be empty")
