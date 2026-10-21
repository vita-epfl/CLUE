# Copyright (c) Meta Platforms, Inc. and affiliates.
import functools
import logging
import os


def _rank() -> int:
    for name in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        value = os.environ.get(name)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                return 0
    return 0


def rank_zero_only(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if _rank() == 0:
            return fn(*args, **kwargs)
        return None

    return wrapped


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    )
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger
