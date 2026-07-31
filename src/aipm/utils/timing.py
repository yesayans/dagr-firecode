"""Stage timing so the pipeline can report where the seconds went."""

from __future__ import annotations

import time
from contextlib import contextmanager

from aipm.utils.logging import get_logger

log = get_logger(__name__)


@contextmanager
def stage(name: str):
    start = time.perf_counter()
    log.info("start %s", name)
    try:
        yield
    finally:
        log.info("done  %s in %.2fs", name, time.perf_counter() - start)
