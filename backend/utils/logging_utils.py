from __future__ import annotations

import logging
import os


def configure_logging() -> None:
    level_name = os.getenv("AI_ASSIST_LOG_LEVEL", "DEBUG").upper()
    level = getattr(logging, level_name, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
