"""`python -m aicocode` 入口。"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

from aicocode.config import ConfigError, load_config

def main() -> None:
    Path(".aicocode").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        filename=".aicocode/debug.log",
        filemode="w",
    )

    try:
        config = load_config()
    except ConfigError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    from aicocode.app import CodeApp
    from aicocode.driver import NoAltScreenDriver

    app = CodeApp(
        providers=config.providers,
        driver_class=NoAltScreenDriver,
    )
    app.run()