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
from aicocode.Permissions import PermissionMode

def main() -> None:
    Path(".aicocode").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        filename=".aicocode/debug.log",
        filemode="w",
    )

    parser = argparse.ArgumentParser(prog="aicocode", description="AicoCode AI coding assistant")
    parser.add_argument(
        "--permissionmode",
        choices=[m.value for m in PermissionMode],
        default=None,
        help="Permission mode (overrides config.yaml)",
    )

    args = parser.parse_args()

    try:
        config = load_config()
    except ConfigError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    mode_str = args.permissionmode if args.permissionmode else config.permission_mode
    permission_mode = PermissionMode(mode_str)

    from aicocode.app import CodeApp
    from aicocode.driver import NoAltScreenDriver

    app = CodeApp(
        providers=config.providers,
        permission_mode=permission_mode,
        driver_class=NoAltScreenDriver,
        sandbox_config=config.sandbox,
        mcp_servers=config.mcp_servers,
    )
    app.run()