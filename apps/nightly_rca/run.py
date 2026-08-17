#!/usr/bin/env python3
"""CLI for the serial nightly RCA pipeline."""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from dataclasses import replace
from pathlib import Path

from .config import (
    Settings,
    build_effective_configuration_provenance,
    current_bootstrap_provenance,
)
from .pipeline import NightlyPipeline, PHASE_ORDER
from .state import RunStore
from .transport import McpToolClient


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Nightly RCA v6 serial learning pipeline")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--commit", action="store_true", help="Enable authorized writes")
    mode.add_argument("--dry-run", action="store_true", help="Force preview/read-only mode")
    p.add_argument("--resume", type=Path, help="Resume from a runs/<id>/state.json checkpoint")
    p.add_argument("--stop-after", choices=PHASE_ORDER)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--no-notify", action="store_true")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return p


async def _main(args: argparse.Namespace) -> int:
    settings = Settings()
    if args.output_dir:
        settings = replace(settings, output_dir=args.output_dir)
    if args.commit:
        settings = replace(settings, commit=True)
    elif args.dry_run:
        settings = replace(settings, commit=False)
    if args.no_notify:
        settings = replace(settings, notify=False)

    state = None
    store = None
    if args.resume:
        store, state = RunStore.load(args.resume)
        settings = replace(settings, output_dir=store.output_dir, commit=(state.mode == "commit"))

    # code_tools_mcp is optional — degrade gracefully if not configured
    OPTIONAL_SERVERS = {"code_tools_mcp"}
    missing_urls = [
        name for name, url in settings.server_urls.items()
        if not url and name not in OPTIONAL_SERVERS
    ]
    if missing_urls:
        logging.getLogger("nightly_rca.run").error(
            "missing MCP URL configuration for: %s", ", ".join(sorted(missing_urls))
        )
        return 3

    bootstrap_env_path, bootstrap_keys = current_bootstrap_provenance()
    cli_mode_source = "CLI_COMMIT" if args.commit else "CLI_DRY_RUN" if args.dry_run else ""
    effective_configuration = build_effective_configuration_provenance(
        settings,
        process_env=os.environ,
        cli_mode_source=cli_mode_source,
        resumed=bool(args.resume),
        env_file_path=os.environ.get("NIGHTLY_RCA_LAUNCHER_ENV_SOURCE") or bootstrap_env_path,
        bootstrapped_keys=bootstrap_keys,
    )

    async with McpToolClient(settings.server_urls, aws_region=settings.aws_region) as client:
        pipeline = NightlyPipeline(settings=settings, client=client, state=state, store=store)
        pipeline.state.data["effective_configuration"] = effective_configuration
        pipeline.store.save(pipeline.state)
        return await pipeline.run(stop_after=args.stop_after)


def main() -> int:
    args = parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )
    return asyncio.run(_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
