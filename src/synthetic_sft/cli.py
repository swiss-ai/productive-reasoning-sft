from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthetic_sft.config import load_config
from synthetic_sft.prepare import prepare_source
from synthetic_sft.schemas import quality_details_schema


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synthetic-sft")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="materialize normalized source seeds")
    prepare.add_argument("config", type=Path)
    prepare.add_argument("--force", action="store_true")

    run = subparsers.add_parser("run", help="run generation, quality, and selection")
    run.add_argument("config", type=Path)
    run.add_argument("--force-prepare", action="store_true")

    submit_parser = subparsers.add_parser("submit", help="submit one Slurm allocation")
    submit_parser.add_argument("config", type=Path)
    submit_parser.add_argument("--dry-run", action="store_true")

    schema = subparsers.add_parser("quality-schema", help="print quality JSON Schema")
    schema.add_argument("--compact", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "quality-schema":
        indent = None if args.compact else 2
        print(json.dumps(quality_details_schema(), indent=indent, sort_keys=True))
        return

    config = load_config(args.config)
    if args.command == "prepare":
        print(prepare_source(config, force=args.force))
    elif args.command == "run":
        from synthetic_sft.pipeline import run_pipeline

        print(run_pipeline(config, force_prepare=args.force_prepare))
    elif args.command == "submit":
        from synthetic_sft.submit import submit

        print(submit(config, args.config, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
