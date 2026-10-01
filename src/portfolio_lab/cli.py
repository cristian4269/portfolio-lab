"""Command-line entry point: `uv run portfolio-lab <command>`."""

import argparse
import sys

from portfolio_lab import __version__


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="portfolio-lab",
        description="Build portfolios, backtest them honestly, and measure their risk.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command")
    sub.add_parser("reproduce", help="regenerate every result in research/ (not built yet)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command is None:
        build_parser().print_help()
        return 0
    print(f"`{args.command}` is not implemented yet (see plans/portfolio-lab-blueprint.md).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
