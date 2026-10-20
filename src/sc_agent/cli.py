"""Phase 0 command-line checks."""

import argparse

from .config import get_settings
from .sc_rag_client import ScRagClient


def main() -> None:
    parser = argparse.ArgumentParser(prog="sc-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="check the SC-RAG service boundary")
    check.add_argument("--query", default=None)
    args = parser.parse_args()

    if args.command == "check":
        client = ScRagClient(get_settings())
        print(client.health())
        if args.query:
            print(client.search(args.query))
