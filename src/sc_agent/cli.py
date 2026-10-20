"""Phase 0 command-line checks."""

import argparse
import json

from .config import get_settings
from .models import ResearchRequest
from .phase1 import Phase1Executor
from .sc_rag_client import ScRagClient


def main() -> None:
    parser = argparse.ArgumentParser(prog="sc-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="check the SC-RAG service boundary")
    check.add_argument("--query", default=None)
    run = subparsers.add_parser("run", help="run the deterministic Phase 1 retrieval workflow")
    run.add_argument("question")
    run.add_argument("--paper", action="append", dest="paper_aliases", default=[])
    args = parser.parse_args()

    if args.command == "check":
        client = ScRagClient(get_settings())
        print(client.health())
        if args.query:
            print(client.search(args.query))
    elif args.command == "run":
        result = Phase1Executor().run(ResearchRequest(question=args.question, paper_aliases=args.paper_aliases))
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
