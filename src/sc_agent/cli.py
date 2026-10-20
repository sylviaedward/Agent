"""Phase 0 command-line checks."""

import argparse
import json

from .config import get_settings
from .models import ResearchRequest, SimulationSpec
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
    graph = subparsers.add_parser("graph", help="run the Phase 2 LangGraph workflow")
    graph.add_argument("question")
    graph.add_argument("--paper", action="append", dest="paper_aliases", default=[])
    graph.add_argument("--x", type=float)
    graph.add_argument("--y", type=float)
    graph.add_argument("--length", type=int, default=100_000)
    graph.add_argument("--seed", type=int, default=7)
    phase4 = subparsers.add_parser("phase4", help="run Phase 4 extraction and metric alignment")
    phase4.add_argument("question")
    phase4.add_argument("--paper", action="append", dest="paper_aliases", default=[])
    phase4.add_argument("--x", type=float)
    phase4.add_argument("--y", type=float)
    phase4.add_argument("--length", type=int, default=100_000)
    phase4.add_argument("--seed", type=int, default=7)
    phase5 = subparsers.add_parser("phase5", help="run guarded LLM extraction and approval preparation")
    phase5.add_argument("question")
    phase5.add_argument("--paper", action="append", dest="paper_aliases", default=[])
    phase5.add_argument("--x", type=float)
    phase5.add_argument("--y", type=float)
    phase5.add_argument("--length", type=int, default=100_000)
    phase5.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.command == "check":
        client = ScRagClient(get_settings())
        print(client.health())
        if args.query:
            print(client.search(args.query))
    elif args.command == "run":
        result = Phase1Executor().run(ResearchRequest(question=args.question, paper_aliases=args.paper_aliases))
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    elif args.command == "graph":
        from .graph import Phase2Graph

        simulation = None
        if args.x is not None or args.y is not None:
            if args.x is None or args.y is None:
                parser.error("--x and --y must be provided together")
            simulation = SimulationSpec(x=args.x, y=args.y, length=args.length, seed=args.seed)
        result = Phase2Graph().invoke(
            ResearchRequest(question=args.question, paper_aliases=args.paper_aliases, simulation=simulation)
        )
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    elif args.command == "phase4":
        from .phase4 import Phase4Workflow

        simulation = None
        if args.x is not None or args.y is not None:
            if args.x is None or args.y is None:
                parser.error("--x and --y must be provided together")
            simulation = SimulationSpec(x=args.x, y=args.y, length=args.length, seed=args.seed)
        result = Phase4Workflow().run(
            ResearchRequest(question=args.question, paper_aliases=args.paper_aliases, simulation=simulation)
        )
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    elif args.command == "phase5":
        from .phase5 import OpenAICompatibleExtractionClient, Phase5Workflow

        simulation = None
        if args.x is not None or args.y is not None:
            if args.x is None or args.y is None:
                parser.error("--x and --y must be provided together")
            simulation = SimulationSpec(x=args.x, y=args.y, length=args.length, seed=args.seed)
        result = Phase5Workflow(OpenAICompatibleExtractionClient.from_env()).run(
            ResearchRequest(question=args.question, paper_aliases=args.paper_aliases, simulation=simulation)
        )
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
