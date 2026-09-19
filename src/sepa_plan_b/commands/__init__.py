"""Public command entry point; importing it does not load PyTorch.

The original top-level package modules remain the runtime for recorded
experiments. Command parsing and orchestration live separately here.
"""

import json
import sys

from .parser import build_parser


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    forwarded = []
    if "--" in arguments:
        separator = arguments.index("--")
        arguments, forwarded = arguments[:separator], arguments[separator + 1 :]

    parser = build_parser()
    args = parser.parse_args(arguments)
    if forwarded and (args.command != "experiment" or args.action != "run"):
        parser.error("Arguments after -- are only supported by experiment run")

    if args.command == "experiment":
        from .experiments import execute

        return execute(args, forwarded, parser)

    from .handlers import dispatch

    result = dispatch(args, parser)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0
