"""`praman` command-line interface. Offline by default.

    praman verify --artifacts artifacts/praman-verifier \
        --output "The drug was approved in 2019." \
        --evidence "The agency approved the drug in 2021." --alpha 0.05
"""
from __future__ import annotations

import argparse
import sys

import orjson


def _pretty(obj) -> str:
    return orjson.dumps(obj, option=orjson.OPT_INDENT_2).decode()


def cmd_verify(args: argparse.Namespace) -> int:
    from .pipeline import Praman
    p = Praman.load(args.artifacts, offline=not args.online)
    evidence = args.evidence if len(args.evidence) > 1 else (args.evidence[0] if args.evidence else "")
    policy = {"class": args.policy_class} if args.policy_class else None
    out = p.verify(output_text=args.output, evidence=evidence, alpha=args.alpha, policy=policy)
    if args.audit_jsonl:
        from .audit import write_jsonl
        n = write_jsonl(out["audit"], args.audit_jsonl)
        print(f"[audit] wrote {n} records -> {args.audit_jsonl}", file=sys.stderr)
    print(_pretty(out))
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path
    man = json.loads((Path(args.artifacts) / "manifest.json").read_text(encoding="utf-8"))
    print(_pretty(man))
    return 0


def cmd_mapping(_: argparse.Namespace) -> int:
    from .audit import field_mapping
    print(_pretty(field_mapping()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="praman", description="On-prem grounded-claim verifier.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="verify an output against evidence")
    v.add_argument("--artifacts", required=True, help="artifact dir (Praman.load)")
    v.add_argument("--output", required=True, help="the generated output_text to check")
    v.add_argument("--evidence", nargs="+", required=True, help="one or more evidence passages")
    v.add_argument("--alpha", type=float, default=None, help="target missed-approval rate")
    v.add_argument("--policy-class", default=None, help="policy class (e.g. clinical, bulk)")
    v.add_argument("--audit-jsonl", default=None, help="also append audit records here")
    v.add_argument("--online", action="store_true", help="allow network (default offline)")
    v.set_defaults(func=cmd_verify)

    i = sub.add_parser("info", help="show artifact manifest")
    i.add_argument("--artifacts", required=True)
    i.set_defaults(func=cmd_info)

    m = sub.add_parser("mapping", help="print the regulator field mapping")
    m.set_defaults(func=cmd_mapping)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
