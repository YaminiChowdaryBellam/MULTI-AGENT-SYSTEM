"""
python -m evals — runs the Phase 4 evaluation harness:
  1. Routing accuracy over the ~50-query gold set (Step 4.1)
  2. LLM-as-judge faithfulness + citation coverage over the ~25 e2e cases (Step 4.2)
Then writes the aggregate metrics to evals/results/ and prints a regression
comparison against the previous run (Step 4.4).

Human-in-the-loop review flagging (Step 4.3) isn't a separate eval pass —
it's exercised live inside the judge eval's full graph runs, and its rate
(review_rate) is reported and tracked as one of the metrics.
"""

import argparse
import sys
from collections import defaultdict

from dotenv import load_dotenv

load_dotenv()

from evals.judge_eval import evaluate_judge  # noqa: E402
from evals.results import compare, latest_results, write_results  # noqa: E402
from evals.routing_eval import evaluate_routing  # noqa: E402


def _print_routing_summary(routing: dict) -> None:
    print("=" * 60)
    print("ROUTING ACCURACY  (Step 4.1)")
    print("=" * 60)
    print(f"  n_queries          : {routing['n_queries']}")
    print(f"  routing_accuracy   : {routing['routing_accuracy']:.1%}  (exact agent-set match)")
    print(f"  routing_precision  : {routing['routing_precision']:.1%}")
    print(f"  routing_recall     : {routing['routing_recall']:.1%}")
    print(f"  routing_f1         : {routing['routing_f1']:.1%}")

    by_category = defaultdict(list)
    for p in routing["per_query"]:
        by_category[p["category"]].append(p["exact_match"])
    print("\n  By category:")
    for category, matches in sorted(by_category.items()):
        acc = sum(matches) / len(matches)
        print(f"    {category:<20} {acc:.1%}  ({sum(matches)}/{len(matches)})")

    misses = [p for p in routing["per_query"] if not p["exact_match"]]
    if misses:
        print(f"\n  Misses ({len(misses)}):")
        for p in misses:
            print(f"    [{p['id']}] expected={p['expected_agents']} actual={p['actual_agents']}")
            print(f"           {p['query'][:90]}")


def _print_judge_summary(judge: dict) -> None:
    print()
    print("=" * 60)
    print("LLM-AS-JUDGE  (Step 4.2, end-to-end)")
    print("=" * 60)
    print(f"  n_queries          : {judge['n_queries']}")
    print(f"  faithfulness       : {judge['faithfulness']:.1%}")
    print(f"  citation_coverage  : {judge['citation_coverage']:.1%}")
    print(f"  review_rate        : {judge['review_rate']:.1%}  (Step 4.3 — flagged instead of returned)")

    weak = [p for p in judge["per_query"] if p["faithfulness"] < 0.8]
    if weak:
        print(f"\n  Weak faithfulness (<80%) ({len(weak)}):")
        for p in weak:
            print(f"    [{p['id']}] faithfulness={p['faithfulness']:.0%}  {p['query'][:70]}")
            if p["unsupported_claims"]:
                print(f"           unsupported: {p['unsupported_claims']}")


def _print_regressions(current: dict, previous_record: dict | None, tolerance: float) -> bool:
    """Returns True if any metric regressed beyond tolerance."""
    print()
    print("=" * 60)
    print("REGRESSION CHECK  (Step 4.4)")
    print("=" * 60)
    if previous_record is None:
        print("  No previous results to compare against — this is the baseline run.")
        return False

    print(f"  Comparing against: {previous_record.get('timestamp', '?')} "
          f"(git {previous_record.get('git_sha', '?')})")
    comparison = compare(current, previous_record, tolerance=tolerance)
    any_regression = False
    for metric, c in comparison.items():
        flag = " <-- REGRESSION" if c["regressed"] else ""
        if flag:
            any_regression = True
        print(f"    {metric:<20} {c['previous']:.4f} -> {c['current']:.4f}  (delta {c['delta']:+.4f}){flag}")
    if not any_regression:
        print("  No regressions detected.")
    return any_regression


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-Agent Clinical Research Assistant eval harness")
    parser.add_argument("--routing-limit", type=int, default=None, help="Limit routing eval to first N gold-set entries")
    parser.add_argument("--judge-limit", type=int, default=None, help="Limit judge eval to first N e2e-flagged entries")
    parser.add_argument("--skip-judge", action="store_true", help="Skip the expensive end-to-end judge pass")
    parser.add_argument("--tolerance", type=float, default=0.02, help="Regression tolerance (default 0.02)")
    parser.add_argument("--no-write", action="store_true", help="Don't write results to evals/results/")
    args = parser.parse_args(argv)

    routing = evaluate_routing(limit=args.routing_limit)
    _print_routing_summary(routing)

    metrics = {
        "routing_accuracy": routing["routing_accuracy"],
        "routing_precision": routing["routing_precision"],
        "routing_recall": routing["routing_recall"],
        "routing_f1": routing["routing_f1"],
    }

    if not args.skip_judge:
        judge = evaluate_judge(limit=args.judge_limit)
        _print_judge_summary(judge)
        metrics.update({
            "faithfulness": judge["faithfulness"],
            "citation_coverage": judge["citation_coverage"],
            "review_rate": judge["review_rate"],
        })

    # Compare against the baseline regardless of --no-write — CI runs on a PR
    # branch want the regression gate without committing a new baseline file;
    # only the post-merge workflow (on main) should write one.
    previous = latest_results()
    regressed = _print_regressions(metrics, previous, args.tolerance)

    if not args.no_write:
        path = write_results(metrics)
        print(f"\nResults written to {path}")

    return 1 if regressed else 0


if __name__ == "__main__":
    sys.exit(main())
