"""
Automated benchmark evaluator — runs all 20 tasks and writes a report.

Usage:
    python evaluation/evaluator.py
    python evaluation/evaluator.py --tasks evaluation/benchmark_tasks.json \
                                   --output evaluation/results
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    id: str
    category: str
    task: str
    success: bool
    score: float
    iterations: int
    elapsed_s: float
    answer: str
    keywords_hit: list[str] = field(default_factory=list)
    keywords_miss: list[str] = field(default_factory=list)
    error: str | None = None


def score(answer: str, keywords: list[str]) -> tuple[float, list, list]:
    a = answer.lower()
    hit = [k for k in keywords if k.lower() in a]
    miss = [k for k in keywords if k.lower() not in a]
    return (len(hit) / len(keywords)) if keywords else 1.0, hit, miss


def run_evaluation(tasks_path: str, output_dir: str) -> list[TaskResult]:
    from agent.graph import run_agent

    with open(tasks_path) as f:
        tasks = json.load(f)

    results: list[TaskResult] = []

    for i, t in enumerate(tasks, 1):
        logger.info(f"[{i}/{len(tasks)}] {t['id']} — {t['task'][:60]}…")
        t0 = time.perf_counter()
        error = None
        answer = ""
        iterations = 0

        try:
            state = run_agent(t["task"])
            answer = state.get("final_answer", "")
            iterations = state.get("iteration", 0)
        except Exception as e:
            error = str(e)
            logger.error(f"  FAILED: {e}")

        elapsed = time.perf_counter() - t0
        sc, hit, miss = score(answer, t.get("keywords", []))

        r = TaskResult(
            id=t["id"], category=t["category"], task=t["task"],
            success=sc >= 0.5, score=sc, iterations=iterations,
            elapsed_s=round(elapsed, 2), answer=answer[:400],
            keywords_hit=hit, keywords_miss=miss, error=error,
        )
        results.append(r)
        logger.info(f"  score={sc:.0%}  steps={iterations}  time={elapsed:.1f}s  hit={hit}")

    _save(results, output_dir)
    _summary(results)
    return results


def _save(results: list[TaskResult], output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "results.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    cats = sorted({r.category for r in results})
    md = ["# Benchmark Results\n"]
    md.append(f"**Tasks:** {len(results)}  ")
    md.append(f"**Avg Score:** {sum(r.score for r in results)/len(results):.1%}  ")
    md.append(f"**Success Rate:** {sum(r.success for r in results)/len(results):.1%}  ")
    md.append(f"**Avg Steps:** {sum(r.iterations for r in results)/len(results):.1f}\n")
    md.append("\n## By Category\n")
    for cat in cats:
        cr = [r for r in results if r.category == cat]
        md.append(f"### {cat.replace('_',' ').title()}")
        md.append(f"- Score: {sum(r.score for r in cr)/len(cr):.1%}  "
                  f"Avg Steps: {sum(r.iterations for r in cr)/len(cr):.1f}\n")
    md.append("\n## All Results\n")
    md.append("| ID | Score | Steps | Time | OK |")
    md.append("|---|---|---|---|---|")
    for r in results:
        md.append(f"| {r.id} | {r.score:.0%} | {r.iterations} | {r.elapsed_s}s | {'✅' if r.success else '❌'} |")

    with open(out / "report.md", "w") as f:
        f.write("\n".join(md))

    logger.info(f"Results written to {out}/")


def _summary(results: list[TaskResult]) -> None:
    n = len(results)
    wins = sum(r.success for r in results)
    avg_sc = sum(r.score for r in results) / n
    avg_it = sum(r.iterations for r in results) / n
    avg_t = sum(r.elapsed_s for r in results) / n
    print("\n" + "=" * 55)
    print(" BENCHMARK SUMMARY")
    print("=" * 55)
    print(f"  Tasks:        {n}")
    print(f"  Success:      {wins}/{n}  ({wins/n:.1%})")
    print(f"  Avg Score:    {avg_sc:.1%}")
    print(f"  Avg Steps:    {avg_it:.1f}")
    print(f"  Avg Time:     {avg_t:.2f}s")
    print("=" * 55)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="evaluation/benchmark_tasks.json")
    ap.add_argument("--output", default="evaluation/results")
    args = ap.parse_args()
    run_evaluation(args.tasks, args.output)
