# harness.py --- shared plumbing for every eval: load goldens, fold the
# raw deepeval result into per-metric numbers, print them the same way.
import json
import os
import sys

# Windows consoles default to cp1252, which blows up on the emoji in
# deepeval's own report. Force UTF-8 on the way out.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# Each eval needs different fields: the retriever compares against an ideal
# answer, the generator is fed an ideal context. Callers say which.
REQUIRED_GOLDEN_KEYS = ("query", "ideal_answer")


# 1. LOAD --- the golden set is the fixed input; fail loudly if it is malformed
def load_goldens(path, required=REQUIRED_GOLDEN_KEYS):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"golden set not found: {path} (run evals from the project root)"
        )

    with open(path, encoding="utf-8") as f:
        goldens = json.load(f)

    if not goldens:
        raise ValueError(f"golden set is empty: {path}")

    for i, g in enumerate(goldens):
        missing = [k for k in required if not g.get(k)]
        if missing:
            raise ValueError(
                f"golden {g.get('id', f'#{i}')} in {path} is missing: {', '.join(missing)}"
            )

    return goldens


# 2. FOLD --- one row per metric, averaged over every test case
def summarize_by_metric(result):
    summary = {}

    for test in result.test_results:
        for m in test.metrics_data or []:
            row = summary.setdefault(
                m.name,
                {"n": 0, "passed": 0, "errored": 0, "scores": [], "failures": []},
            )
            row["n"] += 1

            if m.error:
                row["errored"] += 1
                row["failures"].append({"input": test.input, "reason": m.error})
                continue

            if m.score is not None:
                row["scores"].append(m.score)

            if m.success:
                row["passed"] += 1
            else:
                row["failures"].append(
                    {"input": test.input, "score": m.score, "reason": m.reason}
                )

    # collapse the running lists into the numbers we actually report
    for row in summary.values():
        scores = row.pop("scores")
        row["mean_score"] = sum(scores) / len(scores) if scores else None
        row["pass_rate"] = row["passed"] / row["n"] if row["n"] else None

    return summary


# 3. PRINT --- a compact scoreboard plus the cases worth looking at
def print_summary(name, summary, show_failures=5):
    print(f"\n=== {name} ===")

    if not summary:
        print("no metrics recorded")
        return

    width = max(len(m) for m in summary)
    for metric, row in summary.items():
        mean = "n/a" if row["mean_score"] is None else f"{row['mean_score']:.3f}"
        print(
            f"{metric:<{width}}  mean {mean}  "
            f"pass {row['passed']}/{row['n']} ({row['pass_rate']:.0%})"
            + (f"  errors {row['errored']}" if row["errored"] else "")
        )

    for metric, row in summary.items():
        if not row["failures"]:
            continue
        print(f"\n-- {metric} failures ({len(row['failures'])}) --")
        for f in row["failures"][:show_failures]:
            score = "error" if f.get("score") is None else f"{f['score']:.2f}"
            print(f"  [{score}] {f['input']}")
            if f.get("reason"):
                print(f"          {f['reason']}")
        if len(row["failures"]) > show_failures:
            print(f"  ... {len(row['failures']) - show_failures} more")
