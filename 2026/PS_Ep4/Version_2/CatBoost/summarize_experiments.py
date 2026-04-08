"""Summarize model performance across all CatBoost experiments.

Parses the per-experiment log files in logs/ and produces:
  - A formatted table to stdout (and saved to logs/performance_summary.txt)
  - Per-fold breakdowns, overall OOF balanced accuracy, std, min, max
  - A ranked leaderboard sorted by overall balanced accuracy
"""

import os
import re
import glob

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(LOG_DIR, "performance_summary.txt")

# Experiment descriptions (pulled from docstrings)
DESCRIPTIONS = {
    1:  "Baseline (decimal digits + binary flags)",
    2:  "Interaction features",
    3:  "Ratio features",
    4:  "Log transforms + polynomial features",
    5:  "Frequency encoding of categoricals",
    6:  "Deeper trees (depth=6) + lower LR",
    7:  "Second decimal digit + quantile binning",
    8:  "KMeans clustering features",
    9:  "All combined features + LR stacking",
    10: "Lossguide grow policy + all features",
    11: "10-fold CV + best features + tuned HP",
}


def parse_log(log_path):
    """Extract fold scores and overall score from a log file."""
    fold_scores = []
    overall = None
    with open(log_path, "r") as f:
        for line in f:
            m = re.search(
                r"Fold\s+(\d+)\s+Balanced Accuracy:\s+([\d.]+)", line
            )
            if m:
                fold_scores.append(float(m.group(2)))
            m2 = re.search(
                r"Overall OOF Balanced Accuracy:\s+([\d.]+)", line
            )
            if m2:
                overall = float(m2.group(1))
    return fold_scores, overall


def main():
    log_files = sorted(
        glob.glob(os.path.join(LOG_DIR, "cat_*.log")),
        key=lambda p: int(re.search(r"cat_(\d+)\.log", p).group(1)),
    )

    if not log_files:
        print("No log files found in", LOG_DIR)
        return

    results = []
    for lf in log_files:
        num = int(re.search(r"cat_(\d+)\.log", lf).group(1))
        fold_scores, overall = parse_log(lf)
        if overall is None and fold_scores:
            import numpy as np
            overall = float(np.mean(fold_scores))
        results.append({
            "num": num,
            "desc": DESCRIPTIONS.get(num, ""),
            "folds": fold_scores,
            "overall": overall,
            "n_folds": len(fold_scores),
        })

    # Sort by overall score descending for the leaderboard
    ranked = sorted(results, key=lambda r: r["overall"] or 0, reverse=True)

    lines = []

    # ---- Header ---------------------------------------------------
    lines.append("=" * 90)
    lines.append("  CatBoost Experiments -- Performance Summary")
    lines.append("=" * 90)
    lines.append("")

    # ---- Leaderboard table ----------------------------------------
    lines.append(
        f"{'Rank':<5} {'Script':<11} {'OOF BA':>10} {'Std':>10} "
        f"{'Min Fold':>10} {'Max Fold':>10} {'Folds':>6}  Description"
    )
    lines.append("-" * 90)

    for rank, r in enumerate(ranked, 1):
        if r["overall"] is None:
            lines.append(
                f"{rank:<5} cat_{r['num']:<6}  {'N/A':>10} {'':>10} "
                f"{'':>10} {'':>10} {r['n_folds']:>6}  {r['desc']}"
            )
            continue

        folds = r["folds"]
        if folds:
            import numpy as np
            std = float(np.std(folds))
            mn = min(folds)
            mx = max(folds)
        else:
            std = mn = mx = 0.0

        lines.append(
            f"{rank:<5} cat_{r['num']:<6} {r['overall']:>10.6f} "
            f"{std:>10.6f} {mn:>10.6f} {mx:>10.6f} {r['n_folds']:>6}  "
            f"{r['desc']}"
        )

    lines.append("-" * 90)
    lines.append("")

    # ---- Per-experiment fold details --------------------------------
    lines.append("=" * 90)
    lines.append("  Per-Fold Breakdown")
    lines.append("=" * 90)

    for r in results:
        lines.append("")
        lines.append(
            f"  cat_{r['num']}.py -- {r['desc']}"
        )
        if not r["folds"]:
            lines.append("    (no fold results found)")
            continue
        for i, s in enumerate(r["folds"], 1):
            lines.append(f"    Fold {i:>2}: {s:.6f}")
        if r["overall"] is not None:
            lines.append(f"    {'Overall':>8}: {r['overall']:.6f}")

    lines.append("")
    lines.append("=" * 90)

    # ---- Best experiment highlight ----------------------------------
    best = ranked[0]
    lines.append("")
    lines.append(
        f"  BEST: cat_{best['num']}.py  "
        f"(OOF Balanced Accuracy = {best['overall']:.6f})"
    )
    lines.append(f"        {best['desc']}")
    lines.append("")

    # Improvement over baseline
    baseline = next((r for r in results if r["num"] == 1), None)
    if baseline and baseline["overall"] and best["overall"]:
        delta = best["overall"] - baseline["overall"]
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"  Improvement over baseline (cat_1): "
            f"{sign}{delta:.6f} ({sign}{delta * 100:.4f}%)"
        )
        lines.append("")

    lines.append("=" * 90)

    report = "\n".join(lines)
    print(report)

    # Save to file
    with open(OUTPUT_FILE, "w") as f:
        f.write(report + "\n")
    print(f"\nSummary saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
