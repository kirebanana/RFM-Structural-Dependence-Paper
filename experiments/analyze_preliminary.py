"""Analyze a completed saved matrix without importing RT or loading weights."""

import argparse
import json
from pathlib import Path

from rfm_structure.analysis import analyze_matrix, summary_markdown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_matrix(args.run_directory)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.output / "SUMMARY.md").write_text(summary_markdown(result))
    print(args.output / "SUMMARY.md")


if __name__ == "__main__":
    main()
