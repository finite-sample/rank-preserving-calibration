#!/usr/bin/env python3
"""Run all experiments and generate LaTeX outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from real_data_experiment import run_real_data
from synthetic_experiment import run_synthetic


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run rank-preserving calibration experiments"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for results (default: output)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="Number of seeds for real-data experiment (default: 5)",
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Run only the synthetic experiment",
    )
    parser.add_argument(
        "--real-only",
        action="store_true",
        help="Run only the real-data experiment",
    )
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not args.real_only:
        print("=" * 60)
        print("Running Synthetic 'No Free Lunch' Experiment")
        print("=" * 60)
        run_synthetic(output_dir=args.output_dir, seed=args.seed)
        print()

    if not args.synthetic_only:
        print("=" * 60)
        print("Running Real-Data Shifted Deployment Experiment")
        print("=" * 60)
        run_real_data(output_dir=args.output_dir, n_seeds=args.n_seeds, seed=args.seed)
        print()

    print("=" * 60)
    print(f"All experiments complete. Outputs in {output_path.absolute()}")
    print("=" * 60)
    print()
    print("Generated files:")
    for f in sorted(output_path.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
