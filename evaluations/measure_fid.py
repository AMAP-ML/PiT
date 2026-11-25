"""
Measure FID between two image distributions.

Each argument can be a folder containing images or a precomputed .npz
statistics file (with 'mu' and 'sigma' arrays).

Usage:
    python evaluations/measure_fid.py <path1> <path2> [--batch_size 50] [--dims 2048] [--device cuda]

Example:
    python evaluations/measure_fid.py /path/to/generated_images /path/to/fid_stats.npz
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluations.fid_score import calculate_fid_given_paths


def main():
    parser = argparse.ArgumentParser(description="Measure FID between two paths (image folder or .npz stats).")
    parser.add_argument("path1", type=str, help="Path to image folder or .npz statistics file")
    parser.add_argument("path2", type=str, help="Path to image folder or .npz statistics file")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for Inception feature extraction")
    parser.add_argument("--dims", type=int, default=2048, help="Inception feature dimensionality")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (default: cuda if available else cpu)")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers")
    args = parser.parse_args()

    fid = calculate_fid_given_paths(
        (args.path1, args.path2),
        device=args.device,
        batch_size=args.batch_size,
        dims=args.dims,
        num_workers=args.num_workers,
    )
    print(f"FID: {fid}")


if __name__ == "__main__":
    main()
