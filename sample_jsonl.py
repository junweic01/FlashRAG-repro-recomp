#!/usr/bin/env python3
import argparse
import os
import random

def sample_jsonl(input_path, keep_ratio=0.1, seed=None):
    # Derive output path: same folder, with suffix ".sample10.jsonl"
    folder, filename = os.path.split(input_path)
    name, ext = os.path.splitext(filename)
    if not ext:  # if file has no extension, just append
        ext = ".jsonl"
    output_filename = f"{name}.sample{int(keep_ratio * 100)}{ext}"
    output_path = os.path.join(folder, output_filename)

    if seed is not None:
        random.seed(seed)

    total = 0
    kept = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            # Keep each line with probability = keep_ratio (≈10%)
            if random.random() < keep_ratio:
                fout.write(line)
                kept += 1

    print(f"Input file:  {input_path}")
    print(f"Output file: {output_path}")
    print(f"Total lines seen: {total}")
    print(f"Lines kept (~{keep_ratio*100:.1f}%): {kept}")

def main():
    parser = argparse.ArgumentParser(
        description="Randomly sample ~10% of a large JSONL file without loading it into memory."
    )
    parser.add_argument("--input_path", help="Path to the input .jsonl file (e.g. wiki18_100w.jsonl)")
    parser.add_argument(
        "--ratio", "-r",
        type=float,
        default=0.1,
        help="Fraction of lines to keep (default: 0.1 = 10%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)"
    )

    args = parser.parse_args()
    sample_jsonl(args.input_path, keep_ratio=args.ratio, seed=args.seed)

if __name__ == "__main__":
    main()
