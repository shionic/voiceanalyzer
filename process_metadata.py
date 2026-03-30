#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import List

from voiceanalyzer.metadata import (
    MetadataFile,
    process_mozilla_common_voice,
    process_voxceleb2,
)


def main():
    parser = argparse.ArgumentParser(
        description="Convert dataset metadata into internal metadata format"
    )
    parser.add_argument(
        "process_type",
        choices=["mozilla_common_voice", "voxceleb2"],
        help="Type of dataset to process",
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing dataset files (e.g., train.tsv)",
    )
    parser.add_argument(
        "output_metadata",
        type=Path,
        help="Output metadata file (.json, .jsonl, or .csv)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "jsonl", "csv"],
        default="json",
        help="Output metadata format (default: json)",
    )

    args = parser.parse_args()

    if args.process_type == "mozilla_common_voice":
        entries = process_mozilla_common_voice(args.input_dir)
    elif args.process_type == "voxceleb2":
        entries = process_voxceleb2(args.input_dir)
    else:
        raise ValueError(f"Unsupported process type: {args.process_type}")

    mf = MetadataFile(str(args.output_metadata))
    mf.write(entries, format=args.format)

    print(f"Wrote {len(entries)} metadata entries to {args.output_metadata}")


if __name__ == "__main__":
    main()
