#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Iterator, List

from voiceanalyzer.metadata import (
    MetadataFile,
    process_mozilla_common_voice,
    process_voxceleb2,
)
from voiceanalyzer.metadata.metadata_file import MetadataEntry



def _chunked(items: List[MetadataEntry], chunk_size: int) -> Iterator[List[MetadataEntry]]:
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def _build_split_output_path(base_path: Path, part_index: int) -> Path:
    return base_path.with_name(f"{base_path.stem}.part{part_index:04d}{base_path.suffix}")


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
    parser.add_argument(
        "--split-size",
        type=int,
        default=0,
        help=(
            "Maximum number of metadata entries per output file. "
            "When > 0, writes multiple files like output.part0001.json"
        ),
    )

    args = parser.parse_args()

    if args.process_type == "mozilla_common_voice":
        entries = process_mozilla_common_voice(args.input_dir)
    elif args.process_type == "voxceleb2":
        entries = process_voxceleb2(args.input_dir)
    else:
        raise ValueError(f"Unsupported process type: {args.process_type}")

    if args.split_size < 0:
        parser.error("--split-size must be >= 0")

    if args.split_size > 0:
        total_files = 0
        for chunk_index, chunk in enumerate(_chunked(entries, args.split_size), start=1):
            output_part = _build_split_output_path(args.output_metadata, chunk_index)
            mf = MetadataFile(str(output_part))
            mf.write(chunk, format=args.format)
            total_files += 1
        print(
            f"Wrote {len(entries)} metadata entries into {total_files} files "
            f"(split size: {args.split_size})"
        )
        return

    mf = MetadataFile(str(args.output_metadata))
    mf.write(entries, format=args.format)

    print(f"Wrote {len(entries)} metadata entries to {args.output_metadata}")


if __name__ == "__main__":
    main()
