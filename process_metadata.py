#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path
from typing import List

from metadata_file import MetadataEntry, MetadataFile


AUTHOR_SOURCE_MCV = "MozillaCommonVoices"


def normalize_gender_tag(gender_value: str) -> List[str]:
    if not gender_value:
        return []
    g = gender_value.strip().lower()
    if "female" in g or "feminine" in g:
        return ["female"]
    if "male" in g or "masculine" in g:
        return ["male"]
    return []


def normalize_age_tag(age_value: str) -> List[str]:
    if not age_value:
        return []
    return [age_value.strip().lower()]


def compute_unreliable_rating(up_votes: str, down_votes: str) -> float:
    try:
        up = float(up_votes) if up_votes else 0.0
        down = float(down_votes) if down_votes else 0.0
    except ValueError:
        return None

    total = up + down
    if total == 0:
        return None

    return up / total


def process_mozilla_common_voice(input_dir: Path) -> List[MetadataEntry]:
    tsv_path = input_dir / "train.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(f"train.tsv not found in {input_dir}")

    entries: List[MetadataEntry] = []

    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            rel_path = row.get("path")
            if not rel_path:
                continue

            tags = []
            tags.extend(normalize_gender_tag(row.get("gender")))
            tags.extend(normalize_age_tag(row.get("age")))

            unreliable_rating = compute_unreliable_rating(
                row.get("up_votes"),
                row.get("down_votes"),
            )

            entry = MetadataEntry(
                filepath=str(Path("clips/"+rel_path)),  # relative to input dir (as required)
                author=row.get("client_id"),
                author_source=AUTHOR_SOURCE_MCV,
                tags=tags,
                reliable_quality_rating=None,
                unreliable_quality_rating=unreliable_rating,
            )

            entries.append(entry)

    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Convert dataset metadata into internal metadata format"
    )
    parser.add_argument(
        "process_type",
        choices=["mozilla_common_voice"],
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
    else:
        raise ValueError(f"Unsupported process type: {args.process_type}")

    mf = MetadataFile(str(args.output_metadata))
    mf.write(entries, format=args.format)

    print(f"Wrote {len(entries)} metadata entries to {args.output_metadata}")


if __name__ == "__main__":
    main()
