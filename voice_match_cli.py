#!/usr/bin/env python3
"""
CLI tool: analyze voice and find closest male/female speakers from DB.
"""

import argparse
import json
import sys

from voiceanalyzer.matching import VoiceMatchService, format_output_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze audio, compute x-vector and show proximity to speakers "
            "tagged male/female in database."
        )
    )
    parser.add_argument("input_file", help="Path to audio file")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output instead of text summary",
    )
    args = parser.parse_args()

    service = VoiceMatchService()

    try:
        result = service.process_file(args.input_file)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(format_output_text(result))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
