#!/usr/bin/env python3
"""CLI entrypoint for batch voice analysis processing."""

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

from voiceanalyzer.batch import AudioFileProcessor
from voiceanalyzer.storage import VoiceDatabase
from voiceanalyzer.metadata import MetadataFile


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch Voice Analysis Processor - Process audio files and store in database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files in a directory
  python process_batch.py --input /path/to/audio --author "John Doe" --tags interview,english

  # Process from metadata file
  python process_batch.py --input-meta metadata.json

  # Move processed files
  python process_batch.py --input /audio --move-processed-to /processed

  # Initialize database first
  python process_batch.py --init-db
        """,
    )

    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument("--input", type=str, help="Input directory containing audio files (recursive)")
    input_group.add_argument("--input-meta", type=str, help="Metadata file (JSON, JSONL, or CSV)")

    parser.add_argument("--author", type=str, help="Default author/speaker name for all files")
    parser.add_argument("--author-source", type=str, help="Default source of recordings")
    parser.add_argument("--tags", type=str, help="Default comma-separated tags")
    parser.add_argument("--reliable-quality", type=float, help="Default reliable quality rating (0-1)")
    parser.add_argument("--unreliable-quality", type=float, help="Default unreliable quality rating (0-1)")

    parser.add_argument("--move-processed-to", type=str, help="Move processed files to this directory")
    parser.add_argument("--include-frames", action="store_true", help="Include frame-by-frame analysis")
    parser.add_argument(
        "--split-long-audio",
        action="store_true",
        help="Split long files into fragments (disabled by default)",
    )
    parser.add_argument("--force-reprocess", action="store_true", help="Process files even if already in DB")

    parser.add_argument("--db-host", type=str, default="localhost")
    parser.add_argument("--db-port", type=int, default=5432)
    parser.add_argument("--db-name", type=str, default="voice_analysis")
    parser.add_argument("--db-user", type=str, default="postgres")
    parser.add_argument("--db-password", type=str, default="")

    parser.add_argument("--init-db", action="store_true", help="Initialize database schema and exit")
    parser.add_argument("--create-template", type=str, help="Create metadata template file and exit")
    parser.add_argument("--template-format", type=str, default="json", choices=["json", "jsonl", "csv"])

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")

    args = parser.parse_args()

    db_config = {
        "host": args.db_host,
        "port": args.db_port,
        "database": args.db_name,
        "user": args.db_user,
        "password": args.db_password,
    }

    if args.init_db:
        print("Initializing database...")
        db = VoiceDatabase(**db_config)
        db.initialize_database()
        print("✅ Database initialized successfully!")
        return

    if args.create_template:
        print(f"Creating template metadata file: {args.create_template}")
        MetadataFile.create_template(args.create_template, format=args.template_format)
        print("✅ Template created! Edit it to add your file metadata.")
        return

    if not args.input and not args.input_meta:
        parser.error("Either --input or --input-meta is required (or use --init-db / --create-template)")

    default_tags = [t.strip() for t in args.tags.split(",")] if args.tags else None

    processor = AudioFileProcessor(
        db_config=db_config,
        verbose=args.verbose and not args.quiet,
        skip_existing=not args.force_reprocess,
        include_frames=args.include_frames,
        split_long_audio=args.split_long_audio,
    )

    start_time = datetime.now()
    try:
        move_to = Path(args.move_processed_to) if args.move_processed_to else None
        if args.input:
            input_dir = Path(args.input)
            if not input_dir.exists():
                print(f"Error: Input directory does not exist: {input_dir}")
                sys.exit(1)
            stats = processor.process_directory(
                input_dir=input_dir,
                default_author=args.author,
                default_source=args.author_source,
                default_tags=default_tags,
                default_reliable_quality=args.reliable_quality,
                default_unreliable_quality=args.unreliable_quality,
                move_processed_to=move_to,
            )
        else:
            metadata_file = Path(args.input_meta)
            if not metadata_file.exists():
                print(f"Error: Metadata file does not exist: {metadata_file}")
                sys.exit(1)
            stats = processor.process_metadata_file(
                metadata_file=metadata_file,
                default_author=args.author,
                default_source=args.author_source,
                default_tags=default_tags,
                default_reliable_quality=args.reliable_quality,
                default_unreliable_quality=args.unreliable_quality,
                move_processed_to=move_to,
            )

        if not args.quiet:
            processor.print_summary()
            elapsed = datetime.now() - start_time
            print(f"\nTotal time: {elapsed}")
            if stats["processed"] > 0:
                print(f"Average time per file: {elapsed.total_seconds() / stats['processed']:.2f}s")

    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")
        processor.print_summary()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
