#!/usr/bin/env python3
"""
Voice Analysis Batch Processor
Recursively processes audio files, analyzes them, and stores results in database.
"""

import argparse
import sys
import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import shutil
import tempfile
from datetime import datetime
import traceback

# Progress bar
from tqdm import tqdm

# Import our components
try:
    from voice_analyzer import VoiceAnalyzer, VoiceAnalysisEncoder
    from db import VoiceDatabase
    from ml_funcs import wav_to_embedding
    from metadata_file import MetadataFile, MetadataEntry, validate_metadata_entries
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Ensure voice_analyzer.py, db.py, ml_funcs.py, and metadata_file.py are in the same directory.")
    sys.exit(1)

import numpy as np
import librosa
import soundfile as sf


class AudioFileProcessor:
    """Handles processing of audio files with voice analysis and database storage"""
    
    # Supported audio formats
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.opus'}
    MIN_DURATION_SEC = 4.0
    MAX_DURATION_SEC = 10.0
    
    def __init__(self,
                 db_config: Dict[str, Any],
                 verbose: bool = False,
                 skip_existing: bool = True,
                 include_frames: bool = False):
        """
        Initialize the audio file processor.
        
        Args:
            db_config: Database configuration dictionary
            verbose: Enable verbose output
            skip_existing: Skip files that are already in database (by hash)
            include_frames: Include frame-by-frame analysis
        """
        self.db = VoiceDatabase(**db_config)
        self.analyzer = VoiceAnalyzer()
        self.verbose = verbose
        self.skip_existing = skip_existing
        self.include_frames = include_frames
        
        # Statistics
        self.stats = {
            'total': 0,
            'processed': 0,
            'skipped': 0,
            'errors': 0,
            'already_exists': 0
        }

    def _suppress_noise_basic(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply basic spectral subtraction noise suppression."""
        if audio.size == 0:
            return audio

        n_fft = 1024
        hop_length = 256
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude, phase = np.abs(stft), np.angle(stft)

        # Estimate stationary noise floor from low-energy percentile per frequency bin
        noise_profile = np.percentile(magnitude, 10, axis=1, keepdims=True)
        suppression_factor = 1.5
        denoised_mag = np.maximum(magnitude - suppression_factor * noise_profile, 0.0)

        denoised_stft = denoised_mag * np.exp(1j * phase)
        denoised = librosa.istft(denoised_stft, hop_length=hop_length, length=len(audio))
        return denoised.astype(np.float32)

    def _trim_silence(self, audio: np.ndarray) -> np.ndarray:
        """Trim leading and trailing silence."""
        if audio.size == 0:
            return audio
        trimmed, _ = librosa.effects.trim(audio, top_db=30)
        return trimmed.astype(np.float32)

    def _split_fragments(self, audio: np.ndarray, sr: int) -> List[np.ndarray]:
        """Split long audio into fragments in [MIN_DURATION_SEC, MAX_DURATION_SEC]."""
        duration = len(audio) / sr if sr > 0 else 0.0
        if duration < self.MIN_DURATION_SEC:
            return []
        if duration <= self.MAX_DURATION_SEC:
            return [audio.astype(np.float32)]

        n_parts = int(np.ceil(duration / self.MAX_DURATION_SEC))
        boundaries = np.linspace(0, len(audio), n_parts + 1, dtype=int)

        fragments: List[np.ndarray] = []
        for i in range(n_parts):
            frag = audio[boundaries[i]:boundaries[i + 1]]
            frag_duration = len(frag) / sr if sr > 0 else 0.0
            if frag_duration >= self.MIN_DURATION_SEC:
                fragments.append(frag.astype(np.float32))

        return fragments

    def _prepare_fragments(self, filepath: Path) -> tuple[List[np.ndarray], int]:
        """Load file, suppress noise, trim silence, and split if needed."""
        audio, sr = librosa.load(str(filepath), sr=self.analyzer.sample_rate, mono=True)
        denoised = self._suppress_noise_basic(audio, sr)
        trimmed = self._trim_silence(denoised)
        fragments = self._split_fragments(trimmed, sr)
        return fragments, sr
    
    def find_audio_files(self, input_dir: Path) -> List[Path]:
        """
        Recursively find all audio files in directory.
        
        Args:
            input_dir: Input directory path
            
        Returns:
            List of audio file paths
        """
        audio_files = []
        for ext in self.SUPPORTED_FORMATS:
            audio_files.extend(input_dir.rglob(f'*{ext}'))
        
        return sorted(audio_files)
    
    def process_file(self,
                    filepath: Path,
                    author: Optional[str] = None,
                    author_source: Optional[str] = None,
                    tags: Optional[List[str]] = None,
                    reliable_quality_rating: Optional[float] = None,
                    unreliable_quality_rating: Optional[float] = None) -> Optional[int]:
        """
        Process a single audio file.
        
        Args:
            filepath: Path to audio file
            author: Speaker name/identifier
            author_source: Source of recording
            tags: List of tags
            reliable_quality_rating: Reliable quality rating
            unreliable_quality_rating: Unreliable quality rating
            
        Returns:
            First database record ID if at least one fragment is successful, None otherwise
        """
        temp_files: List[str] = []
        try:
            fragments, sr = self._prepare_fragments(filepath)

            if not fragments:
                if self.verbose:
                    print(f"  ⏭  Skipped: trimmed audio shorter than {self.MIN_DURATION_SEC:.1f}s")
                self.stats['skipped'] += 1
                return None

            first_record_id: Optional[int] = None
            for idx, fragment in enumerate(fragments, start=1):
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp_path = tmp.name
                temp_files.append(tmp_path)
                sf.write(tmp_path, fragment, sr)

                # Calculate file hash for this fragment
                file_hash = self.db.calculate_file_hash(tmp_path)

                # Check if already exists
                if self.skip_existing:
                    existing = self.db.get_recording_by_hash(file_hash)
                    if existing:
                        if self.verbose:
                            print(f"  ⏭  Fragment {idx}/{len(fragments)} already in database (ID: {existing['id']})")
                        self.stats['already_exists'] += 1
                        if first_record_id is None:
                            first_record_id = existing['id']
                        continue

                # Perform voice analysis on fragment
                if self.verbose:
                    print(f"  🔍 Analyzing fragment {idx}/{len(fragments)}...")

                result = self.analyzer.analyze(
                    tmp_path,
                    include_frames=self.include_frames
                )

                # Convert to JSON-serializable dict
                analysis_json = json.loads(
                    json.dumps(result, cls=VoiceAnalysisEncoder)
                )
                analysis_json['filename'] = f"{filepath.name}#part{idx}"
                analysis_json['source_file'] = str(filepath)
                analysis_json['fragment_index'] = idx
                analysis_json['fragment_count'] = len(fragments)

                # Extract x-vector using ml_funcs
                if self.verbose:
                    print(f"  🧠 Extracting x-vector embedding for fragment {idx}/{len(fragments)}...")

                try:
                    wav_16k = librosa.resample(fragment, orig_sr=sr, target_sr=16000)
                    x_vector = wav_to_embedding(wav_16k, 16000)
                except Exception as e:
                    if self.verbose:
                        print(f"  ⚠️  Warning: Could not extract x-vector: {e}")
                    x_vector = None

                # Insert into database
                if self.verbose:
                    print(f"  💾 Storing fragment {idx}/{len(fragments)} in database...")

                record_id = self.db.insert_voice_recording(
                    analysis_data=analysis_json,
                    file_hash=file_hash,
                    duration=result.duration,
                    author=author,
                    author_source=author_source,
                    tags=tags or [],
                    reliable_quality_rating=reliable_quality_rating,
                    unreliable_quality_rating=unreliable_quality_rating,
                    x_vector=x_vector
                )

                self.stats['processed'] += 1
                if first_record_id is None:
                    first_record_id = record_id

                if self.verbose:
                    print(f"  ✅ Fragment stored (ID: {record_id}, duration: {result.duration:.2f}s)")

            if first_record_id is not None:
                return first_record_id

            self.stats['skipped'] += 1
            return None
            
        except Exception as e:
            self.stats['errors'] += 1
            print(f"  ❌ Error processing file: {e}")
            if self.verbose:
                traceback.print_exc()
            return None
        finally:
            for tmp_path in temp_files:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
    
    def process_directory(self,
                         input_dir: Path,
                         default_author: Optional[str] = None,
                         default_source: Optional[str] = None,
                         default_tags: Optional[List[str]] = None,
                         default_reliable_quality: Optional[float] = None,
                         default_unreliable_quality: Optional[float] = None,
                         move_processed_to: Optional[Path] = None) -> Dict[str, Any]:
        """
        Process all audio files in directory.
        
        Args:
            input_dir: Input directory
            default_author: Default author for all files
            default_source: Default source for all files
            default_tags: Default tags for all files
            default_reliable_quality: Default reliable quality rating
            default_unreliable_quality: Default unreliable quality rating
            move_processed_to: Optional directory to move processed files
            
        Returns:
            Statistics dictionary
        """
        # Find all audio files
        audio_files = self.find_audio_files(input_dir)
        self.stats['total'] = len(audio_files)
        
        if self.stats['total'] == 0:
            print(f"No audio files found in {input_dir}")
            return self.stats
        
        print(f"Found {self.stats['total']} audio files")
        
        # Process each file
        with tqdm(total=self.stats['total'], desc="Processing files") as pbar:
            for filepath in audio_files:
                pbar.set_description(f"Processing {filepath.name}")
                
                if self.verbose:
                    print(f"\n📁 {filepath}")
                
                record_id = self.process_file(
                    filepath=filepath,
                    author=default_author,
                    author_source=default_source,
                    tags=default_tags,
                    reliable_quality_rating=default_reliable_quality,
                    unreliable_quality_rating=default_unreliable_quality
                )
                
                # Move file if requested and processing was successful
                if move_processed_to and record_id is not None:
                    self._move_file(filepath, input_dir, move_processed_to)
                
                pbar.update(1)
        
        return self.stats
    
    def process_metadata_file(self,
                             metadata_file: Path,
                             default_author: Optional[str] = None,
                             default_source: Optional[str] = None,
                             default_tags: Optional[List[str]] = None,
                             default_reliable_quality: Optional[float] = None,
                             default_unreliable_quality: Optional[float] = None,
                             move_processed_to: Optional[Path] = None) -> Dict[str, Any]:
        """
        Process files listed in metadata file.
        
        Args:
            metadata_file: Path to metadata file
            default_author: Default author (file values take precedence)
            default_source: Default source (file values take precedence)
            default_tags: Default tags (file values take precedence)
            default_reliable_quality: Default reliable quality
            default_unreliable_quality: Default unreliable quality
            move_processed_to: Optional directory to move processed files
            
        Returns:
            Statistics dictionary
        """
        # Read metadata file
        print(f"Reading metadata file: {metadata_file}")
        mf = MetadataFile(str(metadata_file))
        entries = mf.read()
        
        # Validate entries
        errors = validate_metadata_entries(entries)
        if errors:
            print("\n⚠️  Validation errors found:")
            for error in errors:
                print(f"  - {error}")
            
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return self.stats
        
        # Merge with defaults
        merged_entries = mf.merge_with_defaults(
            default_author=default_author,
            default_source=default_source,
            default_tags=default_tags,
            default_reliable_quality=default_reliable_quality,
            default_unreliable_quality=default_unreliable_quality
        )
        
        self.stats['total'] = len(merged_entries)
        print(f"Processing {self.stats['total']} files from metadata")
        
        # Get base directory for relative paths
        metadata_base_dir = metadata_file.parent
        
        # Process each entry
        with tqdm(total=self.stats['total'], desc="Processing files") as pbar:
            for entry in merged_entries:
                # Resolve filepath (handle both absolute and relative paths)
                filepath = Path(entry.filepath)
                if not filepath.is_absolute():
                    filepath = metadata_base_dir / filepath
                
                pbar.set_description(f"Processing {filepath.name}")
                
                if self.verbose:
                    print(f"\n📁 {filepath}")
                
                if not filepath.exists():
                    print(f"  ⚠️  File not found: {filepath}")
                    self.stats['skipped'] += 1
                    pbar.update(1)
                    continue
                
                record_id = self.process_file(
                    filepath=filepath,
                    author=entry.author,
                    author_source=entry.author_source,
                    tags=entry.tags,
                    reliable_quality_rating=entry.reliable_quality_rating,
                    unreliable_quality_rating=entry.unreliable_quality_rating
                )
                
                # Move file if requested and processing was successful
                if move_processed_to and record_id is not None:
                    # Determine original base for preserving hierarchy
                    original_base = metadata_base_dir
                    self._move_file(filepath, original_base, move_processed_to)
                
                pbar.update(1)
        
        return self.stats
    
    def _move_file(self, filepath: Path, original_base: Path, target_base: Path):
        """
        Move processed file to target directory, preserving directory hierarchy.
        
        Args:
            filepath: File to move
            original_base: Original base directory
            target_base: Target base directory
        """
        try:
            # Calculate relative path
            try:
                rel_path = filepath.relative_to(original_base)
            except ValueError:
                # If not relative, just use filename
                rel_path = filepath.name
            
            # Create target path
            target_path = target_base / rel_path
            
            # Create target directory
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move file
            shutil.move(str(filepath), str(target_path))
            
            if self.verbose:
                print(f"  📦 Moved to: {target_path}")
                
        except Exception as e:
            print(f"  ⚠️  Warning: Could not move file: {e}")
    
    def print_summary(self):
        """Print processing summary"""
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"Total files:        {self.stats['total']}")
        print(f"Processed:          {self.stats['processed']}")
        print(f"Already existed:    {self.stats['already_exists']}")
        print(f"Skipped:            {self.stats['skipped']}")
        print(f"Errors:             {self.stats['errors']}")
        print("="*60)


def main():
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
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument('--input', type=str,
                           help='Input directory containing audio files (recursive)')
    input_group.add_argument('--input-meta', type=str,
                           help='Metadata file (JSON, JSONL, or CSV) with file paths and metadata')
    
    # Metadata options
    parser.add_argument('--author', type=str,
                       help='Default author/speaker name for all files')
    parser.add_argument('--author-source', type=str,
                       help='Default source of recordings')
    parser.add_argument('--tags', type=str,
                       help='Default comma-separated tags')
    parser.add_argument('--reliable-quality', type=float,
                       help='Default reliable quality rating (0-1)')
    parser.add_argument('--unreliable-quality', type=float,
                       help='Default unreliable quality rating (0-1)')
    
    # Processing options
    parser.add_argument('--move-processed-to', type=str,
                       help='Move processed files to this directory (preserves hierarchy)')
    parser.add_argument('--include-frames', action='store_true',
                       help='Include frame-by-frame analysis (increases storage)')
    parser.add_argument('--force-reprocess', action='store_true',
                       help='Process files even if already in database')
    
    # Database options
    parser.add_argument('--db-host', type=str, default='localhost',
                       help='Database host (default: localhost)')
    parser.add_argument('--db-port', type=int, default=5432,
                       help='Database port (default: 5432)')
    parser.add_argument('--db-name', type=str, default='voice_analysis',
                       help='Database name (default: voice_analysis)')
    parser.add_argument('--db-user', type=str, default='postgres',
                       help='Database user (default: postgres)')
    parser.add_argument('--db-password', type=str, default='',
                       help='Database password')
    
    # Utility options
    parser.add_argument('--init-db', action='store_true',
                       help='Initialize database schema and exit')
    parser.add_argument('--create-template', type=str,
                       help='Create metadata template file and exit (specify path)')
    parser.add_argument('--template-format', type=str, default='json',
                       choices=['json', 'jsonl', 'csv'],
                       help='Template format (default: json)')
    
    # Output options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()
    
    # Database configuration
    db_config = {
        'host': args.db_host,
        'port': args.db_port,
        'database': args.db_name,
        'user': args.db_user,
        'password': args.db_password
    }
    
    # Handle utility commands
    if args.init_db:
        print("Initializing database...")
        db = VoiceDatabase(**db_config)
        db.initialize_database()
        print("✅ Database initialized successfully!")
        return
    
    if args.create_template:
        print(f"Creating template metadata file: {args.create_template}")
        MetadataFile.create_template(
            args.create_template,
            format=args.template_format
        )
        print("✅ Template created! Edit it to add your file metadata.")
        return
    
    # Require either --input or --input-meta
    if not args.input and not args.input_meta:
        parser.error("Either --input or --input-meta is required (or use --init-db / --create-template)")
    
    # Parse tags
    default_tags = None
    if args.tags:
        default_tags = [t.strip() for t in args.tags.split(',')]
    
    # Create processor
    processor = AudioFileProcessor(
        db_config=db_config,
        verbose=args.verbose and not args.quiet,
        skip_existing=not args.force_reprocess,
        include_frames=args.include_frames
    )
    
    # Process files
    start_time = datetime.now()
    
    try:
        if args.input:
            # Directory mode
            input_dir = Path(args.input)
            if not input_dir.exists():
                print(f"Error: Input directory does not exist: {input_dir}")
                sys.exit(1)
            
            move_to = Path(args.move_processed_to) if args.move_processed_to else None
            
            stats = processor.process_directory(
                input_dir=input_dir,
                default_author=args.author,
                default_source=args.author_source,
                default_tags=default_tags,
                default_reliable_quality=args.reliable_quality,
                default_unreliable_quality=args.unreliable_quality,
                move_processed_to=move_to
            )
            
        else:
            # Metadata file mode
            metadata_file = Path(args.input_meta)
            if not metadata_file.exists():
                print(f"Error: Metadata file does not exist: {metadata_file}")
                sys.exit(1)
            
            move_to = Path(args.move_processed_to) if args.move_processed_to else None
            
            stats = processor.process_metadata_file(
                metadata_file=metadata_file,
                default_author=args.author,
                default_source=args.author_source,
                default_tags=default_tags,
                default_reliable_quality=args.reliable_quality,
                default_unreliable_quality=args.unreliable_quality,
                move_processed_to=move_to
            )
        
        # Print summary
        if not args.quiet:
            processor.print_summary()
            
            elapsed = datetime.now() - start_time
            print(f"\nTotal time: {elapsed}")
            
            if stats['processed'] > 0:
                avg_time = elapsed.total_seconds() / stats['processed']
                print(f"Average time per file: {avg_time:.2f}s")
    
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
