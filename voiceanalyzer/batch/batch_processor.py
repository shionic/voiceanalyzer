"""Batch processing services for audio file ingestion."""

from __future__ import annotations

import json
import hashlib
import shutil
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
from tqdm import tqdm

from voiceanalyzer.audio import preprocess_audio_basic, trim_silence
from voiceanalyzer.constants import SUPPORTED_AUDIO_FORMATS
from voiceanalyzer.storage import VoiceDatabase
from voiceanalyzer.metadata import MetadataFile, validate_metadata_entries
from voiceanalyzer.embeddings import configure_torch_threads, wav_to_embedding
from voiceanalyzer.analysis import VoiceAnalysisEncoder, VoiceAnalyzer


class AudioPreprocessor:
    """Prepare audio into valid fragments for analysis."""

    def __init__(self, min_duration_sec: float = 4.0, max_duration_sec: float = 10.0):
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec

    def suppress_noise_basic(self, audio: np.ndarray) -> np.ndarray:
        return preprocess_audio_basic(audio, silence_top_db=120)

    def trim_silence(self, audio: np.ndarray) -> np.ndarray:
        return trim_silence(audio, top_db=30)

    def split_fragments(self, audio: np.ndarray, sr: int) -> List[np.ndarray]:
        duration = len(audio) / sr if sr > 0 else 0.0
        if duration < self.min_duration_sec:
            return []
        if duration <= self.max_duration_sec:
            return [audio.astype(np.float32)]

        n_parts = int(np.ceil(duration / self.max_duration_sec))
        boundaries = np.linspace(0, len(audio), n_parts + 1, dtype=int)

        fragments: List[np.ndarray] = []
        for i in range(n_parts):
            frag = audio[boundaries[i]:boundaries[i + 1]]
            frag_duration = len(frag) / sr if sr > 0 else 0.0
            if frag_duration >= self.min_duration_sec:
                fragments.append(frag.astype(np.float32))
        return fragments

    def prepare_fragments(self, filepath: Path, target_sr: int) -> tuple[List[np.ndarray], int]:
        audio, sr = librosa.load(str(filepath), sr=target_sr, mono=True)
        trimmed = preprocess_audio_basic(audio, silence_top_db=30)
        fragments = self.split_fragments(trimmed, sr)
        return fragments, sr


class AudioFileProcessor:
    """Handles processing of audio files with voice analysis and database storage."""

    MIN_DURATION_SEC = 4.0
    MAX_DURATION_SEC = 10.0

    def __init__(
        self,
        db_config: Dict[str, Any],
        verbose: bool = False,
        skip_existing: bool = True,
        include_frames: bool = False,
        split_long_audio: bool = False,
        max_workers: int = 1,
        torch_intra_threads: Optional[int] = None,
        torch_inter_threads: Optional[int] = None,
    ):
        configure_torch_threads(
            intra_op_threads=torch_intra_threads,
            inter_op_threads=torch_inter_threads,
        )
        self.db = VoiceDatabase(**db_config)
        self.preprocessor = AudioPreprocessor(
            min_duration_sec=self.MIN_DURATION_SEC,
            max_duration_sec=self.MAX_DURATION_SEC,
        )
        self._thread_local = threading.local()
        self._sample_rate = VoiceAnalyzer().sample_rate
        self.max_workers = max(1, max_workers)
        self.verbose = verbose
        self.skip_existing = skip_existing
        self.include_frames = include_frames
        self.split_long_audio = split_long_audio
        self._stats_lock = threading.Lock()

        self.stats = {
            "total": 0,
            "processed": 0,
            "skipped": 0,
            "errors": 0,
            "already_exists": 0,
        }

    def _get_analyzer(self) -> VoiceAnalyzer:
        analyzer = getattr(self._thread_local, "analyzer", None)
        if analyzer is None:
            analyzer = VoiceAnalyzer()
            self._thread_local.analyzer = analyzer
        return analyzer

    def _inc_stat(self, key: str, value: int = 1) -> None:
        with self._stats_lock:
            self.stats[key] += value

    def find_audio_files(self, input_dir: Path) -> List[Path]:
        audio_files: List[Path] = []
        for ext in SUPPORTED_AUDIO_FORMATS:
            audio_files.extend(input_dir.rglob(f"*{ext}"))
        return sorted(audio_files)

    def process_file(
        self,
        filepath: Path,
        author: Optional[str] = None,
        author_source: Optional[str] = None,
        tags: Optional[List[str]] = None,
        reliable_quality_rating: Optional[float] = None,
        unreliable_quality_rating: Optional[float] = None,
    ) -> Optional[int]:
        try:
            if self.split_long_audio:
                fragments, sr = self.preprocessor.prepare_fragments(filepath, self._sample_rate)
            else:
                audio, sr = librosa.load(str(filepath), sr=self._sample_rate, mono=True)
                trimmed = self.preprocessor.trim_silence(audio)
                duration = len(trimmed) / sr if sr > 0 else 0.0
                if duration < self.MIN_DURATION_SEC:
                    fragments = []
                else:
                    fragments = [trimmed.astype(np.float32)]

            if not fragments:
                if self.verbose:
                    print(f"  ⏭  Skipped: trimmed audio shorter than {self.MIN_DURATION_SEC:.1f}s")
                self._inc_stat("skipped")
                return None

            first_record_id: Optional[int] = None
            for idx, fragment in enumerate(fragments, start=1):
                file_hash = self._calculate_fragment_hash(fragment, sr)

                if self.skip_existing:
                    existing = self.db.get_recording_by_hash(file_hash)
                    if existing:
                        if self.verbose:
                            print(f"  ⏭  Fragment {idx}/{len(fragments)} already in database (ID: {existing['id']})")
                        self._inc_stat("already_exists")
                        if first_record_id is None:
                            first_record_id = existing["id"]
                        continue

                if self.verbose:
                    print(f"  🔍 Analyzing fragment {idx}/{len(fragments)}...")

                analyzer = self._get_analyzer()
                result = analyzer.analyze_audio(
                    audio=fragment,
                    filepath=str(filepath),
                    include_frames=self.include_frames,
                    verbose=False,
                )
                analysis_json = json.loads(json.dumps(result, cls=VoiceAnalysisEncoder))
                analysis_json["filename"] = f"{filepath.name}#part{idx}"
                analysis_json["source_file"] = str(filepath)
                analysis_json["fragment_index"] = idx
                analysis_json["fragment_count"] = len(fragments)

                if self.verbose:
                    print(f"  🧠 Extracting x-vector embedding for fragment {idx}/{len(fragments)}...")

                try:
                    x_vector = wav_to_embedding(fragment, sr)
                except Exception as e:
                    if self.verbose:
                        print(f"  ⚠️  Warning: Could not extract x-vector: {e}")
                    x_vector = None

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
                    x_vector=x_vector,
                )

                self._inc_stat("processed")
                if first_record_id is None:
                    first_record_id = record_id

                if self.verbose:
                    print(f"  ✅ Fragment stored (ID: {record_id}, duration: {result.duration:.2f}s)")

            if first_record_id is not None:
                return first_record_id

            self._inc_stat("skipped")
            return None

        except Exception as e:
            self._inc_stat("errors")
            print(f"  ❌ Error processing file: {e}")
            if self.verbose:
                traceback.print_exc()
            return None

    @staticmethod
    def _calculate_fragment_hash(fragment: np.ndarray, sample_rate: int) -> str:
        """Stable SHA256 hash for in-memory fragment content."""
        normalized = np.ascontiguousarray(fragment.astype(np.float32, copy=False))
        hasher = hashlib.sha256()
        hasher.update(str(sample_rate).encode("utf-8"))
        hasher.update(b"|")
        hasher.update(normalized.tobytes())
        return hasher.hexdigest()

    def process_directory(
        self,
        input_dir: Path,
        default_author: Optional[str] = None,
        default_source: Optional[str] = None,
        default_tags: Optional[List[str]] = None,
        default_reliable_quality: Optional[float] = None,
        default_unreliable_quality: Optional[float] = None,
        move_processed_to: Optional[Path] = None,
    ) -> Dict[str, Any]:
        audio_files = self.find_audio_files(input_dir)
        self.stats["total"] = len(audio_files)

        if self.stats["total"] == 0:
            print(f"No audio files found in {input_dir}")
            return self.stats

        print(f"Found {self.stats['total']} audio files")

        worker_count = min(self.max_workers, len(audio_files))
        if worker_count <= 1:
            with tqdm(total=self.stats["total"], desc="Processing files") as pbar:
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
                        unreliable_quality_rating=default_unreliable_quality,
                    )

                    if move_processed_to and record_id is not None:
                        self._move_file(filepath, input_dir, move_processed_to)

                    pbar.update(1)
        else:
            with tqdm(total=self.stats["total"], desc=f"Processing files ({worker_count} threads)") as pbar:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    future_to_path = {
                        executor.submit(
                            self.process_file,
                            filepath,
                            default_author,
                            default_source,
                            default_tags,
                            default_reliable_quality,
                            default_unreliable_quality,
                        ): filepath
                        for filepath in audio_files
                    }

                    for future in as_completed(future_to_path):
                        filepath = future_to_path[future]
                        try:
                            record_id = future.result()
                            if move_processed_to and record_id is not None:
                                self._move_file(filepath, input_dir, move_processed_to)
                        except Exception as e:
                            self._inc_stat("errors")
                            print(f"  ❌ Error processing file {filepath}: {e}")
                            if self.verbose:
                                traceback.print_exc()
                        finally:
                            pbar.update(1)

        return self.stats

    def process_metadata_file(
        self,
        metadata_file: Path,
        default_author: Optional[str] = None,
        default_source: Optional[str] = None,
        default_tags: Optional[List[str]] = None,
        default_reliable_quality: Optional[float] = None,
        default_unreliable_quality: Optional[float] = None,
        move_processed_to: Optional[Path] = None,
    ) -> Dict[str, Any]:
        print(f"Reading metadata file: {metadata_file}")
        mf = MetadataFile(str(metadata_file))
        entries = mf.read()
        metadata_base_dir = metadata_file.parent

        errors = validate_metadata_entries(entries, base_dir=metadata_base_dir)
        if errors:
            print("\n⚠️  Validation errors found:")
            for error in errors:
                print(f"  - {error}")

            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != "y":
                print("Aborted.")
                return self.stats

        merged_entries = mf.merge_with_defaults(
            default_author=default_author,
            default_source=default_source,
            default_tags=default_tags,
            default_reliable_quality=default_reliable_quality,
            default_unreliable_quality=default_unreliable_quality,
        )

        self.stats["total"] = len(merged_entries)
        print(f"Processing {self.stats['total']} files from metadata")

        tasks: List[tuple[Path, Any]] = []
        with tqdm(total=self.stats["total"], desc="Processing files") as pbar:
            for entry in merged_entries:
                filepath = Path(entry.filepath)
                if not filepath.is_absolute():
                    filepath = metadata_base_dir / filepath

                if not filepath.exists():
                    print(f"  ⚠️  File not found: {filepath}")
                    self._inc_stat("skipped")
                    pbar.update(1)
                    continue

                tasks.append((filepath, entry))

            worker_count = min(self.max_workers, len(tasks)) if tasks else 1

            if worker_count <= 1:
                for filepath, entry in tasks:
                    pbar.set_description(f"Processing {filepath.name}")

                    if self.verbose:
                        print(f"\n📁 {filepath}")

                    record_id = self.process_file(
                        filepath=filepath,
                        author=entry.author,
                        author_source=entry.author_source,
                        tags=entry.tags,
                        reliable_quality_rating=entry.reliable_quality_rating,
                        unreliable_quality_rating=entry.unreliable_quality_rating,
                    )

                    if move_processed_to and record_id is not None:
                        self._move_file(filepath, metadata_base_dir, move_processed_to)

                    pbar.update(1)
            else:
                pbar.set_description(f"Processing files ({worker_count} threads)")
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    future_to_file = {
                        executor.submit(
                            self.process_file,
                            filepath,
                            entry.author,
                            entry.author_source,
                            entry.tags,
                            entry.reliable_quality_rating,
                            entry.unreliable_quality_rating,
                        ): filepath
                        for filepath, entry in tasks
                    }

                    for future in as_completed(future_to_file):
                        filepath = future_to_file[future]
                        try:
                            record_id = future.result()
                            if move_processed_to and record_id is not None:
                                self._move_file(filepath, metadata_base_dir, move_processed_to)
                        except Exception as e:
                            self._inc_stat("errors")
                            print(f"  ❌ Error processing file {filepath}: {e}")
                            if self.verbose:
                                traceback.print_exc()
                        finally:
                            pbar.update(1)

        return self.stats

    def _move_file(self, filepath: Path, original_base: Path, target_base: Path) -> None:
        try:
            try:
                rel_path = filepath.relative_to(original_base)
            except ValueError:
                rel_path = filepath.name

            target_path = target_base / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(filepath), str(target_path))

            if self.verbose:
                print(f"  📦 Moved to: {target_path}")

        except Exception as e:
            print(f"  ⚠️  Warning: Could not move file: {e}")

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total files:        {self.stats['total']}")
        print(f"Processed:          {self.stats['processed']}")
        print(f"Already existed:    {self.stats['already_exists']}")
        print(f"Skipped:            {self.stats['skipped']}")
        print(f"Errors:             {self.stats['errors']}")
        print("=" * 60)
