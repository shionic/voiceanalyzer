"""Dataset-specific metadata transformation helpers."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import List, Optional

from voiceanalyzer.constants import SUPPORTED_AUDIO_FORMATS
from voiceanalyzer.metadata.metadata_file import MetadataEntry


AUTHOR_SOURCE_MCV = "MozillaCommonVoices"
AUTHOR_SOURCE_VOXCELEB2 = "VoxCeleb2"


def normalize_gender_tag(gender_value: str) -> List[str]:
    if not gender_value:
        return []
    g = gender_value.strip().lower()
    if g in {"f", "female", "woman", "girl"}:
        return ["female"]
    if g in {"m", "male", "man", "boy"}:
        return ["male"]
    if "female" in g or "feminine" in g:
        return ["female"]
    if "male" in g or "masculine" in g:
        return ["male"]
    return []


def normalize_age_tag(age_value: str) -> List[str]:
    if not age_value:
        return []
    return [age_value.strip().lower()]


def compute_unreliable_rating(up_votes: str, down_votes: str) -> Optional[float]:
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

            tags = [
                *normalize_gender_tag(row.get("gender")),
                *normalize_age_tag(row.get("age")),
            ]

            entries.append(
                MetadataEntry(
                    filepath=str(Path("clips") / rel_path),
                    author=row.get("client_id"),
                    author_source=AUTHOR_SOURCE_MCV,
                    tags=tags,
                    reliable_quality_rating=None,
                    unreliable_quality_rating=compute_unreliable_rating(
                        row.get("up_votes"),
                        row.get("down_votes"),
                    ),
                )
            )

    return entries


def _extract_voxceleb2_speaker_id(audio_path: Path) -> Optional[str]:
    for part in audio_path.parts:
        if re.fullmatch(r"id\d+", part):
            return part
    return None


def process_voxceleb2(input_dir: Path) -> List[MetadataEntry]:
    meta_path = input_dir / "vox2_meta.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"vox2_meta.csv not found in {input_dir}")

    speaker_meta = {}
    with open(meta_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            normalized = {k.strip().lower(): (v.strip() if isinstance(v, str) else v) for k, v in row.items() if k}
            vox_id = (
                normalized.get("voxceleb2 id")
                or normalized.get("id")
                or normalized.get("speaker_id")
                or next(
                    (
                        v.strip()
                        for v in normalized.values()
                        if isinstance(v, str) and re.fullmatch(r"id\d+", v.strip())
                    ),
                    None,
                )
            )
            if not vox_id:
                continue
            speaker_meta[vox_id] = {
                "vggface2_id": normalized.get("vggface2 id"),
                "gender": normalized.get("gender"),
                "set": normalized.get("set"),
            }

    # Some VoxCeleb2 CSV variants may contain only speaker IDs without metadata columns.
    # In that case DictReader can produce no rows, so we fallback to plain CSV parsing.
    if not speaker_meta:
        with open(meta_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                for value in row:
                    if not isinstance(value, str):
                        continue
                    candidate = value.strip()
                    if re.fullmatch(r"id\d+", candidate):
                        speaker_meta[candidate] = {
                            "vggface2_id": None,
                            "gender": None,
                            "set": None,
                        }

    entries: List[MetadataEntry] = []
    for speaker_id, meta in speaker_meta.items():
        speaker_dir = input_dir / "aac" / speaker_id
        if not speaker_dir.exists():
            continue

        # VoxCeleb2 distributions can store audio under `aac/` but with different
        # extensions (e.g. .aac or .m4a depending on the source/package).
        # Iterate all files and keep only supported audio extensions.
        for file_path in speaker_dir.rglob("*"):
            if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_AUDIO_FORMATS:
                continue

            rel_path = file_path.relative_to(input_dir)
            dataset_set = meta.get("set")
            tags = [*normalize_gender_tag(meta.get("gender"))]
            if dataset_set:
                tags.append(dataset_set.lower())

            entries.append(
                MetadataEntry(
                    filepath=str(rel_path),
                    author=speaker_id,
                    author_source=AUTHOR_SOURCE_VOXCELEB2,
                    tags=tags,
                    reliable_quality_rating=None,
                    unreliable_quality_rating=None,
                    vggface2_id=meta.get("vggface2_id"),
                    voxceleb2_set=dataset_set.lower() if dataset_set else None,
                )
            )

    return entries
