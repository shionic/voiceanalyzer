#!/usr/bin/env python3
"""
Shared voice processing pipeline for CLI and Telegram bot.

Pipeline steps:
1) Load audio and run voice_analyzer.py feature extraction
2) Compute speaker embedding (x-vector style) via ml_funcs.py
3) Compare against database speakers tagged as male/female
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from voiceanalyzer.analysis import VoiceAnalyzer
from voiceanalyzer.audio import preprocess_audio_basic
from voiceanalyzer.embeddings import wav_to_embedding, cosine_similarity
from voiceanalyzer.storage import VoiceDatabase


@dataclass
class SimilarityHit:
    record_id: int
    author: Optional[str]
    similarity: float
    cosine_distance: float
    tags: List[str]
    reference_duration: Optional[float]
    reference_author_source: Optional[str]
    reference_pitch_mean: Optional[float]
    reference_voicing_rate: Optional[float]
    reference_pitch: Dict[str, Optional[float]]
    reference_energy: Dict[str, Optional[float]]
    reference_formants_hz: Dict[str, Optional[float]]
    reference_spectral: Dict[str, Optional[float]]
    diff_pitch_stats: Dict[str, Optional[float]]
    diff_pitch_stats_pct: Dict[str, Optional[float]]
    diff_energy_mean: Optional[float]
    diff_energy_mean_pct: Optional[float]
    diff_formants_hz: Dict[str, Optional[float]]
    diff_formants_pct: Dict[str, Optional[float]]
    diff_spectral: Dict[str, Optional[float]]
    diff_spectral_pct: Dict[str, Optional[float]]


@dataclass
class VoiceMatchOutput:
    filename: str
    duration: float
    male_best: Optional[SimilarityHit]
    female_best: Optional[SimilarityHit]
    male_female_similarity_gap: Optional[float]
    pitch_mean: float
    voicing_rate: float
    pitch_std: float
    pitch_min: float
    pitch_max: float
    pitch_p5: float
    pitch_p95: float
    pitch_median: float
    energy_mean: float
    energy_std: float
    energy_min: float
    energy_max: float
    energy_p5: float
    energy_p95: float
    energy_dynamic_range: float
    formants_hz: Dict[str, Optional[float]]
    spectral: Dict[str, Optional[float]]
    mfcc_mean: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_tags(tags: Optional[List[str]]) -> List[str]:
    if not tags:
        return []
    return [str(t).strip().lower() for t in tags]


class VoiceMatchService:
    def __init__(
        self,
        db_client: Optional[VoiceDatabase] = None,
        analyzer: Optional[VoiceAnalyzer] = None,
    ):
        self.db = db_client or VoiceDatabase(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "voice_analysis"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
        )
        self.analyzer = analyzer or VoiceAnalyzer()

    def process_file(self, audio_path: str) -> VoiceMatchOutput:
        wav, sr = self.analyzer.load_audio(audio_path)
        preprocessed_wav = preprocess_audio_basic(wav)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            sf.write(tmp_path, preprocessed_wav, sr)
            analysis = self.analyzer.analyze(tmp_path, include_frames=False)
            query_vec = wav_to_embedding(preprocessed_wav, sr)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        query_pitch_mean = float(analysis.pitch_statistics.get("mean", 0.0))
        query_voicing_rate = float(analysis.pitch_statistics.get("voicing_rate", 0.0))

        query_energy_mean = float(analysis.energy_statistics.get("mean", 0.0))
        query_pitch_stats = {
            "mean": float(analysis.pitch_statistics.get("mean", 0.0)),
            "std": float(analysis.pitch_statistics.get("std", 0.0)),
            "min": float(analysis.pitch_statistics.get("min", 0.0)),
            "max": float(analysis.pitch_statistics.get("max", 0.0)),
            "p5": float(analysis.pitch_statistics.get("p5", 0.0)),
            "p95": float(analysis.pitch_statistics.get("p95", 0.0)),
            "median": float(analysis.pitch_statistics.get("median", 0.0)),
        }
        query_formants_hz = {
            "f1": getattr(analysis.mean_formants, "f1", None) if analysis.mean_formants else None,
            "f2": getattr(analysis.mean_formants, "f2", None) if analysis.mean_formants else None,
            "f3": getattr(analysis.mean_formants, "f3", None) if analysis.mean_formants else None,
            "f4": getattr(analysis.mean_formants, "f4", None) if analysis.mean_formants else None,
        }
        query_spectral = {
            "centroid": getattr(analysis.global_spectral, "centroid", None) if analysis.global_spectral else None,
            "bandwidth": getattr(analysis.global_spectral, "bandwidth", None) if analysis.global_spectral else None,
            "flatness": getattr(analysis.global_spectral, "flatness", None) if analysis.global_spectral else None,
            "rolloff": getattr(analysis.global_spectral, "rolloff", None) if analysis.global_spectral else None,
            "zero_crossing_rate": getattr(analysis.global_spectral, "zero_crossing_rate", None) if analysis.global_spectral else None,
            "rms_energy": getattr(analysis.global_spectral, "rms_energy", None) if analysis.global_spectral else None,
        }

        male_best = self._best_by_tag(
            query_vec,
            "male",
            query_pitch_mean,
            query_voicing_rate,
            query_pitch_stats=query_pitch_stats,
            query_energy_mean=query_energy_mean,
            query_formants_hz=query_formants_hz,
            query_spectral=query_spectral,
        )
        female_best = self._best_by_tag(
            query_vec,
            "female",
            query_pitch_mean,
            query_voicing_rate,
            query_pitch_stats=query_pitch_stats,
            query_energy_mean=query_energy_mean,
            query_formants_hz=query_formants_hz,
            query_spectral=query_spectral,
        )

        male_female_gap = None
        if male_best is not None and female_best is not None:
            male_female_gap = abs(male_best.similarity - female_best.similarity)

        return VoiceMatchOutput(
            filename=os.path.basename(audio_path),
            duration=analysis.duration,
            male_best=male_best,
            female_best=female_best,
            male_female_similarity_gap=male_female_gap,
            pitch_mean=query_pitch_mean,
            voicing_rate=query_voicing_rate,
            pitch_std=float(analysis.pitch_statistics.get("std", 0.0)),
            pitch_min=float(analysis.pitch_statistics.get("min", 0.0)),
            pitch_max=float(analysis.pitch_statistics.get("max", 0.0)),
            pitch_p5=float(analysis.pitch_statistics.get("p5", 0.0)),
            pitch_p95=float(analysis.pitch_statistics.get("p95", 0.0)),
            pitch_median=float(analysis.pitch_statistics.get("median", 0.0)),
            energy_mean=float(analysis.energy_statistics.get("mean", 0.0)),
            energy_std=float(analysis.energy_statistics.get("std", 0.0)),
            energy_min=float(analysis.energy_statistics.get("min", 0.0)),
            energy_max=float(analysis.energy_statistics.get("max", 0.0)),
            energy_p5=float(analysis.energy_statistics.get("p5", 0.0)),
            energy_p95=float(analysis.energy_statistics.get("p95", 0.0)),
            energy_dynamic_range=float(analysis.energy_statistics.get("dynamic_range", 0.0)),
            formants_hz={
                "f1": getattr(analysis.mean_formants, "f1", None) if analysis.mean_formants else None,
                "f2": getattr(analysis.mean_formants, "f2", None) if analysis.mean_formants else None,
                "f3": getattr(analysis.mean_formants, "f3", None) if analysis.mean_formants else None,
                "f4": getattr(analysis.mean_formants, "f4", None) if analysis.mean_formants else None,
            },
            spectral={
                "centroid": getattr(analysis.global_spectral, "centroid", None) if analysis.global_spectral else None,
                "bandwidth": getattr(analysis.global_spectral, "bandwidth", None) if analysis.global_spectral else None,
                "flatness": getattr(analysis.global_spectral, "flatness", None) if analysis.global_spectral else None,
                "rolloff": getattr(analysis.global_spectral, "rolloff", None) if analysis.global_spectral else None,
                "zero_crossing_rate": getattr(analysis.global_spectral, "zero_crossing_rate", None) if analysis.global_spectral else None,
                "rms_energy": getattr(analysis.global_spectral, "rms_energy", None) if analysis.global_spectral else None,
            },
            mfcc_mean=[float(v) for v in (analysis.global_mfcc or [])],
        )

    def _best_by_tag(
        self,
        query_vec: np.ndarray,
        tag: str,
        query_pitch_mean: float,
        query_voicing_rate: float,
        query_pitch_stats: Optional[Dict[str, Optional[float]]] = None,
        query_energy_mean: float = 0.0,
        query_formants_hz: Optional[Dict[str, Optional[float]]] = None,
        query_spectral: Optional[Dict[str, Optional[float]]] = None,
    ) -> Optional[SimilarityHit]:
        records = self.db.search_by_tags([tag], match_all=False)
        best: Optional[Tuple[Dict[str, Any], float]] = None

        for rec in records:
            db_vec = rec.get("x_vector")
            if db_vec is None:
                continue
            if isinstance(db_vec, list):
                db_vec = np.asarray(db_vec, dtype=np.float32)
            elif not isinstance(db_vec, np.ndarray):
                continue

            score = float(cosine_similarity(query_vec, db_vec))
            if best is None or score > best[1]:
                best = (rec, score)

        if best is None:
            return None

        rec, score = best
        analysis_data = rec.get("analysis_data") or {}
        pitch_stats = analysis_data.get("pitch_statistics") or {}
        energy_stats = analysis_data.get("energy_statistics") or {}
        mean_formants = analysis_data.get("mean_formants") or {}
        global_spectral = analysis_data.get("global_spectral") or {}
        ref_pitch = pitch_stats.get("mean")
        ref_voicing = pitch_stats.get("voicing_rate")

        def pct_delta(delta: Optional[float], ref: Optional[float]) -> Optional[float]:
            if delta is None or ref is None or float(ref) == 0.0:
                return None
            return (float(delta) / float(ref)) * 100.0

        query_pitch_stats = query_pitch_stats or {
            "mean": query_pitch_mean,
            "std": None,
            "min": None,
            "max": None,
            "p5": None,
            "p95": None,
            "median": None,
        }
        diff_pitch_stats: Dict[str, Optional[float]] = {}
        diff_pitch_stats_pct: Dict[str, Optional[float]] = {}
        for k, qv in query_pitch_stats.items():
            rv = pitch_stats.get(k)
            if qv is None or rv is None:
                diff_pitch_stats[k] = None
                diff_pitch_stats_pct[k] = None
            else:
                d = float(qv) - float(rv)
                diff_pitch_stats[k] = d
                diff_pitch_stats_pct[k] = pct_delta(d, float(rv))

        ref_energy_mean = energy_stats.get("mean")
        diff_energy_mean = None
        diff_energy_mean_pct = None
        if ref_energy_mean is not None:
            diff_energy_mean = query_energy_mean - float(ref_energy_mean)
            diff_energy_mean_pct = pct_delta(diff_energy_mean, float(ref_energy_mean))

        query_formants_hz = query_formants_hz or {}
        diff_formants_hz: Dict[str, Optional[float]] = {}
        diff_formants_pct: Dict[str, Optional[float]] = {}
        for k in ("f1", "f2", "f3", "f4"):
            qv = query_formants_hz.get(k)
            rv = mean_formants.get(k)
            if qv is None or rv is None:
                diff_formants_hz[k] = None
                diff_formants_pct[k] = None
            else:
                d = float(qv) - float(rv)
                diff_formants_hz[k] = d
                diff_formants_pct[k] = pct_delta(d, float(rv))

        query_spectral = query_spectral or {}
        diff_spectral: Dict[str, Optional[float]] = {}
        diff_spectral_pct: Dict[str, Optional[float]] = {}
        for k in ("centroid", "bandwidth", "flatness", "rolloff", "zero_crossing_rate", "rms_energy"):
            qv = query_spectral.get(k)
            rv = global_spectral.get(k)
            if qv is None or rv is None:
                diff_spectral[k] = None
                diff_spectral_pct[k] = None
            else:
                d = float(qv) - float(rv)
                diff_spectral[k] = d
                diff_spectral_pct[k] = pct_delta(d, float(rv))

        return SimilarityHit(
            record_id=int(rec["id"]),
            author=rec.get("author"),
            similarity=score,
            cosine_distance=1.0 - score,
            tags=rec.get("tags") or [],
            reference_duration=rec.get("duration"),
            reference_author_source=rec.get("author_source"),
            reference_pitch_mean=float(ref_pitch) if ref_pitch is not None else None,
            reference_voicing_rate=float(ref_voicing) if ref_voicing is not None else None,
            reference_pitch={
                "mean": float(pitch_stats.get("mean")) if pitch_stats.get("mean") is not None else None,
                "std": float(pitch_stats.get("std")) if pitch_stats.get("std") is not None else None,
                "min": float(pitch_stats.get("min")) if pitch_stats.get("min") is not None else None,
                "max": float(pitch_stats.get("max")) if pitch_stats.get("max") is not None else None,
                "p5": float(pitch_stats.get("p5")) if pitch_stats.get("p5") is not None else None,
                "p95": float(pitch_stats.get("p95")) if pitch_stats.get("p95") is not None else None,
                "median": float(pitch_stats.get("median")) if pitch_stats.get("median") is not None else None,
                "voicing_rate": float(pitch_stats.get("voicing_rate")) if pitch_stats.get("voicing_rate") is not None else None,
            },
            reference_energy={
                "mean": float(energy_stats.get("mean")) if energy_stats.get("mean") is not None else None,
                "std": float(energy_stats.get("std")) if energy_stats.get("std") is not None else None,
                "min": float(energy_stats.get("min")) if energy_stats.get("min") is not None else None,
                "max": float(energy_stats.get("max")) if energy_stats.get("max") is not None else None,
                "p5": float(energy_stats.get("p5")) if energy_stats.get("p5") is not None else None,
                "p95": float(energy_stats.get("p95")) if energy_stats.get("p95") is not None else None,
                "dynamic_range": float(energy_stats.get("dynamic_range")) if energy_stats.get("dynamic_range") is not None else None,
            },
            reference_formants_hz={
                "f1": float(mean_formants.get("f1")) if mean_formants.get("f1") is not None else None,
                "f2": float(mean_formants.get("f2")) if mean_formants.get("f2") is not None else None,
                "f3": float(mean_formants.get("f3")) if mean_formants.get("f3") is not None else None,
                "f4": float(mean_formants.get("f4")) if mean_formants.get("f4") is not None else None,
            },
            reference_spectral={
                "centroid": float(global_spectral.get("centroid")) if global_spectral.get("centroid") is not None else None,
                "bandwidth": float(global_spectral.get("bandwidth")) if global_spectral.get("bandwidth") is not None else None,
                "flatness": float(global_spectral.get("flatness")) if global_spectral.get("flatness") is not None else None,
                "rolloff": float(global_spectral.get("rolloff")) if global_spectral.get("rolloff") is not None else None,
                "zero_crossing_rate": float(global_spectral.get("zero_crossing_rate")) if global_spectral.get("zero_crossing_rate") is not None else None,
                "rms_energy": float(global_spectral.get("rms_energy")) if global_spectral.get("rms_energy") is not None else None,
            },
            diff_pitch_stats=diff_pitch_stats,
            diff_pitch_stats_pct=diff_pitch_stats_pct,
            diff_energy_mean=diff_energy_mean,
            diff_energy_mean_pct=diff_energy_mean_pct,
            diff_formants_hz=diff_formants_hz,
            diff_formants_pct=diff_formants_pct,
            diff_spectral=diff_spectral,
            diff_spectral_pct=diff_spectral_pct,
        )


def format_output_text(result: VoiceMatchOutput) -> str:
    def fmt_opt_float(v: Optional[float], digits: int = 2) -> str:
        if v is None:
            return "n/a"
        return f"{v:.{digits}f}"

    def _name(hit: SimilarityHit) -> str:
        if hit.author and len(hit.author) <= 48:
            return hit.author
        # Fallback for long hash-like IDs
        return f"record#{hit.record_id}"

    def fmt_ref_block() -> str:
        male = result.male_best
        female = result.female_best

        if male is None and female is None:
            return "No reference records found for tags: male/female"

        def fmt_one(label: str, hit: Optional[SimilarityHit]) -> List[str]:
            if hit is None:
                return [f"{label}: not found"]
            return [
                f"{label}: {_name(hit)}",
                f"  - similarity: {hit.similarity:.4f}",
                f"  - distance: {hit.cosine_distance:.4f}",
                f"  - tags: {', '.join(hit.tags) if hit.tags else 'n/a'}",
                f"  - ref duration: {fmt_opt_float(hit.reference_duration)}s",
                f"  - ref source: {hit.reference_author_source or 'n/a'}",
                f"  - ref pitch mean: {fmt_opt_float(hit.reference_pitch_mean)} Hz",
                f"  - Δenergy mean: {fmt_opt_float(hit.diff_energy_mean, 6)} ({fmt_opt_float(hit.diff_energy_mean_pct)}%)",
                (
                    "  - ref pitch stats: "
                    f"mean={fmt_opt_float(hit.reference_pitch.get('mean'))}, "
                    f"std={fmt_opt_float(hit.reference_pitch.get('std'))}, "
                    f"min={fmt_opt_float(hit.reference_pitch.get('min'))}, "
                    f"max={fmt_opt_float(hit.reference_pitch.get('max'))}, "
                    f"p5={fmt_opt_float(hit.reference_pitch.get('p5'))}, "
                    f"p95={fmt_opt_float(hit.reference_pitch.get('p95'))}, "
                    f"median={fmt_opt_float(hit.reference_pitch.get('median'))}"
                ),
                (
                    "  - Δpitch stats (input-ref): "
                    f"Δmean={fmt_opt_float(hit.diff_pitch_stats.get('mean'))} ({fmt_opt_float(hit.diff_pitch_stats_pct.get('mean'))}%), "
                    f"Δstd={fmt_opt_float(hit.diff_pitch_stats.get('std'))} ({fmt_opt_float(hit.diff_pitch_stats_pct.get('std'))}%), "
                    f"Δmin={fmt_opt_float(hit.diff_pitch_stats.get('min'))} ({fmt_opt_float(hit.diff_pitch_stats_pct.get('min'))}%), "
                    f"Δmax={fmt_opt_float(hit.diff_pitch_stats.get('max'))} ({fmt_opt_float(hit.diff_pitch_stats_pct.get('max'))}%), "
                    f"Δp5={fmt_opt_float(hit.diff_pitch_stats.get('p5'))} ({fmt_opt_float(hit.diff_pitch_stats_pct.get('p5'))}%), "
                    f"Δp95={fmt_opt_float(hit.diff_pitch_stats.get('p95'))} ({fmt_opt_float(hit.diff_pitch_stats_pct.get('p95'))}%), "
                    f"Δmedian={fmt_opt_float(hit.diff_pitch_stats.get('median'))} ({fmt_opt_float(hit.diff_pitch_stats_pct.get('median'))}%)"
                ),
                (
                    "  - ref spectral: "
                    f"centroid={fmt_opt_float(hit.reference_spectral.get('centroid'))}, "
                    f"bandwidth={fmt_opt_float(hit.reference_spectral.get('bandwidth'))}, "
                    f"rolloff={fmt_opt_float(hit.reference_spectral.get('rolloff'))}, "
                    f"flatness={fmt_opt_float(hit.reference_spectral.get('flatness'), 6)}, "
                    f"zcr={fmt_opt_float(hit.reference_spectral.get('zero_crossing_rate'), 6)}, "
                    f"rms={fmt_opt_float(hit.reference_spectral.get('rms_energy'), 6)}"
                ),
                (
                    "  - ref formants (Hz): "
                    f"F1={fmt_opt_float(hit.reference_formants_hz.get('f1'))}, "
                    f"F2={fmt_opt_float(hit.reference_formants_hz.get('f2'))}, "
                    f"F3={fmt_opt_float(hit.reference_formants_hz.get('f3'))}, "
                    f"F4={fmt_opt_float(hit.reference_formants_hz.get('f4'))}"
                ),
                (
                    "  - Δformants (input-ref, Hz): "
                    f"ΔF1={fmt_opt_float(hit.diff_formants_hz.get('f1'))} ({fmt_opt_float(hit.diff_formants_pct.get('f1'))}%), "
                    f"ΔF2={fmt_opt_float(hit.diff_formants_hz.get('f2'))} ({fmt_opt_float(hit.diff_formants_pct.get('f2'))}%), "
                    f"ΔF3={fmt_opt_float(hit.diff_formants_hz.get('f3'))} ({fmt_opt_float(hit.diff_formants_pct.get('f3'))}%), "
                    f"ΔF4={fmt_opt_float(hit.diff_formants_hz.get('f4'))} ({fmt_opt_float(hit.diff_formants_pct.get('f4'))}%)"
                ),
                (
                    "  - Δspectral (input-ref): "
                    f"Δcentroid={fmt_opt_float(hit.diff_spectral.get('centroid'))} ({fmt_opt_float(hit.diff_spectral_pct.get('centroid'))}%), "
                    f"Δbandwidth={fmt_opt_float(hit.diff_spectral.get('bandwidth'))} ({fmt_opt_float(hit.diff_spectral_pct.get('bandwidth'))}%), "
                    f"Δrolloff={fmt_opt_float(hit.diff_spectral.get('rolloff'))} ({fmt_opt_float(hit.diff_spectral_pct.get('rolloff'))}%), "
                    f"Δflatness={fmt_opt_float(hit.diff_spectral.get('flatness'), 6)} ({fmt_opt_float(hit.diff_spectral_pct.get('flatness'))}%), "
                    f"Δzcr={fmt_opt_float(hit.diff_spectral.get('zero_crossing_rate'), 6)} ({fmt_opt_float(hit.diff_spectral_pct.get('zero_crossing_rate'))}%), "
                    f"Δrms={fmt_opt_float(hit.diff_spectral.get('rms_energy'), 6)} ({fmt_opt_float(hit.diff_spectral_pct.get('rms_energy'))}%)"
                ),
            ]

        lines: List[str] = []
        lines.extend(fmt_one("MALE reference", male))
        lines.append("")
        lines.extend(fmt_one("FEMALE reference", female))

        if male is not None and female is not None:
            lines.extend([
                "",
                "Male vs Female comparison:",
                f"- |similarity_male - similarity_female|: {abs(male.similarity - female.similarity):.4f}",
                f"- better match: {'male' if male.similarity >= female.similarity else 'female'}",
            ])

        return "\n".join(lines)

    lines = [
        "Voice Analysis Summary",
        "=" * 24,
        f"Input file: {result.filename}",
        f"Duration: {result.duration:.2f}s",
        "",
        "PITCH:",
        f"- Mean: {result.pitch_mean:.2f} Hz",
        f"- Std: {result.pitch_std:.2f} Hz",
        f"- Min/Max: {result.pitch_min:.2f} / {result.pitch_max:.2f} Hz",
        f"- P5/P95: {result.pitch_p5:.2f} / {result.pitch_p95:.2f} Hz",
        f"- Median: {result.pitch_median:.2f} Hz",
        f"- Voicing rate: {result.voicing_rate:.2%}",
        "",
        "SPECTRAL:",
        f"- Centroid: {fmt_opt_float(result.spectral.get('centroid'))} Hz",
        f"- Bandwidth: {fmt_opt_float(result.spectral.get('bandwidth'))} Hz",
        f"- Rolloff: {fmt_opt_float(result.spectral.get('rolloff'))} Hz",
        f"- Flatness: {fmt_opt_float(result.spectral.get('flatness'), 6)}",
        f"- Zero Crossing Rate: {fmt_opt_float(result.spectral.get('zero_crossing_rate'), 6)}",
        f"- RMS (spectral): {fmt_opt_float(result.spectral.get('rms_energy'), 6)}",
        "",
        "FORMANTS:",
        (
            "- Mean formants (Hz): "
            f"F1={fmt_opt_float(result.formants_hz.get('f1'))}, "
            f"F2={fmt_opt_float(result.formants_hz.get('f2'))}, "
            f"F3={fmt_opt_float(result.formants_hz.get('f3'))}, "
            f"F4={fmt_opt_float(result.formants_hz.get('f4'))}"
        ),
        "",
        "ENERGY:",
        f"- Mean: {result.energy_mean:.6f}",
        f"- Std: {result.energy_std:.6f}",
        f"- Min/Max: {result.energy_min:.6f} / {result.energy_max:.6f}",
        f"- P5/P95: {result.energy_p5:.6f} / {result.energy_p95:.6f}",
        f"- Dynamic range: {result.energy_dynamic_range:.6f}",
        "",
        "MFCC:",
        f"- Mean coefficients ({len(result.mfcc_mean)}): " + ", ".join(f"{v:.3f}" for v in result.mfcc_mean),
        "",
        "REFERENCE PROXIMITY (by tag):",
        fmt_ref_block(),
    ]

    return "\n".join(lines)
