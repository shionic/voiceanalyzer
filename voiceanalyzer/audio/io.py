"""Shared audio I/O and preprocessing helpers."""

from __future__ import annotations

from typing import Tuple

import librosa
import numpy as np


def load_audio_mono(
    path: str,
    target_sr: int | None = None,
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, int]:
    """Load audio as mono and optionally resample.

    Args:
        path: Audio file path.
        target_sr: Target sample rate; if None, keep original rate.
        dtype: Output numpy dtype.

    Returns:
        (audio, sample_rate)
    """
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio.astype(dtype, copy=False), sr


def suppress_noise_basic(audio: np.ndarray) -> np.ndarray:
    """Apply lightweight spectral subtraction noise suppression."""
    if audio.size == 0:
        return audio

    n_fft = 1024
    hop_length = 256
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(stft), np.angle(stft)

    noise_profile = np.percentile(magnitude, 10, axis=1, keepdims=True)
    suppression_factor = 1.5
    denoised_mag = np.maximum(magnitude - suppression_factor * noise_profile, 0.0)

    denoised_stft = denoised_mag * np.exp(1j * phase)
    denoised = librosa.istft(denoised_stft, hop_length=hop_length, length=len(audio))
    return denoised.astype(np.float32)


def trim_silence(audio: np.ndarray, top_db: int = 30) -> np.ndarray:
    """Trim leading/trailing silence."""
    if audio.size == 0:
        return audio
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed.astype(np.float32)


def preprocess_audio_basic(audio: np.ndarray, silence_top_db: int = 30) -> np.ndarray:
    """Run the default preprocessing chain used by ingestion/matching flows."""
    denoised = suppress_noise_basic(audio)
    return trim_silence(denoised, top_db=silence_top_db)
