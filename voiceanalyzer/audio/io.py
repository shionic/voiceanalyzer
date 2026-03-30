"""Shared audio I/O helpers."""

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
