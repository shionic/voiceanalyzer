import numpy as np
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from scipy.spatial.distance import cosine

_THREADS_CONFIGURED = False

# Load once at module import (fast at runtime after first load)
_classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": "cpu"}  # change to "cuda" if GPU is available
)


def configure_torch_threads(intra_op_threads: int | None = None, inter_op_threads: int | None = None) -> None:
    """Configure torch CPU thread pools once per process.

    Useful to avoid oversubscription when many worker threads are used.
    """
    global _THREADS_CONFIGURED
    if _THREADS_CONFIGURED:
        return

    if intra_op_threads is not None and intra_op_threads > 0:
        torch.set_num_threads(intra_op_threads)
    if inter_op_threads is not None and inter_op_threads > 0:
        torch.set_num_interop_threads(inter_op_threads)

    _THREADS_CONFIGURED = True


# ------------------------------------------------------------
# Audio preprocessing
# ------------------------------------------------------------

def preprocess_audio(wav: np.ndarray, sr: int, target_sr: int = 16000) -> torch.Tensor:
    """
    Resample and normalize audio for the speaker model.

    Args:
        wav: 1D numpy array (mono audio)
        sr: original sample rate
        target_sr: required sample rate for the model (16 kHz)

    Returns:
        torch.Tensor shape [1, num_samples]
    """
    if wav.ndim != 1:
        raise ValueError("Audio must be mono (1D numpy array).")

    wav_tensor = torch.tensor(wav, dtype=torch.float32)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav_tensor = resampler(wav_tensor)

    # Normalize amplitude (robust scaling)
    max_val = torch.max(torch.abs(wav_tensor))
    if max_val > 0:
        wav_tensor = wav_tensor / max_val

    return wav_tensor.unsqueeze(0)  # shape: [1, time]


# ------------------------------------------------------------
# Embedding extraction
# ------------------------------------------------------------

def wav_to_embedding(wav: np.ndarray, sr: int) -> np.ndarray:
    """
    Convert raw waveform into a fixed-length speaker embedding vector.

    Args:
        wav: mono audio waveform (numpy)
        sr: sample rate

    Returns:
        1D numpy array speaker embedding
    """
    signal = preprocess_audio(wav, sr)

    with torch.no_grad():
        embedding = _classifier.encode_batch(signal)

    # Shape: [1, 1, embedding_dim] → flatten
    embedding = embedding.squeeze().cpu().numpy()

    # L2 normalize (important for cosine similarity)
    norm = np.linalg.norm(embedding, ord=2)
    if norm > 0:
        embedding = embedding / norm

    return embedding

    # ------------------------------------------------------------
# Similarity / comparison
# ------------------------------------------------------------

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two speaker embeddings.
    Returns value in range [-1, 1]; higher = more similar.
    """
    return 1.0 - cosine(vec1, vec2)


def is_same_speaker(vec1: np.ndarray, vec2: np.ndarray, threshold: float = 0.65) -> bool:
    """
    Decide if two embeddings belong to the same speaker.

    Threshold depends on use case:
        ~0.60–0.75 typical for ECAPA embeddings.

    Returns:
        True if similarity >= threshold
    """
    sim = cosine_similarity(vec1, vec2)
    return sim >= threshold


# ------------------------------------------------------------
# Database comparison utilities
# ------------------------------------------------------------

def find_most_similar(query_vec: np.ndarray, db_vectors: dict) -> tuple:
    """
    Compare a query embedding to a database of embeddings.

    Args:
        query_vec: embedding of input voice
        db_vectors: dict {speaker_id: embedding_vector}

    Returns:
        (best_speaker_id, similarity_score)
    """
    best_id = None
    best_score = -1.0

    for speaker_id, db_vec in db_vectors.items():
        score = cosine_similarity(query_vec, db_vec)
        if score > best_score:
            best_score = score
            best_id = speaker_id

    return best_id, best_score