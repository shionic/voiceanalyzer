import numpy as np
import soundfile as sf
import librosa

from voiceanalyzer.audio import load_audio_mono


def audio_to_mfcc(
    audio_path,
    sr=22050,
    n_mfcc=40,          # Increased from 13
    n_fft=4096,         # Larger FFT = better freq resolution
    hop_length=256,     # Smaller hop = better time detail
    n_mels=256,         # More Mel bands
    fmin=20,
    fmax=None
):
    y, sr = load_audio_mono(audio_path, target_sr=sr, dtype=np.float32)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )

    return mfcc, sr


def mfcc_to_audio(
    mfcc,
    sr,
    n_fft=4096,
    hop_length=256,
    n_mels=256,
    n_iter=128,        # More Griffin-Lim iterations
    fmin=20,
    fmax=None
):
    # Step 1: MFCC → Mel power spectrogram
    mel_spec = librosa.feature.inverse.mfcc_to_mel(
        mfcc,
        n_mels=n_mels
    )

    # Ensure strictly positive values for numerical stability
    mel_spec = np.maximum(mel_spec, 1e-10)

    # Step 2: Mel → Linear-frequency magnitude spectrogram
    stft_mag = librosa.feature.inverse.mel_to_stft(
        mel_spec,
        sr=sr,
        n_fft=n_fft,
        power=2.0,  # Match MFCC power scale
        fmin=fmin,
        fmax=fmax
    )

    # Step 3: Griffin–Lim phase reconstruction
    y_recon = librosa.griffinlim(
        stft_mag,
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=n_fft,
        window="hann"
    )

    return y_recon


def convert_and_resynthesize(input_audio_path, output_audio_path):
    mfcc, sr = audio_to_mfcc(input_audio_path)

    y_recon = mfcc_to_audio(mfcc, sr)

    sf.write(output_audio_path, y_recon, sr)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python mfcc_resynth.py <input.wav> <output.wav>")
        sys.exit(1)

    convert_and_resynthesize(sys.argv[1], sys.argv[2])