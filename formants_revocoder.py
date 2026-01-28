import numpy as np
import librosa
import pyworld as pw
import parselmouth
import soundfile as sf
from scipy.signal import lfilter

# ------------------------------------------------------------
# Utility: Load audio
# ------------------------------------------------------------
def load_audio(path, target_sr=16000):
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    y = y.astype(np.float64)  # WORLD requires float64
    return y, sr

# ------------------------------------------------------------
# Step 1: Extract pitch (F0) using WORLD
# ------------------------------------------------------------
def extract_pitch_world(y, sr, frame_period=5.0):
    f0, t = pw.harvest(y, sr, frame_period=frame_period)
    f0 = pw.stonemask(y, f0, t, sr)  # refinement
    return f0, t

# ------------------------------------------------------------
# Step 2: Extract spectral envelope + aperiodicity
# ------------------------------------------------------------
def extract_world_features(y, sr, f0, t):
    sp = pw.cheaptrick(y, f0, t, sr)   # spectral envelope
    ap = pw.d4c(y, f0, t, sr)          # aperiodicity
    return sp, ap

# ------------------------------------------------------------
# Step 3: Extract formants using Praat via Parselmouth
# ------------------------------------------------------------
def extract_formants(y, sr, time_step=0.005, max_formant=5500, n_formants=5):
    snd = parselmouth.Sound(y, sampling_frequency=sr)
    formant = snd.to_formant_burg(time_step=time_step,
                                  max_number_of_formants=n_formants,
                                  maximum_formant=max_formant)

    times = np.arange(0, snd.duration, time_step)
    formants = []

    for t in times:
        frame = []
        for i in range(1, 5):  # F1–F4
            val = formant.get_value_at_time(i, t)
            frame.append(val if val else 0.0)
        formants.append(frame)

    return np.array(formants), times

# ------------------------------------------------------------
# Step 4: Impose formant structure onto spectral envelope
# ------------------------------------------------------------
def emphasize_formants(sp, sr, formants, bandwidth=80):
    """
    Boost spectral energy near each detected formant.
    This makes the resynthesis reflect the extracted vocal tract resonances.
    """
    n_frames, fft_bins = sp.shape
    freqs = np.linspace(0, sr/2, fft_bins)

    modified_sp = sp.copy()

    for i in range(min(n_frames, len(formants))):
        for f in formants[i]:
            if f <= 0:
                continue
            # Gaussian emphasis around each formant frequency
            gaussian = np.exp(-0.5 * ((freqs - f) / bandwidth) ** 2)
            modified_sp[i] *= (1.0 + 2.0 * gaussian)

    return modified_sp

# ------------------------------------------------------------
# Step 5: Resynthesize speech
# ------------------------------------------------------------
def resynthesize(f0, sp, ap, sr, frame_period=5.0):
    y = pw.synthesize(f0, sp, ap, sr, frame_period)
    return y

# ------------------------------------------------------------
# Full pipeline
# ------------------------------------------------------------
def analyze_and_resynthesize(input_path, output_path):
    print("Loading audio...")
    y, sr = load_audio(input_path)

    print("Extracting pitch (F0)...")
    f0, t = extract_pitch_world(y, sr)

    print("Extracting WORLD spectral features...")
    sp, ap = extract_world_features(y, sr, f0, t)

    print("Extracting formants...")
    formants, ftimes = extract_formants(y, sr)

    print("Shaping spectral envelope using formants...")
    sp_mod = emphasize_formants(sp, sr, formants)

    print("Resynthesizing...")
    y_syn = resynthesize(f0, sp_mod, ap, sr)

    print("Saving output...")
    sf.write(output_path, y_syn, sr)
    print("Done.")

# ------------------------------------------------------------
# Run as script
# ------------------------------------------------------------


# -----------------------------
# CLI Entry Point
# -----------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python vocoder.py input.wav output.wav")
        sys.exit(1)

    analyze_and_resynthesize(sys.argv[1], sys.argv[2])