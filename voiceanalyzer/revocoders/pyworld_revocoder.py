import argparse
import numpy as np
import soundfile as sf
import pyworld as pw
import os

from voiceanalyzer.audio import load_audio_mono


def load_wav(path):
    return load_audio_mono(path, target_sr=None, dtype=np.float64)


def extract_world_features(x, fs, frame_period=5.0):
    f0_raw, t = pw.dio(x, fs, frame_period=frame_period)
    f0 = pw.stonemask(x, f0_raw, t, fs)
    sp = pw.cheaptrick(x, f0, t, fs)
    ap = pw.d4c(x, f0, t, fs)
    return f0, sp, ap


def synthesize_world(f0, sp, ap, fs):
    return pw.synthesize(f0, sp, ap, fs)


def save_features(out_dir, f0, sp, ap):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "f0.npy"), f0)
    np.save(os.path.join(out_dir, "sp.npy"), sp)
    np.save(os.path.join(out_dir, "ap.npy"), ap)


def load_features(in_dir):
    f0 = np.load(os.path.join(in_dir, "f0.npy"))
    sp = np.load(os.path.join(in_dir, "sp.npy"))
    ap = np.load(os.path.join(in_dir, "ap.npy"))
    return f0, sp, ap


def main():
    parser = argparse.ArgumentParser(description="WORLD vocoder feature extraction and resynthesis utility")
    parser.add_argument("--input_wav", type=str, help="Path to input wav file")
    parser.add_argument("--feature_dir", type=str, help="Directory to save/load WORLD features")
    parser.add_argument("--output_wav", type=str, required=True, help="Path to output wav file")
    parser.add_argument("--pitch_scale", type=float, default=1.0, help="Scale factor for F0 (e.g., 1.2 raises pitch)")
    parser.add_argument("--frame_period", type=float, default=5.0, help="Frame period in ms (WORLD default 5.0)")
    parser.add_argument("--mode", choices=["extract", "synthesize", "both"], default="both")

    args = parser.parse_args()

    if args.mode in ("extract", "both"):
        if not args.input_wav or not args.feature_dir:
            raise ValueError("Extraction requires --input_wav and --feature_dir")

        x, fs = load_wav(args.input_wav)
        f0, sp, ap = extract_world_features(x, fs, args.frame_period)
        save_features(args.feature_dir, f0, sp, ap)

    if args.mode in ("synthesize", "both"):
        if not args.feature_dir:
            raise ValueError("Synthesis requires --feature_dir")

        f0, sp, ap = load_features(args.feature_dir)

        if args.pitch_scale != 1.0:
            f0 = f0 * args.pitch_scale

        # Need sampling rate; recover from input wav if provided, else store separately in practice
        if args.input_wav:
            _, fs = load_wav(args.input_wav)
        else:
            raise ValueError("Sampling rate required. Provide --input_wav used during extraction.")

        y = synthesize_world(f0, sp, ap, fs)
        sf.write(args.output_wav, y, fs)


if __name__ == "__main__":
    main()
