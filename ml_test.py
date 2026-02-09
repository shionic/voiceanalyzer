import argparse
import numpy as np
import librosa
from ml_funcs import wav_to_embedding, cosine_similarity, is_same_speaker

def load_audio(file_path):
    """
    Load a WAV file as a mono float32 numpy array
    """
    wav, sr = librosa.load(file_path, sr=None, mono=True)
    return wav, sr

def main():
    parser = argparse.ArgumentParser(description="Speaker embedding utility")
    parser.add_argument(
        "--files", "-f",
        nargs="+",
        required=True,
        help="Paths to WAV files to process"
    )
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="Compare embeddings of the provided files (must be exactly 2 files)"
    )
    args = parser.parse_args()

    embeddings = {}
    for file_path in args.files:
        wav, sr = load_audio(file_path)
        emb = wav_to_embedding(wav, sr)
        embeddings[file_path] = emb
        print(f"Embedding for {file_path}:\n{emb}\n")

    if args.compare:
        if len(args.files) != 2:
            print("Error: --compare requires exactly 2 files")
            return

        f1, f2 = args.files
        emb1, emb2 = embeddings[f1], embeddings[f2]
        sim = cosine_similarity(emb1, emb2)
        same = is_same_speaker(emb1, emb2)
        print(f"Similarity between {f1} and {f2}: {sim:.4f}")
        print("Same speaker?" , "Yes" if same else "No")

if __name__ == "__main__":
    main()