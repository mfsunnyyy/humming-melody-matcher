"""
CONTENT-BASED SIMILARITY INDEX BUILDER
========================================
Computes an averaged MFCC vector per song (timbral fingerprint) and stores
it so recommendations can be looked up instantly at query time, instead of
recomputing audio analysis for all songs on every query.

CHANGE: Now also persists the dataset's mean/std vectors used for
z-score normalization, so live-captured audio can later be normalized
into the same feature space for tie-breaking ambiguous fingerprint matches.
"""

import os
import json
import numpy as np
import librosa
import time

AUDIO_DIR = "database/raw_audio"
OUTPUT_FILE = "database/similarity_index.json"


def compute_mfcc_vector(file_path, n_mfcc=13, sr=22050):
    """
    Loads a song and reduces it to a single vector representing its
    average timbral characteristics (MFCC mean + std across the track).
    Drops coefficient 0 (overall loudness/energy) since it isn't a timbral
    feature and its large magnitude otherwise dominates similarity scoring.
    """
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    return compute_mfcc_vector_from_array(y, sr=sr, n_mfcc=n_mfcc)


def compute_mfcc_vector_from_array(audio_array, sr=22050, n_mfcc=13):
    """
    Same as compute_mfcc_vector but operates on an in-memory audio array.
    Used both for database building and for live-captured query audio,
    ensuring identical feature computation on both sides.
    """
    mfcc = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=n_mfcc)
    mfcc = mfcc[1:, :]  # drop c0 (energy) - keep only c1..c12 (timbral shape)

    mean_vec = np.mean(mfcc, axis=1)
    std_vec = np.std(mfcc, axis=1)

    combined = np.concatenate([mean_vec, std_vec])
    return combined.tolist()


def build_index():
    print("\n" + "=" * 75)
    print("PHASE 1B: BUILDING MFCC SIMILARITY INDEX")
    print("=" * 75)

    if not os.path.isdir(AUDIO_DIR):
        print(f"✗ ERROR: Directory '{AUDIO_DIR}' not found. Please create it and add .wav files.")
        return

    wav_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
    total = len(wav_files)

    if total == 0:
        print(f"✗ ERROR: No .wav files found in '{AUDIO_DIR}'.")
        return

    print(f"Computing timbral vectors for {total} songs...\n")

    start_time = time.time()
    raw_index = {}

    for i, filename in enumerate(wav_files, 1):
        song_name = filename.replace(".wav", "")
        file_path = os.path.join(AUDIO_DIR, filename)
        try:
            vector = compute_mfcc_vector(file_path)
            raw_index[song_name] = vector
            print(f"[{i:2d}/{total}] ✓ EXTRACTED | {song_name[:55]}")
        except Exception as e:
            print(f"[{i:2d}/{total}] ✗ FAILED | {song_name[:55]}")
            print(f"    Error: {str(e)[:60]}")

    print("\n[Step 2] Applying Z-Score Normalization across dataset...")
    names = list(raw_index.keys())
    matrix = np.array([raw_index[n] for n in names])

    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds == 0] = 1.0  # prevent divide-by-zero for constant dimensions

    normalized_matrix = (matrix - means) / stds
    index = {names[i]: normalized_matrix[i].tolist() for i in range(len(names))}

    # NEW: persist normalization stats alongside the index so live queries
    # can be projected into the exact same normalized feature space later.
    output_payload = {
        "songs": index,
        "normalization": {
            "means": means.tolist(),
            "stds": stds.tolist(),
            "n_mfcc": 13,
            "sr": 22050,
        },
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=4)

    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print("SIMILARITY INDEX BUILD COMPLETE")
    print("=" * 75)
    print(f"Songs indexed: {len(index)}")
    print(f"Build time: {elapsed:.1f} seconds")
    print(f"File saved to: {OUTPUT_FILE}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    build_index()