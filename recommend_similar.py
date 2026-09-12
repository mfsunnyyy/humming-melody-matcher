"""
CONTENT-BASED SONG RECOMMENDATION
====================================
Given a matched song name, finds the most similar songs using cosine
similarity over precomputed MFCC vectors (timbral fingerprint).

CHANGE: Updated to read the new nested index structure that also
stores normalization stats (means/stds), needed for projecting
live-captured audio into the same feature space for tie-breaking.
"""

import json
import numpy as np
import os

SIMILARITY_INDEX_PATH = "database/similarity_index.json"


def _load_index():
    """Loads the precomputed Z-scored MFCC index and normalization stats from disk."""
    if not os.path.exists(SIMILARITY_INDEX_PATH):
        return None

    with open(SIMILARITY_INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _cosine_similarity(vec_a, vec_b):
    """Computes the cosine similarity between two numeric vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def get_recommendations(matched_song_name, top_n=3):
    """
    Returns a list of (song_name, similarity_score) tuples for the most
    similar songs to the matched one. Returns an empty list with a reason
    string if the matched song has no similarity vector available.
    """
    payload = _load_index()

    if payload is None:
        return [], "⚠ Recommendation index not found. Run 'build_similarity_index.py' first."

    index = payload.get("songs", {})

    if matched_song_name not in index:
        return [], "⚠ No similarity data available for this specific track."

    target_vector = index[matched_song_name]
    scores = []

    for song_name, vector in index.items():
        if song_name == matched_song_name:
            continue
        score = _cosine_similarity(target_vector, vector)
        scores.append((song_name, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_matches = scores[:top_n]

    if not top_matches:
        return [], "⚠ Not enough other songs in the database to compare against."

    return top_matches, None


def get_normalization_stats():
    """
    NEW: Exposes the dataset's MFCC mean/std vectors, needed by the
    upcoming tie-breaker logic in adaptive_live_matcher.py to normalize
    a live-captured MFCC vector into the same space as the stored index.
    Returns (means, stds, n_mfcc, sr) or None if unavailable.
    """
    payload = _load_index()
    if payload is None:
        return None

    norm = payload.get("normalization")
    if norm is None:
        return None

    return (
        np.array(norm["means"]),
        np.array(norm["stds"]),
        norm.get("n_mfcc", 13),
        norm.get("sr", 22050),
    )


def get_song_vector(song_name):
    """
    NEW: Returns the stored normalized MFCC vector for a given song name,
    or None if not found. Used by the tie-breaker to fetch vectors for
    the top ambiguous candidates without reloading the whole index repeatedly.
    """
    payload = _load_index()
    if payload is None:
        return None
    index = payload.get("songs", {})
    return index.get(song_name)


if __name__ == "__main__":
    print("\n--- Testing Recommendation Engine ---")

    payload = _load_index()
    if not payload:
        print(f"Error: {SIMILARITY_INDEX_PATH} missing. Please run build_similarity_index.py.")
    else:
        index_data = payload.get("songs", {})
        if not index_data:
            print("Error: Index loaded but contains no songs.")
        else:
            test_song = list(index_data.keys())[0]
            print(f"Target Song: '{test_song}'\n")

            results, error = get_recommendations(test_song)
            if error:
                print(error)
            else:
                print("Recommended Matches:")
                for rank, (name, score) in enumerate(results, 1):
                    print(f"  {rank}. {score:.4f} | {name}")
            print("-------------------------------------\n")

            norm_stats = get_normalization_stats()
            if norm_stats:
                means, stds, n_mfcc, sr = norm_stats
                print(f"Normalization stats loaded: n_mfcc={n_mfcc}, sr={sr}, vector_len={len(means)}")
            else:
                print("⚠ No normalization stats found in index.")