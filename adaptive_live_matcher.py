"""
MUSIC RETRIEVAL SYSTEM - ADAPTIVE LIVE MATCHER
===============================================
Handles weak microphone input by adapting to actual audio characteristics
"""

import sqlite3
import math
import numpy as np  # FIX: was "from numba import np"
from collections import defaultdict
from typing import Any

from live_capture import capture_audio
from fingerprint_features import extract_peaks_from_array
from recommend_similar import get_recommendations, get_normalization_stats, get_song_vector
from build_similarity_index import compute_mfcc_vector_from_array

# ==========================================
# ADAPTIVE FEATURE EXTRACTION
# ==========================================
def extract_peaks_adaptive(audio_array, neighborhood_size=25, max_peaks=2000, desired_min_peaks=120):
    """
    Adaptive extraction for short/quiet live captures.
    Starts with a strict 75th-percentile threshold (parity with DB), then
    backs off the percentile in steps if too few peaks are found. Also
    down-samples very large peak sets to avoid hash explosions.
    """
    print("\n[Feature Extraction] Detecting spectral peaks (adaptive locked mode)...")

    start_percentile = 75.0
    min_percentile = 50.0
    step = 5.0

    p = start_percentile
    peaks, stats = extract_peaks_from_array(
        audio_array,
        sr=22050,
        neighborhood_size=neighborhood_size,
        min_percentile=p,
        max_percentile=p,
        percentile_step=1.0,
        min_peak_density=0.0,
        freq_bin_min=10,
        freq_bin_max=450,
    )

    while len(peaks) < desired_min_peaks and p > min_percentile:
        p = max(min_percentile, p - step)
        print(f"  Backing off percentile -> {p:.1f}% (found {len(peaks)} peaks)")
        peaks, stats = extract_peaks_from_array(
            audio_array,
            sr=22050,
            neighborhood_size=neighborhood_size,
            min_percentile=p,
            max_percentile=p,
            percentile_step=1.0,
            min_peak_density=0.0,
            freq_bin_min=10,
            freq_bin_max=450,
        )

    if len(peaks) < max(10, desired_min_peaks // 4) and neighborhood_size > 10:
        print("  Few peaks remain after percentile backoff; trying smaller neighborhood...")
        peaks_small_neigh, stats_small = extract_peaks_from_array(
            audio_array,
            sr=22050,
            neighborhood_size=max(10, neighborhood_size // 2),
            min_percentile=p,
            max_percentile=p,
            percentile_step=1.0,
            min_peak_density=0.0,
            freq_bin_min=10,
            freq_bin_max=450,
        )
        if len(peaks_small_neigh) > len(peaks):
            peaks, stats = peaks_small_neigh, stats_small
            print(f"  Improved peak count with smaller neighborhood: {len(peaks)}")

    print(f"  PCEN shape: {stats['shape']}")
    print(f"  Percentile used: {stats['percentile']:.1f}")
    print(f"  Threshold value: {stats['threshold']:.5f}")
    print(f"  Peaks extracted (raw): {len(peaks)}")

    if len(peaks) > max_peaks:
        peaks_sorted = sorted(peaks, key=lambda x: x[0])
        idx = np.linspace(0, len(peaks_sorted) - 1, max_peaks).astype(int)
        peaks = [peaks_sorted[i] for i in idx]
        print(f"  Peaks downsampled to: {len(peaks)} (max_peaks={max_peaks})")

    print(f"  Peaks returned: {len(peaks)}")
    return peaks


# ==========================================
# HASH GENERATION
# ==========================================
def generate_hashes_optimized(peaks, delay_min=10, delay_max=50, delta_freq=250, fan_out=4, freq_fuzz=2, time_fuzz=2):
    """
    True Shazam 'Target Zone' implementation with Hash Quantization (Fuzzy Fingerprinting).
    """
    if not peaks:
        return []

    peaks = sorted(peaks, key=lambda x: x[0])
    hash_list = []

    for i in range(len(peaks)):
        anchor_time, anchor_freq = peaks[i]
        connections_made = 0

        for j in range(i + 1, len(peaks)):
            if connections_made >= fan_out:
                break

            target_time, target_freq = peaks[j]
            time_diff = target_time - anchor_time

            if time_diff < delay_min:
                continue
            if time_diff > delay_max:
                break

            if abs(target_freq - anchor_freq) <= delta_freq:
                q_anchor_freq = int(anchor_freq // freq_fuzz)
                q_target_freq = int(target_freq // freq_fuzz)
                q_time_diff = int(time_diff // time_fuzz)

                hash_str = f"{q_anchor_freq}|{q_target_freq}|{q_time_diff}"
                hash_list.append((hash_str, int(anchor_time)))
                connections_made += 1

    return hash_list


# ==========================================
# MFCC TIE-BREAKER
# ==========================================
AMBIGUITY_RATIO_THRESHOLD = 1.3


def _normalize_live_vector(raw_vector, means, stds):
    """Projects a raw MFCC vector into the same z-scored space as the stored index."""
    v = np.array(raw_vector)
    stds_safe = np.where(stds == 0, 1.0, stds)
    return (v - means) / stds_safe


def resolve_ambiguous_match(audio_array, candidates, cursor, sr=22050, top_k=3):
    """
    When fingerprint evidence is ambiguous between top candidates, use MFCC
    timbral similarity (computed live) as a tie-breaker.
    """
    norm_stats = get_normalization_stats()
    if norm_stats is None:
        print(" ⚠ Tie-breaker skipped: no normalization stats available.")
        return candidates

    means, stds, n_mfcc, expected_sr = norm_stats

    try:
        live_raw_vector = compute_mfcc_vector_from_array(audio_array, sr=sr, n_mfcc=n_mfcc)
        live_norm_vector = _normalize_live_vector(live_raw_vector, means, stds)
    except Exception as e:
        print(f" ⚠ Tie-breaker skipped: MFCC extraction failed ({e}).")
        return candidates

    reranked = []
    for candidate in candidates[:top_k]:
        cursor.execute("SELECT song_name FROM songs WHERE song_id = ?", (candidate["song_id"],))
        result = cursor.fetchone()
        song_name = result[0] if result else None

        stored_vector = get_song_vector(song_name) if song_name else None
        if stored_vector is None:
            timbral_sim = 0.0
        else:
            a, b = np.array(live_norm_vector), np.array(stored_vector)
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            timbral_sim = float(np.dot(a, b) / denom) if denom > 0 else 0.0

        combined_score = candidate["score"] * (0.6 + 0.4 * max(0.0, timbral_sim))
        reranked.append({**candidate, "timbral_sim": timbral_sim, "combined_score": combined_score})

    reranked.sort(key=lambda c: c["combined_score"], reverse=True)

    print("\n[Tie-Breaker] MFCC-adjusted ranking:")
    for rank, c in enumerate(reranked, 1):
        print(f"  {rank}. combined={c['combined_score']:.3f} | fp_score={c['score']:.3f} | "
              f"timbral_sim={c['timbral_sim']:.3f} | song_id={c['song_id']}")

    return reranked + candidates[top_k:]


# ==========================================
# CONFIDENCE-BASED MATCHING
# ==========================================
def identify_song(
    live_hashes,
    audio_array,
    db_path="database/song_index.db",
    max_hash_collisions=1500,
    min_weighted_score=1.8,
    min_raw_aligned=3,
    min_total_raw=10,
    min_top12_ratio=1.08,
):
    """
    Queries database using time-delta histogram matching.
    Returns (winning_song, recommendations, error) or (None, [], None) on fail.
    """
    print(f"\n[Database Query] Searching with {len(live_hashes)} live hashes...")

    if not live_hashes:
        print("  ✗ ERROR: No hashes generated!")
        return None, [], None

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    song_scores = {}
    song_totals = {}
    total_matches = 0
    total_weight = 0.0

    dedup_hashes = defaultdict(list)
    for hash_str, live_offset in live_hashes:
        offsets = dedup_hashes[hash_str]
        if len(offsets) < 6:
            offsets.append(int(live_offset))

    for hash_str, live_offsets in dedup_hashes.items():
        cursor.execute("SELECT song_id, offset FROM fingerprints WHERE hash_str = ?", (hash_str,))
        matches = cursor.fetchall()
        if len(matches) > max_hash_collisions:
            continue
        repetition_boost = math.sqrt(len(live_offsets))
        hash_weight = (1.0 / math.log2(2.0 + len(matches))) * repetition_boost

        for live_offset in live_offsets:
            for db_song_id, db_offset in matches:
                total_matches += 1
                total_weight += hash_weight

                if isinstance(db_offset, bytes):
                    db_offset_int = int.from_bytes(db_offset, byteorder='little')
                else:
                    db_offset_int = int(db_offset)

                time_delta_bin = int(round((db_offset_int - live_offset) / 2.0))

                if db_song_id not in song_scores:
                    song_scores[db_song_id] = {}
                    song_totals[db_song_id] = {"weighted": 0.0, "raw": 0}
                if time_delta_bin not in song_scores[db_song_id]:
                    song_scores[db_song_id][time_delta_bin] = {"weighted": 0.0, "raw": 0}
                song_scores[db_song_id][time_delta_bin]["weighted"] += hash_weight
                song_scores[db_song_id][time_delta_bin]["raw"] += 1
                song_totals[db_song_id]["weighted"] += hash_weight
                song_totals[db_song_id]["raw"] += 1

    print(f"  Total hash matches found: {total_matches}")
    print(f"  Unique query hashes used: {len(dedup_hashes)}")

    if not song_scores:
        print("  ✗ No matching patterns in database")
        conn.close()
        return None, [], None

    candidates = []

    song_ids = list(song_scores.keys())
    song_db_counts = {}
    if song_ids:
        placeholders = ",".join(["?"] * len(song_ids))
        try:
            cursor.execute(
                f"SELECT song_id, COUNT(*) FROM fingerprints WHERE song_id IN ({placeholders}) GROUP BY song_id",
                tuple(song_ids),
            )
            for sid, cnt in cursor.fetchall():
                song_db_counts[sid] = cnt or 0
        except Exception:
            song_db_counts = {}

    for db_song_id, delta_buckets in song_scores.items():
        if not delta_buckets:
            continue
        best_delta, best_stats = max(delta_buckets.items(), key=lambda item: item[1]["weighted"])
        peak_weighted = best_stats["weighted"]
        raw_aligned = best_stats["raw"]
        total_raw_song = song_totals[db_song_id]["raw"]
        total_weighted_song = song_totals[db_song_id]["weighted"]

        song_db_fps = song_db_counts.get(db_song_id, 0)
        base_score = peak_weighted + 0.25 * math.log2(1.0 + total_raw_song)
        size_norm = max(1.0, math.log2(2.0 + song_db_fps) ** 1.5)
        align_ratio = raw_aligned / max(1.0, float(total_raw_song))
        align_bonus = min(2.5, align_ratio * 2.5)
        alignment_coherence = peak_weighted / max(1e-9, total_weighted_song)
        weighted_score = (base_score / size_norm) * (1.0 + align_bonus) * (0.5 + alignment_coherence)
        confidence = total_weighted_song / max(total_weight, 1e-9)

        candidates.append({
            "song_id": db_song_id,
            "score": weighted_score,
            "peak_weighted": peak_weighted,
            "raw_aligned": raw_aligned,
            "total_raw": total_raw_song,
            "confidence": confidence,
            "best_delta": best_delta,
        })

    candidates.sort(key=lambda x: x['score'], reverse=True)
    top1 = candidates[0]["score"] if candidates else 0.0
    top2 = candidates[1]["score"] if len(candidates) > 1 else 0.0
    top12_ratio = (top1 / top2) if top2 > 0 else float("inf")

    # NEW: MFCC tie-breaker for ambiguous matches
    if top12_ratio < AMBIGUITY_RATIO_THRESHOLD and len(candidates) > 1:
        print(f"\n⚠ Ambiguous match (ratio={top12_ratio:.2f} < {AMBIGUITY_RATIO_THRESHOLD}). "
              f"Running MFCC tie-breaker...")
        candidates = resolve_ambiguous_match(audio_array, candidates, cursor, sr=22050)
        top1 = candidates[0]["score"] if candidates else 0.0
        top2 = candidates[1]["score"] if len(candidates) > 1 else 0.0
        top12_ratio = (top1 / top2) if top2 > 0 else float("inf")

    print(f"\n[Results] Top matches:")
    print(f"  {'Rank':<6} {'Song Name':<34} {'Score':<7} {'PeakW':<7} {'Raw':<4} {'Tot':<4} {'Conf'}")
    print(f"  {'-'*95}")

    for rank, candidate in enumerate(candidates[:5], 1):
        cursor.execute("SELECT song_name FROM songs WHERE song_id = ?", (candidate['song_id'],))
        result = cursor.fetchone()
        song_name = result[0][:40] if result else "Unknown"
        confidence_bar = "█" * int(candidate['confidence'] * 20)
        print(
            f"  {rank:<6} {song_name[:34]:<34} {candidate['score']:<7.2f} "
            f"{candidate['peak_weighted']:<7.2f} {candidate['raw_aligned']:<4} "
            f"{candidate['total_raw']:<4} {confidence_bar}"
        )

    print(f"\n  Top1/Top2 ratio: {top12_ratio:.2f}")

    if (
        candidates
        and candidates[0]["score"] >= min_weighted_score
        and (
            candidates[0]["raw_aligned"] >= min_raw_aligned
            or candidates[0]["total_raw"] >= min_total_raw
        )
        and top12_ratio >= min_top12_ratio
    ):
        best = candidates[0]
        cursor.execute("SELECT song_name FROM songs WHERE song_id = ?", (best['song_id'],))
        result = cursor.fetchone()
        winning_song = result[0]

        print(f"\n{'='*75}")
        print(f"✓ MATCH FOUND: '{winning_song}'")
        print(f"  Weighted Score: {best['score']:.2f}")
        print(f"  Peak Weighted: {best['peak_weighted']:.2f}")
        print(f"  Raw Aligned Votes: {best['raw_aligned']}")
        print(f"  Total Song Matches: {best['total_raw']}")
        print(f"  Top1/Top2 Ratio: {top12_ratio:.2f}")
        print(f"  Confidence: {best['confidence']:.1%}")
        print(f"{'='*75}\n")

        conn.close()
        recommendations, rec_error = get_recommendations(winning_song)
        return winning_song, recommendations, rec_error
    else:
        if candidates:
            best = candidates[0]
            relaxed_score_thresh = min_weighted_score * 0.75
            relaxed_raw_aligned = 2
            relaxed_total_raw = 6
            relaxed_ratio = 1.02
            if (
                best['score'] >= relaxed_score_thresh
                and (best['raw_aligned'] >= relaxed_raw_aligned or best['total_raw'] >= relaxed_total_raw)
                and top12_ratio >= relaxed_ratio
            ):
                cursor.execute("SELECT song_name FROM songs WHERE song_id = ?", (best['song_id'],))
                result = cursor.fetchone()
                winning_song = result[0] if result else 'Unknown'
                print(f"\n{'='*75}")
                print(f"⚠ WEAK MATCH (relaxed): '{winning_song}'")
                print(f"  Weighted Score: {best['score']:.2f} (relaxed threshold {relaxed_score_thresh:.2f})")
                print(f"  Raw Aligned Votes: {best['raw_aligned']} | Total: {best['total_raw']}")
                print(f"  Top1/Top2 Ratio: {top12_ratio:.2f}")
                print(f"  Confidence: {best['confidence']:.1%}")
                print(f"{'='*75}\n")
                conn.close()
                recommendations, rec_error = get_recommendations(winning_song)
                return winning_song, recommendations, rec_error

        print(f"\n⚠ Match confidence too low. Try:")
        print(f"  1. Increasing speaker volume")
        print(f"  2. Moving microphone closer")
        print(f"  3. Recording for longer duration")
        conn.close()
        return None, [], None


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*75)
    print("MUSIC RETRIEVAL SYSTEM - ADAPTIVE LIVE QUERY")
    print("="*75)

    print("\n[STEP 1] Capturing audio from microphone...")
    print("         Song should be playing at MAXIMUM volume!")

    live_audio, sr = capture_audio(duration=15)

    print("\n[STEP 2] Extracting spectral features (locked mode)...")
    live_peaks = extract_peaks_adaptive(live_audio)

    if len(live_peaks) < 3:
        print(f"\n✗ CRITICAL: Only {len(live_peaks)} peaks detected!")
        print("\nTroubleshooting:")
        print("  1. Check Windows microphone input level")
        print("  2. Test microphone with: python -c \"import sounddevice as sd; sd.rec(44100, blocking=True); print('Recorded')\"")
        print("  3. Ensure speaker/headphone volume is at MAXIMUM")
    else:
        print("\n[STEP 3] Generating fingerprints...")
        live_hashes = generate_hashes_optimized(live_peaks)
        print(f"  Hashes generated: {len(live_hashes)}")

        if live_hashes:
            print("\n[STEP 4] Querying database...")
            winning_song, recommendations, rec_error = identify_song(live_hashes, live_audio)
            if winning_song and recommendations:
                print("\nRecommended similar songs:")
                for name, score in recommendations:
                    print(f"  {score:.3f}  {name}")
            elif winning_song and rec_error:
                print(f"\n{rec_error}")
        else:
            print("  ✗ Could not generate hashes from peaks")

    print("="*75 + "\n")