import sqlite3
import json
import math
from typing import Dict, List, Tuple

import numpy as np
import librosa


def hz_to_midi_safe(f0_hz: np.ndarray) -> np.ndarray:
    out = np.full_like(f0_hz, np.nan, dtype=np.float32)
    mask = f0_hz > 0
    out[mask] = librosa.hz_to_midi(f0_hz[mask])
    return out


def extract_melody_features_from_array(
    audio: np.ndarray,
    sr: int = 22050,
    fmin_hz: float = 80.0,
    fmax_hz: float = 800.0,
    hop_length: int = 256,
) -> Dict:
    if audio.ndim > 1:
        audio = audio[:, 0]

    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) == 0:
        return {"ok": False, "reason": "empty_audio"}

    audio = audio - np.mean(audio)
    peak = np.max(np.abs(audio))
    if peak > 1e-8:
        audio = audio / peak

    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=fmin_hz,
        fmax=fmax_hz,
        sr=sr,
        hop_length=hop_length,
        frame_length=2048,
    )

    if f0 is None or len(f0) == 0:
        return {"ok": False, "reason": "no_pitch"}

    f0_hz = np.asarray(f0, dtype=np.float32)
    midi = hz_to_midi_safe(f0_hz)

    vf = (
        np.asarray(voiced_flag, dtype=bool)
        if voiced_flag is not None
        else ~np.isnan(midi)
    )
    vp = (
        np.asarray(voiced_prob, dtype=np.float32)
        if voiced_prob is not None
        else np.ones_like(midi, dtype=np.float32)
    )

    mask = vf & ~np.isnan(midi) & (vp >= 0.55)
    voiced = midi[mask]

    if len(voiced) < 20:
        return {
            "ok": False,
            "reason": "too_few_voiced_frames",
            "voiced_frames": int(len(voiced)),
        }

    if len(voiced) >= 5:
        kernel = np.ones(5, dtype=np.float32) / 5.0
        voiced = np.convolve(voiced, kernel, mode="same")

    centered = voiced - np.median(voiced)
    intervals = np.diff(centered)

    if len(intervals) < 10:
        return {"ok": False, "reason": "too_short_interval_seq"}

    intervals = np.clip(intervals, -12.0, 12.0)
    scale = np.std(intervals)
    if scale < 1e-6:
        return {"ok": False, "reason": "flat_intervals"}

    return {
        "ok": True,
        "voiced_midi_centered": centered.astype(np.float32),
        "intervals": (intervals / scale).astype(np.float32),
        "voiced_frames": int(len(voiced)),
        "hop_length": int(hop_length),
        "sr": int(sr),
    }


def extract_melody_features_from_file(path: str, sr: int = 22050) -> Dict:
    audio, sr = librosa.load(path, sr=sr, mono=True)
    return extract_melody_features_from_array(audio, sr=sr)


def dtw_distance(a: np.ndarray, b: np.ndarray, band_ratio: float = 0.15) -> float:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")

    band = max(5, int(max(n, m) * band_ratio))
    inf = 1e18
    dp = np.full((n + 1, m + 1), inf, dtype=np.float64)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - band)
        j_end = min(m, i + band)
        ai = a[i - 1]

        for j in range(j_start, j_end + 1):
            cost = abs(float(ai) - float(b[j - 1]))
            dp[i, j] = cost + min(
                dp[i - 1, j],
                dp[i, j - 1],
                dp[i - 1, j - 1],
            )

    distance = dp[n, m]
    if distance >= inf / 10:
        return float("inf")

    return float(distance / (n + m))


def setup_melody_table(conn: sqlite3.Connection):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS melody_features (
            song_id INTEGER PRIMARY KEY,
            intervals_json TEXT NOT NULL,
            voiced_len INTEGER NOT NULL,
            FOREIGN KEY(song_id) REFERENCES songs(song_id)
        )
        """
    )
    conn.commit()


def upsert_song_melody(
    conn: sqlite3.Connection,
    song_id: int,
    intervals: np.ndarray,
):
    conn.execute(
        """
        INSERT INTO melody_features
            (song_id, intervals_json, voiced_len)
        VALUES (?, ?, ?)
        ON CONFLICT(song_id) DO UPDATE SET
            intervals_json=excluded.intervals_json,
            voiced_len=excluded.voiced_len
        """,
        (song_id, json.dumps(intervals.tolist()), int(len(intervals))),
    )
    conn.commit()


def load_all_melodies(
    conn: sqlite3.Connection,
) -> List[Tuple[int, np.ndarray]]:
    rows = conn.execute(
        "SELECT song_id, intervals_json FROM melody_features"
    ).fetchall()

    result = []
    for song_id, intervals_json in rows:
        sequence = np.asarray(
            json.loads(intervals_json),
            dtype=np.float32,
        )
        result.append((int(song_id), sequence))
    return result


def song_id_to_name(conn: sqlite3.Connection, song_id: int) -> str:
    row = conn.execute(
        "SELECT song_name FROM songs WHERE song_id = ?",
        (song_id,),
    ).fetchone()
    return row[0] if row else str(song_id)


def match_humming_intervals(
    query_intervals: np.ndarray,
    conn: sqlite3.Connection,
    top_k: int = 5,
) -> List[Dict]:
    scored = []

    for song_id, sequence in load_all_melodies(conn):
        if len(sequence) < 10:
            continue

        distance = dtw_distance(query_intervals, sequence)
        if math.isinf(distance):
            continue

        scored.append((distance, song_id))

    scored.sort(key=lambda item: item[0])

    return [
        {
            "song_id": song_id,
            "song_name": song_id_to_name(conn, song_id),
            "distance": float(distance),
        }
        for distance, song_id in scored[:top_k]
    ]


def humming_confident(
    top: List[Dict],
    query_type: str = "humming",
) -> Tuple[bool, Dict]:
    if not top:
        return False, {"reason": "no_candidates"}

    if query_type == "cover":
        max_dist = 0.11
        max_ratio = 0.82
        min_gap = 0.03
    else:
        max_dist = 0.09
        max_ratio = 0.78
        min_gap = 0.04

    d1 = top[0]["distance"]
    d2 = top[1]["distance"] if len(top) > 1 else d1 + 1.0
    ratio = d1 / max(d2, 1e-9)
    gap = d2 - d1

    accepted = (
        d1 <= max_dist
        and ratio <= max_ratio
        and gap >= min_gap
    )

    return accepted, {
        "d1": d1,
        "d2": d2,
        "ratio": ratio,
        "gap": gap,
        "max_dist": max_dist,
        "max_ratio": max_ratio,
        "min_gap": min_gap,
        "query_type": query_type,
    }


def identify_humming_live(
    audio: np.ndarray,
    sr: int = 22050,
    db_path: str = "database/song_index.db",
    top_k: int = 5,
    query_type: str = "humming",
):
    features = extract_melody_features_from_array(audio, sr=sr)
    if not features.get("ok"):
        return None, [], {
            "reason": features.get("reason", "feature_fail")
        }

    conn = sqlite3.connect(db_path)
    try:
        top = match_humming_intervals(
            features["intervals"],
            conn,
            top_k=top_k,
        )
        accepted, info = humming_confident(top, query_type=query_type)

        if accepted:
            return top[0]["song_name"], top, info

        if top and len(top) > 1:
            info["reason"] = "small_top_gap" if info["gap"] < info["min_gap"] else "low_confidence"
        else:
            info["reason"] = "insufficient_candidates"

        return None, top, info
    finally:
        conn.close()
