import sqlite3
import math
import collections
import time
import argparse

from live_capture import capture_audio
from adaptive_live_matcher import generate_hashes_optimized, extract_peaks_adaptive, resolve_ambiguous_match

# ---------- Default Config ----------
DB_PATH = "database/song_index.db"
RECORD_SECONDS = 10
WIN_S = 8.0
STEP_S = 4.0
MAX_HASH_COLLISIONS = 120
MIN_WINS_RATIO = 0.5
MIN_WEIGHTED_TOTAL = 2.0
DEFAULT_MIN_TOP12_RATIO = 1.5
DEFAULT_MAX_PEAKS = 2000
AMBIGUITY_RATIO_THRESHOLD = 1.3  # NEW: below this, trigger MFCC tie-breaker
# ----------------------------

# NEW: module-level default so decide_winner() never hits an undefined global
# even if called from a script (like multi_window_batch_eval.py) that never
# sets it explicitly via the __main__ block below.
MIN_TOP12_RATIO = DEFAULT_MIN_TOP12_RATIO


def ascii_safe(s):
    try:
        return s.encode('ascii', 'replace').decode('ascii')
    except Exception:
        return ''.join(ch if ord(ch) < 128 else '?' for ch in str(s))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['auto', 'strict', 'aggressive'], default='auto',
                    help='Matching mode: auto (try strict then aggressive), strict, or aggressive')
    p.add_argument('--record', type=float, default=RECORD_SECONDS, help='Recording duration in seconds')
    p.add_argument('--win', type=float, default=WIN_S, help='Window length in seconds')
    p.add_argument('--step', type=float, default=STEP_S, help='Window step in seconds')
    p.add_argument('--log', type=str, default='multi_window_query_log.json', help='Path to write run diagnostics (JSON)')
    return p.parse_args()


def query_db_hashes(hash_list, db_path=DB_PATH, max_hash_collisions=MAX_HASH_COLLISIONS):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    totals = collections.defaultdict(lambda: [0.0, 0])  # song_id -> [weighted, raw]
    counts = collections.Counter(h for h, _ in hash_list)
    for hash_str, live_offset in hash_list:
        matches = cur.execute(
            "SELECT song_id, offset FROM fingerprints WHERE hash_str = ?", (hash_str,)
        ).fetchall()
        if not matches or len(matches) > max_hash_collisions:
            continue
        idf_weight = 1.0 / math.log2(2.0 + len(matches))
        rep_boost = math.sqrt(min(4, counts[hash_str]))
        weight = idf_weight * rep_boost
        for song_id, db_offset in matches:
            totals[song_id][0] += weight
            totals[song_id][1] += 1
    conn.close()
    return totals


def process_window(audio_segment, sr):
    peaks = extract_peaks_adaptive(audio_segment)
    hashes = generate_hashes_optimized(peaks)
    return query_db_hashes(hashes)


def process_window_aggressive(audio_segment, sr):
    peaks = extract_peaks_adaptive(audio_segment)
    hashes = generate_hashes_optimized(
        peaks,
        delay_min=5,
        delay_max=300,
        delta_freq=400,
        fan_out=6,
        freq_fuzz=1,
        time_fuzz=1,
    )
    return query_db_hashes(hashes)


def split_windows(audio, sr, win_s=WIN_S, step_s=STEP_S):
    win_samples = int(win_s * sr)
    step_samples = int(step_s * sr)
    windows = []
    if len(audio) <= win_samples:
        return [audio]
    for start in range(0, len(audio) - win_samples + 1, step_samples):
        windows.append(audio[start:start + win_samples])
    if (len(audio) - win_samples) % step_samples != 0 and len(audio) > win_samples:
        windows.append(audio[-win_samples:])
    return windows


def aggregate_window_results(window_results):
    agg = {}
    for res in window_results:
        for sid, (w, r) in res.items():
            if sid not in agg:
                agg[sid] = {"weighted": 0.0, "raw": 0, "wins": 0}
            if w > 0:
                agg[sid]["wins"] += 1
            agg[sid]["weighted"] += w
            agg[sid]["raw"] += r
    return agg


def id_to_name(song_id, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT song_name FROM songs WHERE song_id = ?", (song_id,))
    r = cur.fetchone()
    conn.close()
    return r[0] if r else str(song_id)


def decide_winner(agg, num_windows, audio_for_tiebreak=None, db_path=DB_PATH):
    """
    Decides the winning song from aggregated multi-window evidence.

    NEW: audio_for_tiebreak (optional) — the full captured audio array.
    If provided and the top1/top2 margin is ambiguous, runs the MFCC
    timbral tie-breaker (shared with adaptive_live_matcher.py) to
    re-rank the top candidates before applying acceptance thresholds.
    """
    if not agg:
        return None, None

    items = sorted([(v["weighted"], sid, v) for sid, v in agg.items()], reverse=True)
    top1_w, top1_sid, top1_v = items[0]
    top2_w = items[1][0] if len(items) > 1 else 0.0
    ratio = (top1_w / top2_w) if top2_w > 0 else float("inf")

    # NEW: MFCC tie-breaker for ambiguous multi-window results
    if audio_for_tiebreak is not None and ratio < AMBIGUITY_RATIO_THRESHOLD and len(items) > 1:
        print(f"\n⚠ Ambiguous multi-window match (ratio={ratio:.2f} < {AMBIGUITY_RATIO_THRESHOLD}). "
              f"Running MFCC tie-breaker...")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        pseudo_candidates = [
            {"song_id": sid, "score": w, **v} for w, sid, v in items[:3]
        ]
        try:
            reranked = resolve_ambiguous_match(audio_for_tiebreak, pseudo_candidates, cursor, sr=22050)
        finally:
            conn.close()

        if reranked:
            top1_sid = reranked[0]["song_id"]
            top1_v = agg[top1_sid]
            top1_w = reranked[0].get("combined_score", top1_w)
            top2_w = reranked[1].get("combined_score", top2_w) if len(reranked) > 1 else 0.0
            ratio = (top1_w / top2_w) if top2_w > 0 else float("inf")

    min_wins = max(1, int(math.ceil(num_windows * MIN_WINS_RATIO)))
    if (top1_v["wins"] >= min_wins and top1_w >= MIN_WEIGHTED_TOTAL and ratio >= MIN_TOP12_RATIO) or (
        top1_v["raw"] >= 8 and top1_w >= MIN_WEIGHTED_TOTAL and ratio >= 1.2
    ):
        return top1_sid, {"score": top1_w, **top1_v, "ratio": ratio}
    return None, {"top_candidates": items[:5], "ratio": ratio, "num_windows": num_windows}


if __name__ == "__main__":
    args = parse_args()
    mode = args.mode
    RECORD = args.record
    WIN = args.win
    STEP = args.step

    if mode == 'strict':
        MIN_TOP12_RATIO = DEFAULT_MIN_TOP12_RATIO
        max_peaks = DEFAULT_MAX_PEAKS
        use_aggressive = False
    else:
        MIN_TOP12_RATIO = 1.05
        max_peaks = 1000
        use_aggressive = True

    print(f"Multi-window consensus recognition — mode={mode}, record={RECORD}s")
    audio, sr = capture_audio(duration=RECORD)
    print("Splitting into windows...")
    windows = split_windows(audio, sr, WIN, STEP)
    print(f"Created {len(windows)} windows (win {WIN}s step {STEP}s)")

    window_results = []
    for i, w in enumerate(windows, 1):
        print(f"\nProcessing window {i}/{len(windows)}...")
        peaks = extract_peaks_adaptive(w, max_peaks=max_peaks)
        hashes = generate_hashes_optimized(peaks)
        res = query_db_hashes(hashes)
        print(f"  Candidate songs this window: {len(res)}")
        window_results.append(res)

    agg = aggregate_window_results(window_results)
    winner_sid, info = decide_winner(agg, len(windows), audio_for_tiebreak=audio)  # NEW: pass audio

    diagnostics = {
        'mode': mode,
        'record_seconds': RECORD,
        'win_s': WIN,
        'step_s': STEP,
        'num_windows': len(windows),
        'windows_summary': [],
        'results': {},
    }

    if winner_sid:
        winner_name = id_to_name(winner_sid)
        print('\nDETECTED:', ascii_safe(winner_name), info)
        diagnostics['results']['strict'] = {
            'winner_id': winner_sid,
            'winner_name': winner_name,
            'info': info,
        }
    else:
        print('\nNO CONFIDENT MATCH. Diagnostics:', info)
        strict_candidates = []
        for score, sid, stats in info.get('top_candidates', []):
            strict_candidates.append({'score': float(score), 'song_id': int(sid), 'song_name': id_to_name(sid), 'stats': stats})
        diagnostics['results']['strict'] = {
            'winner_id': None,
            'top_candidates': strict_candidates,
            'info': {'ratio': info.get('ratio'), 'num_windows': info.get('num_windows')}
        }

        if use_aggressive:
            print('\nRunning aggressive fallback pass (more hashes, finer quantization)...')
            window_results_aggr = []
            for i, w in enumerate(windows, 1):
                print(f"\nAggressive processing window {i}/{len(windows)}...")
                peaks = extract_peaks_adaptive(w, max_peaks=max_peaks)
                hashes = generate_hashes_optimized(
                    peaks,
                    delay_min=5, delay_max=300, delta_freq=400,
                    fan_out=6, freq_fuzz=1, time_fuzz=1,
                )
                res = query_db_hashes(hashes)
                print(f"  Agg candidate songs this window: {len(res)}")
                window_results_aggr.append(res)

            agg2 = aggregate_window_results(window_results_aggr)
            winner2_sid, info2 = decide_winner(agg2, len(windows), audio_for_tiebreak=audio)  # NEW: pass audio

            if winner2_sid:
                winner2_name = id_to_name(winner2_sid)
                print('\nDETECTED (aggressive):', ascii_safe(winner2_name), info2)
                diagnostics['results']['aggressive'] = {
                    'winner_id': winner2_sid,
                    'winner_name': winner2_name,
                    'info': info2,
                }
            else:
                print('\nNo confident match after aggressive pass. Diagnostics:', info2)
                aggr_candidates = []
                for score, sid, stats in info2.get('top_candidates', []):
                    aggr_candidates.append({'score': float(score), 'song_id': int(sid), 'song_name': id_to_name(sid), 'stats': stats})
                diagnostics['results']['aggressive'] = {
                    'winner_id': None,
                    'top_candidates': aggr_candidates,
                    'info': {'ratio': info2.get('ratio'), 'num_windows': info2.get('num_windows')}
                }
        else:
            print('\nNo aggressive fallback (strict mode).')

    for i, res in enumerate(window_results, 1):
        diagnostics['windows_summary'].append({'window_index': i, 'candidate_count': len(res)})

    try:
        import json
        with open(args.log, 'w', encoding='utf-8') as jf:
            json.dump(diagnostics, jf, ensure_ascii=False, indent=2)
        print(f"\nDiagnostics written to {args.log}")
    except Exception as e:
        print('Failed to write diagnostics:', e)