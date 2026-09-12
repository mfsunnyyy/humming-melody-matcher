import os
import csv
import sqlite3
import math
import time

import librosa

from multi_window_query import split_windows, process_window, decide_winner, aggregate_window_results

# Config
BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "database", "song_index.db")
RAW_AUDIO_DIR = os.path.join(BASE_DIR, "database", "raw_audio")
SR = 22050
WIN_S = 12.0
STEP_S = 6.0
OUT_CSV = "multi_window_eval_results.csv"


def map_filename_to_song_ids(basename, db_path=DB_PATH):
    # Try to find songs whose name contains the basename (case-insensitive)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    pattern = f"%{basename}%"
    cur.execute("SELECT song_id, song_name FROM songs WHERE lower(song_name) LIKE lower(?)", (pattern,))
    rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows], [r[1] for r in rows]


def evaluate_file(path):
    try:
        audio, sr = librosa.load(path, sr=SR, mono=True)
    except Exception as e:
        return {"file": path, "error": str(e)}
    windows = split_windows(audio, sr, WIN_S, STEP_S)
    window_results = []
    for w in windows:
        res = process_window(w, sr)
        window_results.append(res)
    agg = aggregate_window_results(window_results)
    winner_sid, info = decide_winner(agg, len(windows))
    # get top3
    top3 = []
    if isinstance(info, dict) and info.get("top_candidates"):
        for w, sid, v in info["top_candidates"]:
            top3.append((sid, w))
    else:
        # if winner exists, build sorted top3 from agg
        items = sorted([(v["weighted"], sid, v) for sid, v in agg.items()], reverse=True)
        for it in items[:3]:
            top3.append((it[1], it[0]))
    basename = os.path.splitext(os.path.basename(path))[0]
    mapped_ids, mapped_names = map_filename_to_song_ids(basename)
    top1_hit = None
    top3_hit = None
    if mapped_ids:
        if winner_sid and winner_sid in mapped_ids:
            top1_hit = True
        else:
            top1_hit = False
        if any(sid in mapped_ids for sid, _ in top3):
            top3_hit = True
        else:
            top3_hit = False
    else:
        top1_hit = None
        top3_hit = None
    return {
        "file": path,
        "basename": basename,
        "winner_sid": winner_sid,
        "winner_name": None if not winner_sid else None,
        "mapped_ids": mapped_ids,
        "mapped_names": mapped_names,
        "top3": top3,
        "top1_hit": top1_hit,
        "top3_hit": top3_hit,
        "info": info,
    }


def main():
    entries = []
    files = [os.path.join(RAW_AUDIO_DIR, f) for f in os.listdir(RAW_AUDIO_DIR) if f.lower().endswith('.wav')]
    print(f"Found {len(files)} wav files to evaluate")
    for i, f in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Evaluating: {os.path.basename(f)}")
        r = evaluate_file(f)
        entries.append(r)
        print("  winner:", r.get('winner_sid'), "mapped_ids:", r.get('mapped_ids'))
    # Write CSV
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file", "basename", "winner_sid", "mapped_ids", "mapped_names", "top1_hit", "top3_hit", "info_summary"])
        for e in entries:
            info = e.get('info')
            info_summary = ''
            if isinstance(info, dict) and 'ratio' in info:
                info_summary = f"ratio={info['ratio']},num_windows={info.get('num_windows')}"
            writer.writerow([e.get('file'), e.get('basename'), e.get('winner_sid'), ';'.join(map(str, e.get('mapped_ids') or [])), ';'.join(e.get('mapped_names') or []), e.get('top1_hit'), e.get('top3_hit'), info_summary])
    print(f"\nEvaluation complete. Results saved to {OUT_CSV}")

if __name__ == '__main__':
    main()
