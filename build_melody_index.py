import os
import sqlite3
import librosa

from humming_matcher import (
    extract_melody_features_from_file,
    setup_melody_table,
    upsert_song_melody,
)

DB_PATH = "database/song_index.db"
AUDIO_DIR = "database/raw_audio"

def main():
    conn = sqlite3.connect(DB_PATH)
    setup_melody_table(conn)
    cur = conn.cursor()

    wavs = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(".wav")]
    total = len(wavs)
    ok = 0

    for i, fn in enumerate(wavs, 1):
        song_name = fn[:-4]
        row = cur.execute("SELECT song_id FROM songs WHERE song_name = ?", (song_name,)).fetchone()
        if not row:
            print(f"[{i}/{total}] SKIP song not in songs table: {song_name}")
            continue
        sid = int(row[0])

        path = os.path.join(AUDIO_DIR, fn)
        feat = extract_melody_features_from_file(path, sr=22050)
        if not feat.get("ok"):
            print(f"[{i}/{total}] FAIL {song_name} -> {feat.get('reason')}")
            continue

        upsert_song_melody(conn, sid, feat["intervals"])
        ok += 1
        print(f"[{i}/{total}] OK {song_name} (len={len(feat['intervals'])})")

    conn.close()
    print(f"\nMelody indexing done: {ok}/{total}")

if __name__ == "__main__":
    main()