# gui/worker.py
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from live_capture import capture_audio
from adaptive_live_matcher import extract_peaks_adaptive, generate_hashes_optimized
import multi_window_query
from multi_window_query import (
    split_windows, query_db_hashes, aggregate_window_results,
    decide_winner, id_to_name
)

# Optional humming imports
try:
    from humming_matcher import identify_humming_live
    HUMMING_AVAILABLE = True
except ImportError:
    HUMMING_AVAILABLE = False


class RecognitionWorker(QThread):
    """
    Handles audio capture and heavy DSP math on a separate CPU thread
    so the main GUI remains completely fluid and responsive.
    """
    status_update = pyqtSignal(str)
    finished_success = pyqtSignal(str, list, str)
    finished_fail = pyqtSignal(str)
    wave_active = pyqtSignal(bool)

    def __init__(self, duration, mode):
        super().__init__()
        self.duration = duration
        self.mode = mode

    def run(self):
        try:
            self.status_update.emit(
                f"Recording for {self.duration} seconds. Please play/hum the song..."
            )
            self.wave_active.emit(True)

            # Capture Audio
            audio, sr = capture_audio(duration=self.duration)
            self.wave_active.emit(False)
            self.status_update.emit("Processing digital signal...")

            # --- HUMMING MODE ---
            if self.mode == "humming":
                if not HUMMING_AVAILABLE:
                    self.finished_fail.emit("Humming module not found.")
                    return

                winner, top, info = identify_humming_live(
                    audio,
                    sr=sr,
                    db_path="database/song_index.db",
                    top_k=5,
                    query_type="humming",
                )

                # Extract DTW Leaderboard
                leaderboard = []
                if top:
                    for t in top:
                        leaderboard.append((t["song_name"], t["distance"]))

                if not winner:
                    if leaderboard:
                        self.finished_success.emit(
                            "Uncertain / Weak Match",
                            leaderboard,
                            f"Threshold not met ({info.get('reason', '')}).",
                        )
                    else:
                        self.finished_fail.emit(
                            f"No confident humming match. Info: {info.get('reason', '')}"
                        )
                    return

                self.finished_success.emit(winner, leaderboard, "")
                return

            # --- ORIGINAL / COVER MODE (Multi-Window) ---
            win_s = 8.0 if self.duration >= 10 else 4.0
            step_s = 4.0 if self.duration >= 10 else 2.0
            windows = split_windows(audio, sr, win_s, step_s)

            # Strict Pass
            self.status_update.emit("Running structural alignment...")

            strict_results = []
            for i, w in enumerate(windows, 1):
                peaks = extract_peaks_adaptive(w, max_peaks=2000)
                hashes = generate_hashes_optimized(peaks)
                strict_results.append(query_db_hashes(hashes))

            agg = aggregate_window_results(strict_results)
            multi_window_query.MIN_TOP12_RATIO = 1.5
            winner_sid, info = decide_winner(agg, len(windows))

            best_agg = agg

            # Aggressive Fallback
            if not winner_sid:
                self.status_update.emit("Running aggressive fuzzy fallback...")
                aggr_results = []
                for i, w in enumerate(windows, 1):
                    peaks = extract_peaks_adaptive(w, max_peaks=1200)
                    hashes = generate_hashes_optimized(
                        peaks,
                        delay_min=5,
                        delay_max=300,
                        delta_freq=400,
                        fan_out=6,
                        freq_fuzz=1,
                        time_fuzz=1,
                    )
                    aggr_results.append(query_db_hashes(hashes))

                agg2 = aggregate_window_results(aggr_results)
                best_agg = agg2

                multi_window_query.MIN_TOP12_RATIO = 1.05
                winner_sid2, info2 = decide_winner(agg2, len(windows))

                if winner_sid2:
                    winner_sid = winner_sid2
                elif info2.get("top_candidates") and len(info2["top_candidates"]) >= 2:
                    top_cands = info2["top_candidates"]
                    top1_w, top1_sid, top1_v = top_cands[0]
                    top2_w = top_cands[1][0]
                    ratio = (top1_w / top2_w) if top2_w > 0 else float("inf")
                    if top1_v.get("raw", 0) >= 10 and ratio >= 1.05:
                        winner_sid = top1_sid

            # Extract the JSON Leaderboard dynamically from the best aggregation
            leaderboard = []
            if best_agg:
                items = sorted(
                    [(v["weighted"], sid, v) for sid, v in best_agg.items()],
                    reverse=True,
                )
                for score, sid, stats in items[:5]:
                    leaderboard.append((id_to_name(sid), score))

            if not winner_sid:
                if leaderboard:
                    self.finished_success.emit(
                        "Uncertain / Multiple Candidates",
                        leaderboard,
                        "Model Confused. Displaying search leaderboard.",
                    )
                else:
                    self.finished_fail.emit(
                        "No confident multi-window match found and no candidate data available."
                    )
                return

            winner_name = id_to_name(winner_sid)
            self.finished_success.emit(winner_name, leaderboard, "")

        except Exception as e:
            self.wave_active.emit(False)
            self.finished_fail.emit(f"System Error: {str(e)}")