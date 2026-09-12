# **Real-Time Audio Pattern Recognition System for Music Information Retrieval (MIR)**

An end-to-end, multi-modal audio identification engine built with Python, digital signal processing (DSP) algorithms, and a multithreaded PyQt5 user interface. The system supports live microphone queries across four operational modes: original audio identification, cover song recognition, instrumental matching, and query-by-humming.

---

## **Key Features**

1. **Multi-Modal Recognition Engine**:
1. **Original Track Mode**: Uses Perceptual Channel Energy Normalization (PCEN), spectral peak extraction, combinatorial hashing, and Inverse Document Frequency (IDF) weighting.
2. **Cover Song Mode**: Utilizes 12-dimensional normalized Harmonic Chromagrams with Subsequence Dynamic Time Warping (DTW).
3. **Instrumental Mode**: Evaluates harmonic progression using pitch class profiles.
4. **Query-by-Humming (QBH) Mode**: Extracts monophonic fundamental frequency (F0) contours using the Probabilistic YIN (pYIN) algorithm.


2. **Ambiguity Resolution**: Secondary Mel-Frequency Cepstral Coefficient (MFCC) timbral tie-breaker for reverberant environments.
3. **Asynchronous Multithreading**: Prevents UI freeze by decoupling DSP calculations into background `QThread` workers.
4. **Interactive Desktop GUI**: Custom hardware-accelerated waveform visualizer, dynamic candidate cards, and direct YouTube redirection.

---

## **Project Directory Structure**

```text
Music_Retrieval_System/
├── database/
│   └── song_index.db            # SQLite index storing song metadata and fingerprints
├── gui/
│   ├── worker.py                # Asynchronous worker thread routing DSP execution
│   └── gui_modern.py            # PyQt5 desktop application interface & visualizer
├── live_capture.py              # Microphone ingestion and dynamic peak normalization
├── adaptive_live_matcher.py     # PCEN filtering, peak extraction, hashing & MFCC tie-breaker
├── multi_window_query.py        # Multi-window slicing, IDF scoring & consensus decision logic
└── main.py                      # Application entry point

```

---

## **Prerequisites & Environment Setup**

1. **Python Version**: Python 3.11+
2. **Dependencies**: Install the required Python packages using:

```bash
pip install -r requirements.txt

```

---

## **Database Setup & Indexing**

1. Ensure the `database/` directory exists.
2. The system utilizes SQLite (`database/song_index.db`) with an inverted index structure (`hash_str`, `song_id`, `offset`).
3. Execute your offline database ingestion script to populate reference tracks prior to launching queries.

---

## **How to Run**

Launch the graphical user interface from the project root directory:

```bash
python main.py

```

### **Operational Workflow**

1. Select query mode (**Original**, **Cover**, **Instrumental**, or **Humming**).
2. Adjust recording duration (e.g., 5s – 10s).
3. Click **Record / Query** to capture live audio from your microphone.
4. View real-time waveform visualization, top matched results, and confidence scores.
5. Click on candidate cards to open and play the track via YouTube.

---

## **Performance Benchmarks**

1. **Original Clean Audio**: 98.0% Top-1 Accuracy (0.42s mean latency).
2. **Original Noisy Hall Audio**: 92.0% Top-1 Accuracy (0.58s mean latency).
3. **Cover Song Recognition**: 76.7% Top-1 Accuracy.
4. **Query-by-Humming**: 67.5% Top-1 Accuracy.
