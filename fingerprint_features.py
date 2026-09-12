import librosa
import numpy as np
from scipy.ndimage import maximum_filter

N_FFT = 2048
HOP_LENGTH = 512


def _extract_peaks_from_pcen(
    pcen_spec,
    neighborhood_size=35,
    min_percentile=70.0,
    max_percentile=92.0,
    percentile_step=4.0,
    min_peak_density=8.0,
    freq_bin_min=10,
    freq_bin_max=450,
):
    """Extract constellation peaks using percentile thresholding with density backoff."""
    freq_count, frame_count = pcen_spec.shape
    fmin = max(0, int(freq_bin_min))
    fmax = min(freq_count - 1, int(freq_bin_max))
    band = pcen_spec[fmin : fmax + 1, :]

    local_max = maximum_filter(pcen_spec, size=neighborhood_size)
    target_min_peaks = max(20, int(min_peak_density * (frame_count * HOP_LENGTH / 22050.0)))

    chosen_percentile = min(max_percentile, 99.0)
    chosen_threshold = np.percentile(band, chosen_percentile)
    chosen_peaks = []

    percentile = min(max_percentile, 99.0)
    while percentile >= min_percentile:
        threshold = np.percentile(band, percentile)
        peak_mask = (pcen_spec == local_max) & (pcen_spec >= threshold)
        freq_bins, time_frames = np.where(peak_mask)
        valid = (freq_bins >= fmin) & (freq_bins <= fmax)
        peaks = list(zip(time_frames[valid], freq_bins[valid]))

        chosen_percentile = percentile
        chosen_threshold = threshold
        chosen_peaks = peaks

        if len(peaks) >= target_min_peaks:
            break
        percentile -= percentile_step

    stats = {
        "percentile": float(chosen_percentile),
        "threshold": float(chosen_threshold),
        "target_min_peaks": int(target_min_peaks),
    }
    return chosen_peaks, stats


def extract_peaks_from_array(
    audio_array,
    sr=22050,
    neighborhood_size=35,
    min_percentile=70.0,
    max_percentile=92.0,
    percentile_step=4.0,
    min_peak_density=8.0,
    freq_bin_min=10,
    freq_bin_max=450,
):
    """PCEN-based peak extraction robust to loudness and soft vocals."""
    stft_mag = np.abs(librosa.stft(audio_array, n_fft=N_FFT, hop_length=HOP_LENGTH))
    power_spec = stft_mag ** 2
    pcen_spec = librosa.pcen(power_spec, sr=sr, hop_length=HOP_LENGTH)

    peaks, stats = _extract_peaks_from_pcen(
        pcen_spec,
        neighborhood_size=neighborhood_size,
        min_percentile=min_percentile,
        max_percentile=max_percentile,
        percentile_step=percentile_step,
        min_peak_density=min_peak_density,
        freq_bin_min=freq_bin_min,
        freq_bin_max=freq_bin_max,
    )
    stats["shape"] = pcen_spec.shape
    return peaks, stats


def extract_peaks_from_file(file_path, **kwargs):
    """Load file at 22050 mono and run PCEN + percentile peak extraction."""
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    return extract_peaks_from_array(y, sr=sr, **kwargs)
