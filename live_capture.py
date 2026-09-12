import sounddevice as sd
import numpy as np
import scipy.signal


def capture_audio(duration=10, hardware_sr=44100, target_sr=22050, target_rms=0.12):
    """
    Capture live audio, stabilize level, and resample for fingerprinting.
    """
    print("--- INITIALIZING MICROPHONE ---")
    print(f"Recording for {duration} seconds...")

    # Probe default input device and pick a safe channel count.
    try:
        dev_info = sd.query_devices(None, 'input')
    except Exception:
        dev_info = None

    if dev_info is None:
        # Try sd.default.device fallback
        try:
            in_dev = sd.default.device[0]
            dev_info = sd.query_devices(in_dev)
        except Exception:
            dev_info = None

    max_ch = dev_info['max_input_channels'] if dev_info and 'max_input_channels' in dev_info else 1
    if max_ch < 1:
        raise RuntimeError('No input audio device available. Please check microphone or drivers.')

    channels = 2 if max_ch >= 2 else 1
    print(f"Recording using {channels} input channel(s) (device max: {max_ch})")

    try:
        raw_audio = sd.rec(
            int(duration * hardware_sr),
            samplerate=hardware_sr,
            channels=channels,
            blocking=True,
        )
    except Exception as e:
        # Fallback: try single-channel capture
        if channels == 2:
            print("Stereo capture failed, retrying as mono...")
            raw_audio = sd.rec(
                int(duration * hardware_sr),
                samplerate=hardware_sr,
                channels=1,
                blocking=True,
            )
        else:
            raise

    print("Recording complete. Processing stream...")

    # Use a single channel to avoid stereo phase-cancellation issues.
    if raw_audio.ndim == 2 and raw_audio.shape[1] > 1:
        mono_audio = raw_audio[:, 0]
    else:
        mono_audio = raw_audio.flatten()

    # Remove DC bias.
    mono_audio = mono_audio - np.mean(mono_audio)

    # NEW: Apply a pre-emphasis filter to boost transients and suppress low-end room mud
    # y[t] = x[t] - 0.95 * x[t-1]
    mono_audio = scipy.signal.lfilter([1.0, -0.95], [1.0], mono_audio)

    # RMS normalization (automatic gain).
    rms = np.sqrt(np.mean(mono_audio ** 2))
    if rms > 1e-8:
        mono_audio = mono_audio * (target_rms / rms)

    # Prevent hard clipping.
    mono_audio = np.clip(mono_audio, -0.95, 0.95)

    # Resample to match database sample rate.
    num_target_samples = int(len(mono_audio) * target_sr / hardware_sr)
    processed_audio = scipy.signal.resample(mono_audio, num_target_samples).astype(np.float32)

    final_rms = np.sqrt(np.mean(processed_audio ** 2))
    print(f"Stream standardized: Mono, {target_sr} Hz, {len(processed_audio)} samples.")
    print(f"Final RMS: {final_rms:.6f} | Peak: {np.max(np.abs(processed_audio)):.6f}")

    return processed_audio, target_sr


if __name__ == "__main__":
    test_audio, sr = capture_audio(duration=5)
    print(f"\nArray Type: {type(test_audio)}")
    print(f"Max Amplitude: {np.max(test_audio):.4f}")