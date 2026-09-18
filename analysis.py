"""
analysis.py
===========
Pipeline steps after filtering: peak detection, heart-rate estimation,
beat-detection accuracy, SNR and spectrum.
"""

import numpy as np

try:
    from scipy import signal
except Exception:  # e.g. SciPy DLLs blocked by Windows Smart App Control
    from . import dsp_numpy as signal

MAX_DETECTABLE_BPM = 220


# --------------------------------------------------------------------------
# Peak detection and heart rate
# --------------------------------------------------------------------------
def detect_peaks(x, fs, max_bpm=MAX_DETECTABLE_BPM, prominence_fraction=0.35):
    """
    Find systolic peaks with scipy.signal.find_peaks.

    Two rules decide what counts as a heartbeat:
    * distance   : peaks must be at least 60/max_bpm seconds apart
                   (at 220 BPM a beat lasts 0.27 s)
    * prominence : a peak must stand out by at least 35 % of the signal's
                   robust range (5th-95th percentile). This rejects the smaller
                   diastolic bump and leftover noise wiggles.
    Returns sample indices of the detected peaks.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return np.array([], dtype=int)
    lo, hi = np.percentile(x, [5, 95])
    signal_range = hi - lo
    if signal_range <= 0:
        return np.array([], dtype=int)
    min_distance = max(1, int(round(fs * 60.0 / max_bpm)))
    peaks, _ = signal.find_peaks(
        x, distance=min_distance, prominence=prominence_fraction * signal_range
    )
    return peaks


def heart_rate_metrics(peak_times):
    """
    Heart-rate statistics from peak times (seconds).

    IBI (inter-beat interval) = time between consecutive peaks
    instantaneous BPM         = 60 / IBI
    estimated BPM             = 60 / mean(IBI)   (overall heart rate)
    mean BPM                  = mean of the instantaneous BPM values
    Returns None if fewer than 2 peaks were found.
    """
    peak_times = np.asarray(peak_times, dtype=float)
    if peak_times.size < 2:
        return None
    ibi = np.diff(peak_times)
    inst_bpm = 60.0 / ibi
    return {
        "num_peaks": int(peak_times.size),
        "ibi_s": ibi,
        "mean_ibi_s": float(ibi.mean()),
        "estimated_bpm": float(60.0 / ibi.mean()),
        "mean_bpm": float(inst_bpm.mean()),
        "min_bpm": float(inst_bpm.min()),
        "max_bpm": float(inst_bpm.max()),
        "inst_bpm": inst_bpm,
        # each instantaneous BPM belongs to the midpoint between two peaks
        "inst_bpm_times": (peak_times[:-1] + peak_times[1:]) / 2,
    }


def beat_detection_accuracy(detected_times, true_times):
    """
    Compare detected peaks with the exact synthetic peak times.

    A detected peak counts as correct if it lies within a tolerance of a true
    peak (0.15 s, or less at very high heart rates). Each true peak can be
    matched only once.
      sensitivity = correct / true beats       (how many beats were found)
      precision   = correct / detected peaks   (how many detections were real)
    """
    detected_times = np.asarray(detected_times, dtype=float)
    true_times = np.asarray(true_times, dtype=float)
    if true_times.size == 0:
        return {"correct": 0, "sensitivity": np.nan, "precision": np.nan}

    tolerance = 0.15
    if true_times.size > 1:
        tolerance = min(tolerance, 0.4 * np.diff(true_times).min())

    used = np.zeros(detected_times.size, dtype=bool)
    correct = 0
    for t in true_times:
        if detected_times.size == 0:
            break
        distance = np.abs(detected_times - t)
        distance[used] = np.inf
        j = int(np.argmin(distance))
        if distance[j] <= tolerance:
            used[j] = True
            correct += 1

    return {
        "correct": correct,
        "sensitivity": 100.0 * correct / true_times.size,
        "precision": 100.0 * correct / detected_times.size if detected_times.size else np.nan,
    }


# --------------------------------------------------------------------------
# Signal-to-noise ratio
# --------------------------------------------------------------------------
def snr_db(reference, observed):
    """
    SNR in dB, using the clean synthetic PPG as the reference:

        signal power = mean( (reference - mean(reference))^2 )   (AC power)
        noise power  = mean( (observed - reference)^2 )           (error power)
        SNR (dB)     = 10 * log10(signal power / noise power)

    The mean is removed from the reference so that the constant DC offset of
    the 0..1 normalized signal does not inflate the "signal" power.

    Note: for the filtered signal, anything that differs from the clean
    reference counts as noise - including slight smoothing of the pulse by
    the filter itself. So this is an honest, slightly conservative measure.
    """
    reference = np.asarray(reference, dtype=float)
    observed = np.asarray(observed, dtype=float)
    signal_power = np.mean((reference - reference.mean()) ** 2)
    noise_power = np.mean((observed - reference) ** 2)
    if signal_power == 0:
        raise ValueError("Reference signal is constant; SNR is undefined.")
    if noise_power == 0:
        return float("inf")
    return float(10 * np.log10(signal_power / noise_power))


# --------------------------------------------------------------------------
# Spectrum
# --------------------------------------------------------------------------
def amplitude_spectrum(x, fs):
    """One-sided amplitude spectrum (Hann window, mean removed)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    window = np.hanning(x.size)
    spectrum = np.abs(np.fft.rfft(x * window)) * 2 / window.sum()
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    return freqs, spectrum
