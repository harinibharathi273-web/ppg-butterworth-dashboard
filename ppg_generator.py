"""
ppg_generator.py
================
Pipeline step: BPM data  ->  synthetic PPG-like waveform  ->  normalization.

IMPORTANT (scientific limitation)
---------------------------------
A BPM value only tells us HOW OFTEN the heart beats. It carries no
information about the SHAPE of the real PPG pulse. So this module does NOT
recover the original PPG. It builds an artificial ("synthetic") PPG-like
pulse and places one pulse per heartbeat, so that the pulse spacing matches
the supplied BPM:

    beat frequency (Hz) = BPM / 60        e.g. 75 BPM -> 1.25 Hz
    beat interval  (s)  = 60 / BPM        e.g. 75 BPM -> 0.80 s
"""

import numpy as np
import pandas as pd

MIN_BPM = 30
MAX_BPM = 220

# ---- Pulse-shape parameters ---------------------------------------------
# Positions and widths are fractions of ONE beat period (phase 0 -> 1), so
# the pulse automatically stretches or shrinks with the heart rate.
SYSTOLIC_CENTER = 0.18       # position of the main (systolic) peak
SYSTOLIC_RISE_WIDTH = 0.06   # small width  -> rapid systolic upstroke
SYSTOLIC_FALL_WIDTH = 0.10   # larger width -> slower decay
DIASTOLIC_CENTER = 0.52      # reflected (diastolic) wave
DIASTOLIC_WIDTH = 0.10
DIASTOLIC_AMPLITUDE = 0.40   # relative to the systolic peak (= 1)
NO_NOTCH_FALL_WIDTH = 0.20   # decay width used when the notch is switched off

# ---- Baseline variation (slow breathing-related drift) --------------------
BASELINE_FREQUENCY_HZ = 0.25   # about 15 breaths per minute
BASELINE_AMPLITUDE = 0.05      # relative to the systolic peak

EXPECTED_CSV_FORMAT = (
    "Expected format: a header row with a 'BPM' column and (optionally) a "
    "'Time' column in seconds, for example:\n\n"
    "Time,BPM\n0,68\n1,70\n2,72"
)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate_bpm(bpm_values):
    """Return BPM values as a float array, or raise ValueError if any are invalid."""
    bpm = np.atleast_1d(np.asarray(bpm_values, dtype=float))
    if bpm.size == 0:
        raise ValueError("No BPM values were provided.")
    if not np.all(np.isfinite(bpm)):
        raise ValueError("BPM values must be finite numbers.")
    if bpm.min() < MIN_BPM:
        raise ValueError(
            f"BPM {bpm.min():g} is below the supported range ({MIN_BPM}-{MAX_BPM} BPM)."
        )
    if bpm.max() > MAX_BPM:
        raise ValueError(
            f"BPM {bpm.max():g} is above the supported range ({MIN_BPM}-{MAX_BPM} BPM)."
        )
    return bpm


# --------------------------------------------------------------------------
# Pulse morphology
# --------------------------------------------------------------------------
def _asymmetric_gaussian(phase, center, rise_width, fall_width):
    """Gaussian bump with a different width on its rising and falling side."""
    width = np.where(phase < center, rise_width, fall_width)
    return np.exp(-0.5 * ((phase - center) / width) ** 2)


def pulse_shape(phase, include_dicrotic_notch=True):
    """
    One PPG-like pulse as a function of phase (0 = start of beat, 1 = next beat).

    - Systolic peak : fast rise, slower fall (asymmetric Gaussian).
    - Diastolic wave: a second, smaller Gaussian bump. The dip between the two
      bumps is the dicrotic notch.
    """
    if not include_dicrotic_notch:
        return _asymmetric_gaussian(
            phase, SYSTOLIC_CENTER, SYSTOLIC_RISE_WIDTH, NO_NOTCH_FALL_WIDTH
        )
    systolic = _asymmetric_gaussian(
        phase, SYSTOLIC_CENTER, SYSTOLIC_RISE_WIDTH, SYSTOLIC_FALL_WIDTH
    )
    diastolic = DIASTOLIC_AMPLITUDE * np.exp(
        -0.5 * ((phase - DIASTOLIC_CENTER) / DIASTOLIC_WIDTH) ** 2
    )
    return systolic + diastolic


# --------------------------------------------------------------------------
# Beat timing
# --------------------------------------------------------------------------
def build_beat_schedule(bpm_times, bpm_values, duration_s):
    """
    Place heartbeats one after another.

    Each beat lasts 60 / BPM seconds, where BPM is the heart rate at the moment
    the beat starts (linearly interpolated between CSV rows). A changing BPM
    therefore gives a changing pulse-to-pulse spacing - it is NOT averaged.

    Returns (beat_onset_times, beat_periods) in seconds.
    """
    onsets, periods = [], []
    t = 0.0
    while t < duration_s:
        bpm_now = float(np.interp(t, bpm_times, bpm_values))
        period = 60.0 / bpm_now
        onsets.append(t)
        periods.append(period)
        t += period
    return np.array(onsets), np.array(periods)


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------
def normalize_signal(x):
    """Min-max normalization to 0..1:  x_norm = (x - min) / (max - min)."""
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    if hi - lo == 0:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


# --------------------------------------------------------------------------
# Main generator
# --------------------------------------------------------------------------
def generate_synthetic_ppg(
    bpm_times,
    bpm_values,
    duration_s,
    fs,
    include_dicrotic_notch=True,
    include_baseline_variation=True,
):
    """
    Build a synthetic PPG-like signal whose beat timing follows the BPM data.

    Parameters
    ----------
    bpm_times  : times (s) at which BPM was measured (one value for manual input)
    bpm_values : BPM measurements
    duration_s : length of the generated signal in seconds
    fs         : sampling frequency in Hz

    Returns a dict with:
        time             sample times (s)
        raw              signal before normalization
        ppg              normalized signal (0..1)  -> the "clean" reference
        beat_onsets      start time of every generated beat (s)
        beat_periods     length of every beat (s)
        true_peak_times  exact systolic-peak times (used to check detection)
    """
    bpm_times = np.atleast_1d(np.asarray(bpm_times, dtype=float))
    bpm_values = validate_bpm(bpm_values)
    if bpm_times.size != bpm_values.size:
        raise ValueError("Time and BPM arrays must have the same length.")
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")
    if duration_s <= 0:
        raise ValueError("Duration must be positive.")

    n_samples = int(round(duration_s * fs))
    time = np.arange(n_samples) / fs

    onsets, periods = build_beat_schedule(bpm_times, bpm_values, duration_s)

    raw = np.zeros(n_samples)
    for onset, period in zip(onsets, periods):
        # Evaluate only samples near this beat (slightly before/after so that
        # neighbouring pulses join smoothly).
        start = max(0, int(np.floor((onset - 0.3 * period) * fs)))
        stop = min(n_samples, int(np.ceil((onset + 1.3 * period) * fs)) + 1)
        phase = (time[start:stop] - onset) / period
        raw[start:stop] += pulse_shape(phase, include_dicrotic_notch)

    if include_baseline_variation:
        raw += BASELINE_AMPLITUDE * np.sin(2 * np.pi * BASELINE_FREQUENCY_HZ * time)

    true_peak_times = onsets + SYSTOLIC_CENTER * periods
    true_peak_times = true_peak_times[true_peak_times <= time[-1]]

    return {
        "time": time,
        "raw": raw,
        "ppg": normalize_signal(raw),
        "beat_onsets": onsets,
        "beat_periods": periods,
        "true_peak_times": true_peak_times,
    }


# --------------------------------------------------------------------------
# CSV input
# --------------------------------------------------------------------------
def _find_column(lookup, candidates):
    for name in candidates:
        if name in lookup:
            return lookup[name]
    return None


def load_bpm_csv(file_like):
    """
    Read BPM measurements from a CSV file.

    Returns (times_s, bpm_values, warnings). Times are shifted to start at 0.
    Raises ValueError with a readable explanation if the file is unusable.
    """
    warnings = []
    try:
        df = pd.read_csv(file_like)
    except Exception as exc:  # pandas raises several different error types
        raise ValueError(f"Could not read the file as a CSV ({exc}).\n\n{EXPECTED_CSV_FORMAT}")

    if df.empty or df.shape[1] == 0:
        raise ValueError(f"The CSV file has no data rows.\n\n{EXPECTED_CSV_FORMAT}")

    lookup = {str(col).strip().lower(): col for col in df.columns}
    bpm_col = _find_column(
        lookup, ("bpm", "heart rate", "heart_rate", "heartrate", "hr", "pulse", "pulse rate")
    )
    if bpm_col is None:
        raise ValueError(
            f"No BPM column found. Columns in your file: {list(df.columns)}.\n\n"
            f"{EXPECTED_CSV_FORMAT}"
        )
    time_col = _find_column(
        lookup, ("time", "time (s)", "time(s)", "time_s", "seconds", "sec", "t")
    )

    bpm = pd.to_numeric(df[bpm_col], errors="coerce")
    if time_col is None:
        times = pd.Series(np.arange(len(df), dtype=float), index=df.index)
        warnings.append("No 'Time' column found - assuming one BPM reading per second.")
    else:
        times = pd.to_numeric(df[time_col], errors="coerce")

    valid = bpm.notna() & times.notna()
    skipped = int((~valid).sum())
    if skipped:
        warnings.append(f"{skipped} row(s) with missing or non-numeric values were skipped.")

    data = pd.DataFrame({"Time": times[valid].astype(float), "BPM": bpm[valid].astype(float)})
    if data.empty:
        raise ValueError(f"No valid numeric BPM rows were found.\n\n{EXPECTED_CSV_FORMAT}")

    data = data.sort_values("Time").drop_duplicates("Time", keep="first").reset_index(drop=True)
    validate_bpm(data["BPM"].to_numpy())
    data["Time"] -= data["Time"].iloc[0]
    return data["Time"].to_numpy(), data["BPM"].to_numpy(), warnings


def duration_from_csv_times(times, min_duration_s=5.0):
    """
    Signal duration for CSV input: last timestamp + one sampling step
    (each row covers one interval). At least `min_duration_s` seconds so
    that there are enough beats to analyse.
    """
    times = np.asarray(times, dtype=float)
    step = float(np.median(np.diff(times))) if times.size > 1 else 1.0
    return max(float(times[-1]) + step, min_duration_s)
