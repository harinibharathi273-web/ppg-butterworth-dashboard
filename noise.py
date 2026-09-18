"""
noise.py
========
Pipeline step: add SIMULATED sensor noise to the clean (normalized) PPG,
so that the Butterworth filter has something meaningful to remove.

Noise types (from the project's problem statement: EMI, motion artifacts,
quantization noise):

1. High-frequency / EMI  - sinusoidal interference tones (15-50 Hz; the 50 Hz
                           mains tone is used only when fs is high enough)
2. Gaussian (white)      - random broadband sensor/electronic noise
3. Baseline wander       - slow drift (< 0.5 Hz), e.g. breathing / pressure changes
4. Motion artifacts      - short bursts at 1-3 Hz, i.e. INSIDE the heart-rate band
5. Quantization          - rounding to a limited number of ADC levels

Note for the viva: a LOW-PASS filter removes (1) and most of (2) and (5),
but NOT (3) or (4), because those lie at or below the heart-rate band.
"""

import numpy as np

# Noise level -> scaling factor applied to every noise component
NOISE_LEVELS = {"None": 0.0, "Low": 0.5, "Medium": 1.0, "High": 2.0}

# (frequency in Hz, amplitude at scale 1.0)
EMI_TONES = ((15.0, 0.04), (25.0, 0.05), (35.0, 0.04), (50.0, 0.06))
GAUSSIAN_STD = 0.05
MOTION_AMPLITUDE = 0.30


def _high_frequency_noise(time, fs, scale, rng):
    """Sum of interference tones that lie safely below the Nyquist frequency."""
    nyquist = fs / 2
    tones = [(f, a) for f, a in EMI_TONES if f < 0.9 * nyquist]
    if not tones:  # very low fs: use one tone just below Nyquist
        tones = [(0.4 * fs, 0.05)]
    hf = np.zeros_like(time)
    for freq, amp in tones:
        phase = rng.uniform(0, 2 * np.pi)
        hf += scale * amp * np.sin(2 * np.pi * freq * time + phase)
    return hf


def _baseline_wander(time, scale, rng):
    """Slow drift made of two very-low-frequency sinusoids."""
    p1, p2 = rng.uniform(0, 2 * np.pi, size=2)
    return scale * (
        0.10 * np.sin(2 * np.pi * 0.20 * time + p1)
        + 0.06 * np.sin(2 * np.pi * 0.05 * time + p2)
    )


def _motion_artifacts(time, scale, rng):
    """A few short oscillating bursts (roughly one every 8 s)."""
    duration = float(time[-1]) if time.size else 0.0
    n_events = max(1, int(round(duration / 8.0)))
    artifact = np.zeros_like(time)
    for _ in range(n_events):
        center = rng.uniform(0.15 * duration, 0.85 * duration)
        width = rng.uniform(0.3, 0.7)       # seconds
        freq = rng.uniform(1.0, 3.0)        # Hz - overlaps the heart-rate band
        amp = MOTION_AMPLITUDE * scale * rng.choice([-1.0, 1.0])
        envelope = np.exp(-0.5 * ((time - center) / width) ** 2)
        artifact += amp * envelope * np.cos(2 * np.pi * freq * (time - center))
    return artifact


def _quantize(x, bits):
    """Round the signal to 2**bits evenly spaced levels over its own range."""
    levels = 2 ** int(bits) - 1
    lo, hi = x.min(), x.max()
    if hi - lo == 0:
        return x.copy()
    return np.round((x - lo) / (hi - lo) * levels) / levels * (hi - lo) + lo


def add_noise(
    clean,
    time,
    fs,
    level="Medium",
    high_frequency=True,
    gaussian=True,
    baseline_wander=False,
    motion_artifacts=False,
    quantization=False,
    quantization_bits=6,
    seed=42,
):
    """
    Add the selected noise components to the clean signal.

    Returns (noisy_signal, components) where components is a dict
    {name: noise_array} so that each contribution can be plotted.
    A fixed seed makes results reproducible for the demo.
    """
    if level not in NOISE_LEVELS:
        raise ValueError(f"Unknown noise level '{level}'. Choose from {list(NOISE_LEVELS)}.")

    clean = np.asarray(clean, dtype=float)
    time = np.asarray(time, dtype=float)
    scale = NOISE_LEVELS[level]
    components = {}

    if scale == 0:  # "None" -> no noise at all
        return clean.copy(), components

    rng = np.random.default_rng(seed)

    if high_frequency:
        components["High-frequency / EMI"] = _high_frequency_noise(time, fs, scale, rng)
    if gaussian:
        components["Gaussian (white)"] = rng.normal(0.0, GAUSSIAN_STD * scale, clean.size)
    if baseline_wander:
        components["Baseline wander"] = _baseline_wander(time, scale, rng)
    if motion_artifacts:
        components["Motion artifacts"] = _motion_artifacts(time, scale, rng)

    noisy = clean + sum(components.values()) if components else clean.copy()

    if quantization:
        quantized = _quantize(noisy, quantization_bits)
        components[f"Quantization ({quantization_bits}-bit)"] = quantized - noisy
        noisy = quantized

    return noisy, components
