"""
butterworth.py
==============
Pipeline step: DIGITAL Butterworth Low-Pass Filter ("Butterworth Core").

Magnitude response of an n-th order Butterworth low-pass filter:

    |H(jw)| = 1 / sqrt( 1 + (w / wc)^(2n) )

- Maximally flat passband (no ripple, unlike Chebyshev)
- -3 dB at the cutoff frequency wc
- Roll-off of about 20*n dB per decade above the cutoff

Implementation notes
--------------------
* scipy.signal.butter designs the filter. We ask for "second-order sections"
  (output="sos"): the filter is realised as a cascade of 2nd-order stages,
  which is numerically stable and mirrors the cascaded 2nd-order (Sallen-Key)
  stages in the project's block diagram.
* Zero-phase filtering (sosfiltfilt) runs the filter forward and then backward.
  The phase shifts cancel, so pulse peaks are NOT shifted in time. Side effect:
  the magnitude response is applied twice -> |H|^2 (-6 dB at the cutoff,
  effective order 2n).
"""

import numpy as np

try:
    from scipy import signal
    USING_SCIPY = True
except Exception:  # e.g. SciPy DLLs blocked by Windows Smart App Control
    from . import dsp_numpy as signal
    USING_SCIPY = False

# SciPy >= 1.15 calls it freqz_sos, older versions sosfreqz
_freqz_sos = getattr(signal, "freqz_sos", None) or getattr(signal, "sosfreqz")

DSP_BACKEND = "SciPy" if USING_SCIPY else "Built-in NumPy implementation (SciPy unavailable)"

MIN_ORDER = 1
MAX_ORDER = 10


def validate_filter_parameters(order, cutoff_hz, fs):
    """Raise ValueError with a readable message if the parameters are invalid."""
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")
    if not (MIN_ORDER <= int(order) <= MAX_ORDER):
        raise ValueError(f"Filter order must be between {MIN_ORDER} and {MAX_ORDER}.")
    nyquist = fs / 2
    if cutoff_hz <= 0:
        raise ValueError("Cutoff frequency must be greater than 0 Hz.")
    if cutoff_hz >= nyquist:
        raise ValueError(
            f"Cutoff frequency ({cutoff_hz:g} Hz) must be below the Nyquist "
            f"frequency fs/2 = {nyquist:g} Hz (Nyquist criterion)."
        )


def design_butterworth_lowpass(order, cutoff_hz, fs):
    """Design the digital Butterworth low-pass filter; returns second-order sections."""
    validate_filter_parameters(order, cutoff_hz, fs)
    return signal.butter(int(order), cutoff_hz, btype="lowpass", fs=fs, output="sos")


def minimum_length_for_zero_phase(sos):
    """Minimum number of samples sosfiltfilt needs (its default padding + 1)."""
    n_sections = len(sos)
    trailing_zeros = min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())
    padlen = 3 * (2 * n_sections + 1 - trailing_zeros)
    return padlen + 1


def apply_butterworth(x, sos, zero_phase=True):
    """
    Filter the signal.

    zero_phase=True  -> sosfiltfilt (forward + backward, no time shift)
    zero_phase=False -> sosfilt (ordinary causal filter, peaks are delayed;
                        this is what a real-time microcontroller would do)
    """
    x = np.asarray(x, dtype=float)
    if zero_phase:
        needed = minimum_length_for_zero_phase(sos)
        if x.size < needed:
            raise ValueError(
                f"Signal too short for zero-phase filtering: {x.size} samples, "
                f"need at least {needed}. Increase the duration or sampling frequency."
            )
        return signal.sosfiltfilt(sos, x)

    # Causal filter. Start the filter's internal state at the first sample
    # value so there is no artificial start-up jump from 0.
    zi = signal.sosfilt_zi(sos) * x[0]
    y, _ = signal.sosfilt(sos, x, zi=zi)
    return y


def frequency_response(sos, fs, n_points=4096):
    """
    Returns (frequencies_hz, magnitude_db_single_pass, magnitude_db_zero_phase).
    The zero-phase curve is |H|^2 (the filter is applied twice).
    """
    freqs, h = _freqz_sos(sos, worN=n_points, fs=fs)
    mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
    return freqs, mag_db, 2 * mag_db


def gain_db_at(sos, fs, frequency_hz):
    """Single-pass gain (dB) of the filter at one frequency."""
    _, h = _freqz_sos(sos, worN=[frequency_hz], fs=fs)
    return float(20 * np.log10(max(abs(h[0]), 1e-12)))
