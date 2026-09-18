"""
dsp_numpy.py
============
Pure-NumPy versions of the few SciPy signal-processing functions this project
uses. They are used automatically when SciPy cannot be loaded (for example
when Windows Smart App Control blocks SciPy's compiled DLL files).

They follow the same maths and behaviour as scipy.signal:

butter       - Butterworth design: analog prototype poles -> pre-warping ->
               bilinear transform -> second-order sections
sosfilt      - cascade of 2nd-order IIR sections (Direct Form II transposed)
sosfilt_zi   - steady-state initial conditions
sosfiltfilt  - zero-phase forward-backward filtering with odd-extension padding
freqz_sos    - frequency response of the cascade
find_peaks   - local maxima + minimum distance + prominence rules
"""

import numpy as np


# --------------------------------------------------------------------------
# Butterworth design
# --------------------------------------------------------------------------
def butter(N, Wn, btype="lowpass", fs=None, output="sos"):
    """Digital Butterworth LOW-PASS filter as second-order sections."""
    if btype not in ("low", "lowpass"):
        raise NotImplementedError("Only low-pass Butterworth filters are implemented.")
    if output != "sos":
        raise NotImplementedError("Only output='sos' is implemented.")
    if fs is None:
        raise ValueError("fs must be given.")
    N = int(N)
    nyquist = fs / 2.0
    if not 0 < Wn < nyquist:
        raise ValueError("Cutoff must be between 0 and fs/2.")

    # 1) Pre-warp the cutoff so the digital filter has -3 dB exactly at Wn
    fs2 = 2.0 * fs
    warped = fs2 * np.tan(np.pi * Wn / fs)

    # 2) Analog Butterworth poles: equally spaced on a circle in the left half-plane
    k = np.arange(1, N + 1)
    analog_poles = warped * np.exp(1j * np.pi * (2 * k + N - 1) / (2 * N))
    analog_gain = warped ** N

    # 3) Bilinear transform s -> z   (analog zeros at infinity map to z = -1)
    digital_poles = (fs2 + analog_poles) / (fs2 - analog_poles)
    digital_gain = analog_gain * np.real(1.0 / np.prod(fs2 - analog_poles))

    # 4) Group poles into 2nd-order sections (complex-conjugate pairs)
    sections = []
    complex_poles = digital_poles[np.imag(digital_poles) > 1e-12]
    real_poles = np.real(digital_poles[np.abs(np.imag(digital_poles)) <= 1e-12])
    for p in complex_poles:
        sections.append([1.0, 2.0, 1.0, 1.0, -2.0 * p.real, abs(p) ** 2])
    for p in real_poles:  # odd order -> one first-order section
        sections.append([1.0, 1.0, 0.0, 1.0, -p, 0.0])

    sos = np.array(sections, dtype=float)
    sos[0, :3] *= digital_gain  # overall gain goes into the first section
    return sos


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------
def _lfilter_zi(b, a):
    """Steady-state state of one section for a unit-step input."""
    companion = np.array([[-a[1], -a[2]], [1.0, 0.0]])
    i_minus_a = np.eye(2) - companion.T
    rhs = b[1:] - a[1:] * b[0]
    return np.linalg.solve(i_minus_a, rhs)


def sosfilt_zi(sos):
    zi = np.zeros((len(sos), 2))
    scale = 1.0
    for s, section in enumerate(sos):
        b, a = section[:3], section[3:]
        zi[s] = scale * _lfilter_zi(b, a)
        scale *= b.sum() / a.sum()
    return zi


def sosfilt(sos, x, zi=None):
    """Run the cascade (Direct Form II transposed). Returns y, or (y, zf) if zi given."""
    y = np.asarray(x, dtype=float).tolist()
    states = np.zeros((len(sos), 2)) if zi is None else np.array(zi, dtype=float)
    final_states = np.zeros_like(states)
    for s, (b0, b1, b2, _, a1, a2) in enumerate(sos):
        z0, z1 = float(states[s, 0]), float(states[s, 1])
        for n, xn in enumerate(y):
            yn = b0 * xn + z0
            z0 = b1 * xn - a1 * yn + z1
            z1 = b2 * xn - a2 * yn
            y[n] = yn
        final_states[s] = (z0, z1)
    y = np.array(y)
    return y if zi is None else (y, final_states)


def _padlen(sos):
    trailing_zeros = min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())
    return 3 * (2 * len(sos) + 1 - trailing_zeros)


def sosfiltfilt(sos, x):
    """Zero-phase filtering: forward pass, then backward pass."""
    x = np.asarray(x, dtype=float)
    padlen = _padlen(sos)
    if x.size <= padlen:
        raise ValueError(f"Signal must be longer than {padlen} samples.")
    # odd extension at both ends reduces start/end transients
    left = 2 * x[0] - x[padlen:0:-1]
    right = 2 * x[-1] - x[-2:-(padlen + 2):-1]
    ext = np.concatenate([left, x, right])
    zi = sosfilt_zi(sos)
    y, _ = sosfilt(sos, ext, zi=zi * ext[0])
    y, _ = sosfilt(sos, y[::-1], zi=zi * y[-1])
    return y[::-1][padlen:-padlen]


# --------------------------------------------------------------------------
# Frequency response
# --------------------------------------------------------------------------
def freqz_sos(sos, worN=512, fs=2 * np.pi):
    if np.ndim(worN) == 0:
        freqs = np.linspace(0, fs / 2, int(worN), endpoint=False)
    else:
        freqs = np.asarray(worN, dtype=float)
    z_inv = np.exp(-1j * 2 * np.pi * freqs / fs)
    h = np.ones_like(z_inv)
    for b0, b1, b2, a0, a1, a2 in sos:
        h *= (b0 + b1 * z_inv + b2 * z_inv ** 2) / (a0 + a1 * z_inv + a2 * z_inv ** 2)
    return freqs, h


# --------------------------------------------------------------------------
# Peak detection
# --------------------------------------------------------------------------
def _local_maxima(x):
    """Indices of local maxima (flat tops -> middle sample), like SciPy."""
    peaks = []
    i, last = 1, x.size - 1
    while i < last:
        if x[i - 1] < x[i]:
            ahead = i + 1
            while ahead < last and x[ahead] == x[i]:
                ahead += 1
            if x[ahead] < x[i]:
                peaks.append((i + ahead - 1) // 2)
                i = ahead
        i += 1
    return np.array(peaks, dtype=int)


def _apply_distance(peaks, heights, distance):
    """Keep the highest peaks; drop lower peaks closer than `distance` samples."""
    keep = np.ones(peaks.size, dtype=bool)
    for i in np.argsort(heights)[::-1]:  # same tie-breaking as SciPy
        if not keep[i]:
            continue
        j = i - 1
        while j >= 0 and peaks[i] - peaks[j] < distance:
            keep[j] = False
            j -= 1
        j = i + 1
        while j < peaks.size and peaks[j] - peaks[i] < distance:
            keep[j] = False
            j += 1
    return peaks[keep]


def _prominences(x, peaks):
    """Height of each peak above the higher of its two surrounding minima."""
    prom = np.empty(peaks.size)
    for k, p in enumerate(peaks):
        higher_left = np.nonzero(x[:p][::-1] > x[p])[0]
        start = p - higher_left[0] if higher_left.size else 0
        higher_right = np.nonzero(x[p + 1:] > x[p])[0]
        stop = p + higher_right[0] if higher_right.size else x.size - 1
        left_min = x[start:p + 1].min()
        right_min = x[p:stop + 1].min()
        prom[k] = x[p] - max(left_min, right_min)
    return prom


def find_peaks(x, distance=None, prominence=None):
    x = np.asarray(x, dtype=float)
    peaks = _local_maxima(x)
    if distance is not None and peaks.size:
        peaks = _apply_distance(peaks, x[peaks], distance)
    properties = {}
    if prominence is not None and peaks.size:
        prom = _prominences(x, peaks)
        keep = prom >= prominence
        peaks = peaks[keep]
        properties["prominences"] = prom[keep]
    return peaks, properties
