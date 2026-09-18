# Viva guide

## 1-minute explanation
Our pulse-sensor hardware gives reliable BPM readings, but its raw waveform was not reliable
enough to analyse. So we built a software prototype that takes the BPM readings, generates a
synthetic PPG-like signal with one pulse every 60/BPM seconds, normalizes it to 0–1, and adds
realistic sensor noise: EMI tones, Gaussian noise, and optionally baseline wander, motion
artifacts and ADC quantization. A digital 4th-order Butterworth low-pass filter at 5 Hz,
applied with zero-phase filtering, removes the high-frequency noise without shifting the peaks.
We then detect peaks, estimate heart rate from the inter-beat intervals, and measure SNR
against the known clean signal. At 75 BPM with medium noise, the estimated heart rate is
75.0 BPM and SNR improves from about 11 dB to 18 dB. We clearly state that BPM cannot recover
the real PPG shape; the synthetic signal is a controlled test signal for the filter.

## Likely questions and answers

**Can you get the real PPG back from BPM?**
No. BPM is one number per reading; it contains only beat frequency, not pulse shape. We
generate a synthetic waveform whose timing matches the BPM. Because the clean signal is known,
we can measure exactly how well the filter works, which is not possible with real data.

**How does BPM become a pulse spacing?**
Beat interval = 60/BPM seconds, beat frequency = BPM/60 Hz. 75 BPM → 0.8 s → 1.25 Hz.
For CSV data, each beat uses the BPM at the moment it starts, so the spacing changes with BPM.

**Why Butterworth and not Chebyshev or a simple RC filter?**
Butterworth is maximally flat in the passband (no ripple), so pulse amplitude is not distorted.
Chebyshev rolls off faster but has passband ripple. A 1st-order RC gives only 20 dB/decade;
an n-th order Butterworth gives 20·n dB/decade (80 dB/decade for n = 4).

**Why 5 Hz cutoff?**
Heart rate is 0.5–3.7 Hz (30–220 BPM), and the pulse shape needs a few harmonics. 5 Hz keeps
the fundamental and main harmonics at normal heart rates and removes the EMI (15–50 Hz) and
most white noise. The "cutoff trade-off" plot shows too low a cutoff distorts the pulse and too
high lets noise in; about 6× the beat frequency works best. At high BPM the cutoff must rise.

**What is the effect of filter order?**
Higher order → sharper transition and more stopband attenuation, but more phase distortion
and ringing, and more computation. Order 4 is a common balance. The frequency-response plot
compares orders 1–8.

**What is zero-phase filtering and why use it?**
`filtfilt` filters forward then backward, so phase shifts cancel and peaks stay at the correct
time (important for HRV). The magnitude is applied twice (|H|², −6 dB at fc, effective order 2n).
It is non-causal, so it needs the whole block of data; a real-time device would use a causal
filter and accept a delay. Turning zero-phase off shows the delay: SNR after drops below 0 dB
even though BPM is still right, because the whole signal is shifted in time.

**Why second-order sections?**
High-order IIR filters in one transfer function are numerically sensitive. A cascade of
2nd-order sections is stable and matches the cascaded Sallen-Key stages in our block diagram.

**How is SNR calculated?**
Signal power = variance of the clean reference; noise power = mean squared difference between
the observed and clean signal; SNR = 10·log10(Ps/Pn). Filter distortion counts as noise,
so the result is conservative.

**Why doesn't the filter remove motion artifacts or baseline wander?**
They are at or below the heart-rate band (motion 1–3 Hz, wander < 0.5 Hz). A low-pass filter
cannot remove in-band noise. Baseline wander needs a high-pass (band-pass ≈ 0.5–5 Hz);
motion artifacts need adaptive filtering with an accelerometer reference.

**How does peak detection avoid counting the dicrotic wave?**
`find_peaks` with a minimum distance of 60/220 s and a prominence of 35% of the signal range;
the diastolic bump is much less prominent than the systolic peak.

**What is the Nyquist condition here?**
The cutoff must be below fs/2. At fs = 100 Hz, Nyquist is 50 Hz; the app rejects invalid cutoffs.

**What would you do next?**
Feed real raw PPG from the sensor (analogRead streamed over serial) into the same filter code,
add a high-pass stage for baseline wander, and implement the filter as biquads on the
microcontroller for real-time use.

**Your filter runs even without SciPy - how?**
On a laptop where Windows Smart App Control blocked SciPy's DLLs, the app automatically switches to
`modules/dsp_numpy.py`, our own NumPy implementation of the same steps: analog Butterworth poles on
a circle, frequency pre-warping, bilinear transform to the z-domain, grouping into 2nd-order
sections, Direct Form II transposed filtering, forward-backward zero-phase filtering, and peak
detection with distance and prominence rules. It was checked against SciPy: identical output to
about 1e-12. The "DSP engine" row in Filter Parameters shows which engine is running.
