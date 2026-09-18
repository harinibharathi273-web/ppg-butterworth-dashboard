"""
Signal-processing modules for the PPG / Butterworth project.

ppg_generator : BPM data  -> synthetic PPG-like waveform (+ normalization, CSV loading)
noise         : simulated sensor noise (EMI, Gaussian, baseline wander, motion, quantization)
butterworth   : digital Butterworth low-pass filter design, filtering, frequency response
analysis      : peak detection, heart-rate metrics, SNR, spectrum
"""
