# PPG Signal Processing & Butterworth Filter Analysis
**Synthetic PPG Reconstruction from Heart Rate Data** — software prototype for the
Butterworth Low-Pass Filter project (EE304 Signal Processing).

## Run it (Windows)
```bash
cd ppg_signal_project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
The dashboard opens at http://localhost:8501. No Arduino is needed.
(On macOS/Linux activate with `source venv/bin/activate`. The venv step is optional.)

## Pipeline
```
BPM data → Synthetic PPG → Normalization → Simulated noise → Butterworth low-pass
→ Peak detection → Estimated heart rate → SNR analysis → Visualization
```

| File | Role |
|---|---|
| `app.py` | Streamlit dashboard (inputs, plots, metrics, error messages) |
| `modules/ppg_generator.py` | BPM → synthetic PPG, normalization, CSV loading |
| `modules/noise.py` | EMI, Gaussian, baseline wander, motion artifacts, quantization |
| `modules/butterworth.py` | Filter design (`scipy.signal.butter`, SOS), filtering, frequency response |
| `modules/analysis.py` | Peak detection, BPM/IBI, beat-detection accuracy, SNR, spectrum |
| `sample_data/` | `sample_bpm.csv` (10 s) and `sample_bpm_exercise.csv` (30 s, 65→108→70 BPM) |

## CSV format
```
Time,BPM
0,68
1,70
```
`Time` is in seconds (optional — without it one reading per second is assumed).
Column names are case-insensitive; `HR`, `Heart Rate` and `Pulse` are also accepted.

## Default demo
BPM 75, 10 s, fs 100 Hz, Medium noise, order 4, cutoff 5 Hz, zero-phase.
Expected: 13 peaks, estimated BPM 75.0, SNR ≈ 11.3 dB → 18.3 dB (+7 dB).

## Scientific limitation
The system does not reconstruct the original physiological PPG waveform from BPM alone.
BPM measurements are used to generate a synthetic PPG-like waveform with corresponding beat
intervals. This synthetic signal is then used to demonstrate noise addition, Butterworth
filtering and heart-rate estimation.
