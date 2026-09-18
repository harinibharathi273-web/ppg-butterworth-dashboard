"""
app.py - PPG Signal Processing & Butterworth Filter Analysis
============================================================
Streamlit dashboard. Run with:

    streamlit run app.py

Pipeline:
BPM data -> synthetic PPG -> normalization -> simulated noise
-> Butterworth low-pass filter -> peak detection -> heart rate -> SNR -> plots
"""

import io
import time as time_module

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules import analysis, butterworth, noise, ppg_generator

# --------------------------------------------------------------------------
# Page setup and constants
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="PPG Signal Processing & Butterworth Filter Analysis",
    page_icon="🫀",
    layout="wide",
)

COLORS = {
    "clean": "#16A34A",     # green
    "noisy": "#94A3B8",     # grey
    "filtered": "#2563EB",  # blue
    "peaks": "#DC2626",     # red
    "input": "#F59E0B",     # amber
}

MIN_FS = 20.0
MIN_DURATION_S = 3.0

SAMPLE_CSV = "Time,BPM\n0,68\n1,70\n2,72\n3,75\n4,73\n5,78\n6,80\n7,76\n8,74\n9,72\n"

LIMITATION_TEXT = (
    "The system does not reconstruct the original physiological PPG waveform from "
    "BPM alone. BPM measurements are used to generate a synthetic PPG-like waveform "
    "with corresponding beat intervals. This synthetic signal is then used to "
    "demonstrate noise addition, Butterworth filtering and heart-rate estimation."
)


def make_figure(title, y_title="Normalized amplitude", x_title="Time (s)", height=330):
    """Plotly figure with a consistent layout."""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=height,
        template="plotly_white",
        margin=dict(l=50, r=20, t=60, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


def show_table(df, height=None):
    """st.dataframe needs the pyarrow library; if Windows blocks it, fall back to HTML."""
    try:
        if height is None:
            st.dataframe(df, hide_index=True)
        else:
            st.dataframe(df, hide_index=True, height=height)
    except Exception:
        st.markdown(df.head(200).to_html(index=False), unsafe_allow_html=True)
        if len(df) > 200:
            st.caption(f"Showing the first 200 of {len(df)} rows. Download the CSV for all rows.")


def fail(message):
    """Show an error and stop the script (instead of crashing)."""
    st.error(message)
    st.stop()


def fmt(value, pattern="{:.1f}", fallback="—"):
    """Format a number, or return a dash if it is missing/infinite."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "∞" if value == float("inf") else fallback
    return pattern.format(value)


# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
st.sidebar.header("Controls")
input_method = st.sidebar.radio("Input method", ("Manual BPM", "CSV upload"))

if input_method == "CSV upload":
    st.sidebar.download_button(
        "Download sample CSV", SAMPLE_CSV, file_name="sample_bpm.csv", mime="text/csv"
    )

with st.sidebar.form("controls_form"):
    st.subheader("1. Heart-rate input")
    uploaded_file = None
    manual_bpm = manual_duration = None
    if input_method == "Manual BPM":
        manual_bpm = st.number_input(
            "BPM (beats per minute)", min_value=1.0, max_value=400.0, value=75.0, step=1.0,
            help=f"Supported range: {ppg_generator.MIN_BPM}-{ppg_generator.MAX_BPM} BPM",
        )
        manual_duration = st.number_input(
            "Duration (seconds)", min_value=1.0, max_value=300.0, value=10.0, step=1.0
        )
    else:
        uploaded_file = st.file_uploader("BPM CSV file", type=["csv"])
        st.caption("Format: `Time,BPM` (time in seconds). Without a file, a built-in sample is used.")

    fs = st.number_input(
        "Sampling frequency (Hz)", min_value=1.0, max_value=2000.0, value=100.0, step=10.0
    )
    include_notch = st.checkbox("Include dicrotic notch", value=True)
    include_baseline = st.checkbox("Include breathing baseline variation", value=True)

    st.subheader("2. Simulated noise")
    noise_level = st.select_slider("Noise level", options=list(noise.NOISE_LEVELS), value="Medium")
    use_hf = st.checkbox("High-frequency / EMI noise", value=True)
    use_gaussian = st.checkbox("Random Gaussian noise", value=True)
    use_wander = st.checkbox("Baseline wander", value=False)
    use_motion = st.checkbox("Motion artifacts", value=False)
    use_quant = st.checkbox("ADC quantization noise", value=False)
    quant_bits = st.slider("ADC resolution (bits)", min_value=3, max_value=12, value=6)
    seed = st.number_input(
        "Random seed", min_value=0, max_value=10_000, value=42, step=1,
        help="Same seed gives the same noise, so results are reproducible.",
    )

    st.subheader("3. Butterworth low-pass filter")
    filter_order = st.number_input("Filter order (n)", min_value=1, max_value=10, value=4, step=1)
    cutoff_hz = st.number_input(
        "Cutoff frequency (Hz)", min_value=0.1, max_value=1000.0, value=5.0, step=0.5
    )
    zero_phase = st.checkbox(
        "Zero-phase filtering (filtfilt)", value=True,
        help="Forward + backward filtering: removes the time delay of the filter.",
    )

    st.form_submit_button("Generate", type="primary")


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("PPG Signal Processing & Butterworth Filter Analysis")
st.markdown("#### Synthetic PPG Reconstruction from Heart Rate Data")
st.caption(
    "Pipeline: BPM data → Synthetic PPG → Normalization → Simulated noise → "
    "Butterworth low-pass filter → Peak detection → Heart rate → SNR analysis"
)
st.info(f"**Note:** {LIMITATION_TEXT}")


# --------------------------------------------------------------------------
# PIPELINE (all numbers below come from these calculations)
# --------------------------------------------------------------------------
# Step 1 - BPM input ---------------------------------------------------------
csv_warnings = []
try:
    if input_method == "Manual BPM":
        ppg_generator.validate_bpm([manual_bpm])
        bpm_times = np.array([0.0])
        bpm_values = np.array([float(manual_bpm)])
        duration_s = float(manual_duration)
        source_label = "Manual entry"
    else:
        if uploaded_file is None:
            csv_source = io.StringIO(SAMPLE_CSV)
            source_label = "Built-in sample CSV (upload your own in the sidebar)"
        else:
            csv_source = uploaded_file
            source_label = uploaded_file.name
        bpm_times, bpm_values, csv_warnings = ppg_generator.load_bpm_csv(csv_source)
        duration_s = ppg_generator.duration_from_csv_times(bpm_times)
except ValueError as exc:
    fail(f"**Input error:** {exc}")

if fs < MIN_FS:
    fail(
        f"**Invalid sampling frequency:** {fs:g} Hz is too low. Use at least "
        f"{MIN_FS:g} Hz so the pulse shape and filter cutoff can be represented "
        "(Nyquist: fs must exceed twice the highest frequency of interest)."
    )
if duration_s < MIN_DURATION_S:
    fail(
        f"**Signal too short:** {duration_s:g} s. Use at least {MIN_DURATION_S:g} s "
        "so that several heartbeats can be detected."
    )

# Step 2 - Butterworth filter design (checked early so bad cutoffs stop here)
try:
    sos = butterworth.design_butterworth_lowpass(int(filter_order), float(cutoff_hz), float(fs))
except ValueError as exc:
    fail(f"**Filter error:** {exc}")

# Step 3 - synthetic PPG generation + normalization
try:
    generated = ppg_generator.generate_synthetic_ppg(
        bpm_times, bpm_values, duration_s, fs,
        include_dicrotic_notch=include_notch,
        include_baseline_variation=include_baseline,
    )
except ValueError as exc:
    fail(f"**Signal generation error:** {exc}")

time_axis = generated["time"]
clean = generated["ppg"]

# Step 4 - simulated noise
noisy, noise_components = noise.add_noise(
    clean, time_axis, fs, level=noise_level,
    high_frequency=use_hf, gaussian=use_gaussian,
    baseline_wander=use_wander, motion_artifacts=use_motion,
    quantization=use_quant, quantization_bits=quant_bits, seed=int(seed),
)

# Step 5 - Butterworth low-pass filtering
try:
    filtered = butterworth.apply_butterworth(noisy, sos, zero_phase=zero_phase)
except ValueError as exc:
    fail(f"**Filtering error:** {exc}")

# Step 6 - peak detection and heart rate
peak_indices = analysis.detect_peaks(filtered, fs)
peak_times = time_axis[peak_indices]
hr = analysis.heart_rate_metrics(peak_times)
accuracy = analysis.beat_detection_accuracy(peak_times, generated["true_peak_times"])

input_bpm_curve = np.interp(time_axis, bpm_times, bpm_values)  # BPM at every sample
input_mean_bpm = float(input_bpm_curve.mean())
max_beat_freq = float(bpm_values.max()) / 60.0

# Step 7 - SNR (clean synthetic PPG is the reference)
snr_before = analysis.snr_db(clean, noisy)
snr_after = analysis.snr_db(clean, filtered)
snr_gain = snr_after - snr_before if np.isfinite(snr_before) and np.isfinite(snr_after) else None


# --------------------------------------------------------------------------
# SECTION 1 - INPUT
# --------------------------------------------------------------------------
st.header("1. Input")
cols = st.columns(5)
cols[0].metric("Input method", input_method)
cols[1].metric("Input BPM (mean)", f"{input_mean_bpm:.1f}")
cols[2].metric("BPM range", f"{bpm_values.min():.0f} – {bpm_values.max():.0f}")
cols[3].metric("Duration", f"{duration_s:g} s")
cols[4].metric("Sampling frequency", f"{fs:g} Hz")
st.caption(
    f"Source: {source_label}. {time_axis.size} samples, Nyquist frequency = {fs / 2:g} Hz."
)
for message in csv_warnings:
    st.warning(message)
if input_method == "CSV upload":
    with st.expander("Show BPM data used"):
        show_table(pd.DataFrame({"Time (s)": bpm_times, "BPM": bpm_values}))


# --------------------------------------------------------------------------
# SECTION 2 - SIGNAL GENERATION
# --------------------------------------------------------------------------
st.header("2. Signal Generation")
periods = generated["beat_periods"]
if bpm_values.size == 1 or np.ptp(bpm_values) == 0:
    bpm0 = float(bpm_values[0])
    st.markdown(
        f"At **{bpm0:g} BPM**: beat frequency = {bpm0:g} / 60 = **{bpm0 / 60:.3f} Hz**, "
        f"beat interval = 60 / {bpm0:g} = **{60 / bpm0:.3f} s** → "
        f"**{periods.size} pulses** generated in {duration_s:g} s."
    )
else:
    st.markdown(
        f"BPM varies from **{bpm_values.min():g}** to **{bpm_values.max():g}**, so the "
        f"beat interval varies from **{periods.min():.3f} s** to **{periods.max():.3f} s** "
        f"(interval = 60 / BPM for every beat). **{periods.size} pulses** generated."
    )

fig = make_figure("Synthetic PPG Signal (normalized 0–1)")
fig.add_trace(go.Scatter(x=time_axis, y=clean, name="Synthetic PPG",
                         line=dict(color=COLORS["clean"], width=2)))
st.plotly_chart(fig, key="clean_plot")

with st.expander("Normalization step (before vs after)"):
    st.latex(r"x_{norm}[k] = \frac{x[k] - \min(x)}{\max(x) - \min(x)}")
    fig = make_figure("Raw synthetic signal vs normalized signal", y_title="Amplitude", height=280)
    fig.add_trace(go.Scatter(x=time_axis, y=generated["raw"], name="Before normalization",
                             line=dict(color=COLORS["noisy"])))
    fig.add_trace(go.Scatter(x=time_axis, y=clean, name="After normalization (0–1)",
                             line=dict(color=COLORS["clean"])))
    st.plotly_chart(fig, key="normalization_plot")


# --------------------------------------------------------------------------
# SECTION 3 - NOISE
# --------------------------------------------------------------------------
st.header("3. Noise")
if noise_components:
    st.markdown(
        f"Noise level **{noise_level}** with: " + ", ".join(noise_components.keys()) + "."
    )
else:
    st.markdown("No noise added (level **None** or all noise types switched off).")

fig = make_figure("Noisy PPG Signal")
fig.add_trace(go.Scatter(x=time_axis, y=noisy, name="Noisy PPG",
                         line=dict(color=COLORS["noisy"], width=1.2)))
st.plotly_chart(fig, key="noisy_plot")

if noise_components:
    with st.expander("Individual noise components"):
        fig = make_figure("Noise components", y_title="Amplitude", height=300)
        for name, component in noise_components.items():
            fig.add_trace(go.Scatter(x=time_axis, y=component, name=name, line=dict(width=1)))
        st.plotly_chart(fig, key="components_plot")
if use_wander or use_motion:
    st.warning(
        "Baseline wander (< 0.5 Hz) and motion artifacts (1–3 Hz) lie at or below the "
        "heart-rate band, so a **low-pass** filter cannot remove them. This is expected: "
        "they would need a high-pass/band-pass filter or adaptive methods."
    )


# --------------------------------------------------------------------------
# SECTION 4 - FILTER
# --------------------------------------------------------------------------
st.header("4. Butterworth Low-Pass Filter")
mode_text = "zero-phase (filtfilt)" if zero_phase else "causal (single pass, introduces delay)"
st.markdown(
    f"Order **{int(filter_order)}**, cutoff **{cutoff_hz:g} Hz**, fs **{fs:g} Hz**, {mode_text}. "
    f"Red markers are the detected peaks."
)
suggested_cutoff = round(6 * max_beat_freq, 1)
if cutoff_hz < 3.5 * max_beat_freq:
    st.warning(
        f"The cutoff ({cutoff_hz:g} Hz) is close to the heart-rate frequency "
        f"({max_beat_freq:.2f} Hz at {bpm_values.max():g} BPM). The filter will also "
        f"remove harmonics of the pulse and distort its shape, which lowers the SNR "
        f"after filtering. Try a cutoff around {suggested_cutoff:g} Hz."
    )

fig = make_figure("Butterworth Filtered PPG")
fig.add_trace(go.Scatter(x=time_axis, y=filtered, name="Filtered PPG",
                         line=dict(color=COLORS["filtered"], width=2)))
fig.add_trace(go.Scatter(x=peak_times, y=filtered[peak_indices], mode="markers",
                         name="Detected peaks",
                         marker=dict(color=COLORS["peaks"], size=9, symbol="x")))
st.plotly_chart(fig, key="filtered_plot")


# --------------------------------------------------------------------------
# SECTION 5 - COMPARISON
# --------------------------------------------------------------------------
st.header("5. Comparison")
tab_time, tab_freq = st.tabs(["Time domain", "Frequency domain"])
with tab_time:
    fig = make_figure("Clean vs Noisy vs Filtered PPG", height=380)
    fig.add_trace(go.Scatter(x=time_axis, y=noisy, name="Noisy PPG",
                             line=dict(color=COLORS["noisy"], width=1)))
    fig.add_trace(go.Scatter(x=time_axis, y=clean, name="Clean synthetic PPG",
                             line=dict(color=COLORS["clean"], width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=time_axis, y=filtered, name="Filtered PPG",
                             line=dict(color=COLORS["filtered"], width=2)))
    st.plotly_chart(fig, key="comparison_plot")
with tab_freq:
    fig = make_figure("Amplitude spectrum", y_title="Amplitude (dB)",
                      x_title="Frequency (Hz)", height=380)
    for label, sig, color in (("Noisy PPG", noisy, COLORS["noisy"]),
                              ("Clean synthetic PPG", clean, COLORS["clean"]),
                              ("Filtered PPG", filtered, COLORS["filtered"])):
        freqs, amp = analysis.amplitude_spectrum(sig, fs)
        fig.add_trace(go.Scatter(x=freqs, y=20 * np.log10(amp + 1e-9), name=label,
                                 line=dict(color=color, width=1.5)))
    fig.add_vline(x=cutoff_hz, line_dash="dot", line_color=COLORS["peaks"],
                  annotation_text=f"cutoff {cutoff_hz:g} Hz")
    fig.update_yaxes(range=[-120, 0])
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, key="spectrum_plot")
    st.caption(
        "The heart-rate fundamental and its first harmonics sit below the cutoff; "
        "the EMI tones and most of the white noise sit above it and are attenuated."
    )


# --------------------------------------------------------------------------
# SECTION 6 - RESULTS
# --------------------------------------------------------------------------
st.header("6. Results")
if hr is None:
    st.warning(
        "Fewer than 2 peaks were detected, so heart rate cannot be estimated. "
        "Try a lower noise level, a longer duration or a higher cutoff frequency."
    )

row1 = st.columns(6)
row1[0].metric("Input BPM", f"{input_mean_bpm:.1f}")
row1[1].metric(
    "Estimated BPM",
    fmt(hr["estimated_bpm"]) if hr else "—",
    delta=f"{hr['estimated_bpm'] - input_mean_bpm:+.1f} vs input" if hr else None,
    delta_color="off",
)
row1[2].metric("SNR Before", f"{fmt(snr_before)} dB")
row1[3].metric("SNR After", f"{fmt(snr_after)} dB")
row1[4].metric("SNR Improvement", f"{fmt(snr_gain, '{:+.1f}')} dB")
row1[5].metric("Detected Peaks", f"{peak_indices.size}")

row2 = st.columns(6)
row2[0].metric("Avg inter-beat interval", f"{hr['mean_ibi_s']:.3f} s" if hr else "—")
row2[1].metric("Mean BPM", fmt(hr["mean_bpm"]) if hr else "—")
row2[2].metric("Min BPM", fmt(hr["min_bpm"]) if hr else "—")
row2[3].metric("Max BPM", fmt(hr["max_bpm"]) if hr else "—")
row2[4].metric("Beats found (sensitivity)", f"{fmt(accuracy['sensitivity'], '{:.0f}')} %")
row2[5].metric("Correct detections (precision)", f"{fmt(accuracy['precision'], '{:.0f}')} %")

with st.expander("How are these values calculated?"):
    st.markdown(
        "**Assumption:** because the signal is synthetic, the clean PPG is known exactly "
        "and is used as the *reference*. The noisy signal is the *noisy observation* and "
        "the filtered signal is the *filtered observation*."
    )
    st.latex(r"P_{signal} = \overline{(x_{clean} - \bar{x}_{clean})^2}, \qquad "
             r"P_{noise} = \overline{(x_{observed} - x_{clean})^2}")
    st.latex(r"SNR_{dB} = 10 \log_{10}\left(\frac{P_{signal}}{P_{noise}}\right)")
    st.markdown(
        "- For *SNR After*, any difference from the clean signal counts as noise, including "
        "the slight smoothing of the pulse by the filter itself.\n"
        "- **IBI** = time between consecutive detected peaks; instantaneous BPM = 60 / IBI.\n"
        "- **Estimated BPM** = 60 / mean(IBI). **Mean BPM** = average of the instantaneous values.\n"
        "- **Sensitivity / precision** compare detected peaks with the exact synthetic peak "
        "times (tolerance 0.15 s)."
    )


# --------------------------------------------------------------------------
# SECTION 7 - FILTER PARAMETERS + FREQUENCY RESPONSE
# --------------------------------------------------------------------------
st.header("7. Filter Parameters")
param_col, response_col = st.columns([1, 2])
with param_col:
    params = pd.DataFrame({
        "Parameter": ["Filter type", "Filter order", "Cutoff frequency", "Sampling frequency",
                      "Nyquist frequency", "Normalized cutoff (fc / (fs/2))",
                      "Structure", "Filtering mode", "DSP engine"],
        "Value": ["Butterworth Low-Pass (digital IIR)", f"{int(filter_order)}",
                  f"{cutoff_hz:g} Hz", f"{fs:g} Hz", f"{fs / 2:g} Hz",
                  f"{cutoff_hz / (fs / 2):.3f}",
                  f"{len(sos)} cascaded 2nd-order section(s)", mode_text,
                  butterworth.DSP_BACKEND],
    })
    show_table(params)
    gain_fc = butterworth.gain_db_at(sos, fs, cutoff_hz)
    st.markdown(f"Gain at cutoff: **{gain_fc:.2f} dB** (single pass)")
    if 2 * cutoff_hz < fs / 2:
        st.markdown(f"Gain at 2 × cutoff: **{butterworth.gain_db_at(sos, fs, 2 * cutoff_hz):.1f} dB**")
    st.markdown(f"Theoretical roll-off: **{20 * int(filter_order)} dB/decade**")

with response_col:
    compare_orders = st.checkbox("Compare with other filter orders", value=True)
    freqs, mag_db, mag_db_zero_phase = butterworth.frequency_response(sos, fs)
    fig = make_figure(f"Butterworth magnitude response (n = {int(filter_order)}, "
                      f"fc = {cutoff_hz:g} Hz)",
                      y_title="Magnitude (dB)", x_title="Frequency (Hz)", height=380)
    if compare_orders:
        for other_order in (1, 2, 4, 6, 8):
            if other_order == int(filter_order):
                continue
            other_sos = butterworth.design_butterworth_lowpass(other_order, cutoff_hz, fs)
            f_o, m_o, _ = butterworth.frequency_response(other_sos, fs)
            fig.add_trace(go.Scatter(x=f_o, y=m_o, name=f"n = {other_order}",
                                     line=dict(width=1, dash="dot"), opacity=0.6))
    fig.add_trace(go.Scatter(x=freqs, y=mag_db, name=f"n = {int(filter_order)} (selected)",
                             line=dict(color=COLORS["filtered"], width=3)))
    if zero_phase:
        fig.add_trace(go.Scatter(x=freqs, y=mag_db_zero_phase,
                                 name="Effective response with filtfilt (|H|²)",
                                 line=dict(color=COLORS["peaks"], width=2, dash="dash")))
    fig.add_hline(y=-3.01, line_dash="dot", line_color="gray", annotation_text="−3 dB")
    fig.add_vline(x=cutoff_hz, line_dash="dot", line_color="gray",
                  annotation_text=f"fc = {cutoff_hz:g} Hz")
    fig.update_yaxes(range=[-80, 5])
    fig.update_xaxes(range=[0, min(fs / 2, max(4 * cutoff_hz, 10))])
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, key="response_plot")

with st.expander("Cutoff trade-off: SNR after filtering vs cutoff frequency"):
    st.markdown(
        "Every point below re-runs the filter on the **same noisy signal** with a different "
        "cutoff. Too low a cutoff distorts the pulse; too high a cutoff lets noise through."
    )
    sweep_step = 0.5 if (butterworth.USING_SCIPY or time_axis.size <= 20_000) else 2.0
    sweep_cutoffs = np.arange(1.0, min(0.9 * fs / 2, 25.0) + 0.001, sweep_step)
    sweep_snr = []
    for fc in sweep_cutoffs:
        try:
            sweep_sos = butterworth.design_butterworth_lowpass(int(filter_order), float(fc), fs)
            sweep_snr.append(analysis.snr_db(
                clean, butterworth.apply_butterworth(noisy, sweep_sos, zero_phase)))
        except ValueError:
            sweep_snr.append(np.nan)
    sweep_snr = np.array(sweep_snr)
    fig = make_figure("SNR after filtering vs cutoff", y_title="SNR after (dB)",
                      x_title="Cutoff frequency (Hz)", height=320)
    fig.add_trace(go.Scatter(x=sweep_cutoffs, y=sweep_snr, mode="lines+markers",
                             name="SNR after", line=dict(color=COLORS["filtered"])))
    if np.isfinite(snr_before):
        fig.add_hline(y=snr_before, line_dash="dot", line_color="gray",
                      annotation_text="SNR before filtering")
    fig.add_vline(x=cutoff_hz, line_dash="dot", line_color=COLORS["peaks"],
                  annotation_text="current cutoff")
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, key="sweep_plot")
    if np.any(np.isfinite(sweep_snr)):
        best = int(np.nanargmax(sweep_snr))
        st.markdown(
            f"Best cutoff for this signal: **{sweep_cutoffs[best]:g} Hz** "
            f"(SNR after = {sweep_snr[best]:.1f} dB)."
        )


# --------------------------------------------------------------------------
# SECTION 8 - DATA TABLE
# --------------------------------------------------------------------------
st.header("8. Data Table")
data_table = pd.DataFrame({
    "Time (s)": np.round(time_axis, 4),
    "Clean PPG": np.round(clean, 5),
    "Noisy PPG": np.round(noisy, 5),
    "Filtered PPG": np.round(filtered, 5),
})
show_table(data_table, height=300)
st.download_button(
    "Download processed data (CSV)",
    data_table.to_csv(index=False).encode("utf-8"),
    file_name="ppg_processed_data.csv",
    mime="text/csv",
)


# --------------------------------------------------------------------------
# SECTION 9 - INPUT BPM vs ESTIMATED BPM
# --------------------------------------------------------------------------
st.header("9. Input BPM vs Estimated BPM")
fig = make_figure("Heart rate: input vs estimated from filtered PPG",
                  y_title="Heart rate (BPM)", height=340)
fig.add_trace(go.Scatter(x=time_axis, y=input_bpm_curve, name="Input BPM",
                         line=dict(color=COLORS["input"], width=3)))
if hr is not None:
    fig.add_trace(go.Scatter(x=hr["inst_bpm_times"], y=hr["inst_bpm"],
                             name="Estimated BPM (beat-by-beat, 60 / IBI)",
                             mode="lines+markers",
                             line=dict(color=COLORS["filtered"], width=2)))
    input_at_beats = np.interp(hr["inst_bpm_times"], bpm_times, bpm_values)
    mae = float(np.mean(np.abs(hr["inst_bpm"] - input_at_beats)))
    st.markdown(f"Mean absolute error (beat-by-beat): **{mae:.2f} BPM**")
st.plotly_chart(fig, key="bpm_compare_plot")


# --------------------------------------------------------------------------
# SECTION 10 - REAL-TIME SIMULATION (optional replay)
# --------------------------------------------------------------------------
st.header("10. Real-time Simulation")
st.caption(
    "Replays the processed signal sample by sample, as if it were arriving from a sensor. "
    "The filter was already applied to the whole recording; this is a visual replay."
)
sim_cols = st.columns([1, 1, 3])
speed = sim_cols[0].selectbox("Playback speed", ("1x", "2x", "4x"), index=1)
start_simulation = sim_cols[1].button("▶ Start Simulation", type="primary")
live_metric = st.empty()
live_chart = st.empty()

if start_simulation:
    n_frames = int(min(80, time_axis.size))
    delay_s = duration_s / n_frames / float(speed.rstrip("x"))
    y_min = float(min(noisy.min(), filtered.min())) - 0.05
    y_max = float(max(noisy.max(), filtered.max())) + 0.05
    for frame in range(1, n_frames + 1):
        end = int(time_axis.size * frame / n_frames)
        visible = peak_indices[peak_indices < end]
        frame_fig = make_figure("Live PPG stream")
        frame_fig.add_trace(go.Scatter(x=time_axis[:end], y=noisy[:end], name="Noisy (sensor)",
                                       line=dict(color=COLORS["noisy"], width=1)))
        frame_fig.add_trace(go.Scatter(x=time_axis[:end], y=filtered[:end], name="Filtered",
                                       line=dict(color=COLORS["filtered"], width=2)))
        frame_fig.add_trace(go.Scatter(x=time_axis[visible], y=filtered[visible],
                                       mode="markers", name="Beats",
                                       marker=dict(color=COLORS["peaks"], size=9, symbol="x")))
        frame_fig.update_xaxes(range=[0, duration_s])
        frame_fig.update_yaxes(range=[y_min, y_max])
        live_chart.plotly_chart(frame_fig, key=f"sim_frame_{frame}")
        if visible.size >= 2:
            recent = time_axis[visible[-6:]]  # last up to 5 intervals
            live_metric.metric("Live heart rate (last beats)",
                               f"{60.0 / np.diff(recent).mean():.1f} BPM")
        else:
            live_metric.metric("Live heart rate (last beats)", "waiting for beats…")
        time_module.sleep(delay_s)
    st.success("Simulation finished.")
else:
    live_chart.info("Press ▶ Start Simulation to replay the signal.")


# --------------------------------------------------------------------------
# ABOUT / METHODOLOGY
# --------------------------------------------------------------------------
st.header("About / Methodology")
st.warning(f"**Scientific limitation.** {LIMITATION_TEXT}")
st.markdown(
    """
1. **BPM input** - manual value or a CSV of `Time,BPM` readings (e.g. from the pulse-sensor hardware).
2. **Synthetic PPG generation** - one PPG-like pulse per beat, each beat lasting **60 / BPM** seconds.
   The pulse has a fast systolic upstroke, slower decay, a dicrotic notch and a diastolic wave.
3. **Normalization** - min-max scaling to 0–1 (the "Normalization" block of the project pipeline).
4. **Noise simulation** - EMI tones, Gaussian noise, baseline wander, motion artifacts and
   quantization, representing the noise sources named in the problem statement.
5. **Butterworth low-pass filter** - digital IIR filter,
   |H(jω)| = 1 / √(1 + (ω/ωc)²ⁿ), implemented as cascaded second-order sections;
   zero-phase filtering keeps the peaks at the correct time.
6. **Peak detection** - `scipy.signal.find_peaks` with minimum-distance and prominence rules.
7. **Heart rate & SNR** - BPM = 60 / mean inter-beat interval; SNR computed against the clean reference.
"""
)
