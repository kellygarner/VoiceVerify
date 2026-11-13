import streamlit as st
import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks
import io
from pathlib import Path
import matplotlib.pyplot as plt

st.title("VoiceVerify")
st.write("Human Voice vs. AI-Generated Voice")

st.subheader("Choose How to Use VoiceVerify")

mode = st.radio(
    "Select an option:",
    [
        "Run Amazing Grace Demonstration",
        "Upload Your Own Recordings"
    ]
)

# Choose demonstration recordings or user uploads

if mode == "Run Amazing Grace Demonstration":

    st.subheader("Amazing Grace — VoiceVerify Demonstration")

    st.write(
        "Compare a real human singing performance "
        "with an AI-generated version of the same performance."
    )

    real_path = Path("demo_audio/real_voice.wav")
    ai_path = Path("demo_audio/ai_voice.wav")

    if real_path.exists() and ai_path.exists():

        uploaded_file = io.BytesIO(real_path.read_bytes())
        comparison_file = io.BytesIO(ai_path.read_bytes())

        st.success("Both demonstration recordings loaded successfully!")

        st.write("REAL VOICE — Human Recording")
        st.audio(uploaded_file.getvalue(), format="audio/wav")

        st.write("AI-GENERATED VOICE — Synthetic Recording")
        st.audio(comparison_file.getvalue(), format="audio/wav")

    else:
        uploaded_file = None
        comparison_file = None

        st.error("Demonstration recordings could not be found.")

else:

    st.subheader("Upload Your Own Recordings")

    uploaded_file = st.file_uploader(
        "REAL VOICE — Upload a WAV recording",
        type=["wav"]
    )

    comparison_file = st.file_uploader(
        "AI-GENERATED VOICE — Upload a WAV recording",
        type=["wav"],
        key="comparison"
    )

if mode == "Upload Your Own Recordings" and uploaded_file is not None:
    st.success("Audio file uploaded successfully!")
    st.audio(uploaded_file)

if mode == "Upload Your Own Recordings" and comparison_file is not None:
    st.success("Second audio file uploaded successfully!")
    st.audio(comparison_file)

if uploaded_file is not None:
    sample_rate, audio_data = wavfile.read(io.BytesIO(uploaded_file.getvalue()))
    if audio_data.ndim == 2:
        audio_data = audio_data.mean(axis=1)

    st.subheader("Voice Converted into Numbers")
    st.write("Sampling rate:", sample_rate, "samples per second")
    st.write("Total audio samples:", len(audio_data))
    st.write("First 10 amplitude samples:")
    st.write(audio_data[:10])

    st.subheader("Voice Waveform")
    time = np.arange(len(audio_data)) / sample_rate
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, audio_data)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Recorded Voice Waveform")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Fourier Frequency Spectrum")
    # Prefer the sustained part of the user's ~9-second recording.
    # For shorter clips, analyze the middle portion instead.
    duration = len(audio_data) / sample_rate
    if duration >= 7.0:
        start, end = int(2.0 * sample_rate), int(7.0 * sample_rate)
    else:
        start, end = int(0.2 * len(audio_data)), int(0.8 * len(audio_data))
    segment = np.asarray(audio_data[start:end], dtype=np.float64)

    if len(segment) < 256:
        st.warning("This recording is too short for reliable frequency analysis. Please upload a longer WAV file.")
    else:
        segment = segment - np.mean(segment)
        window = np.hanning(len(segment))
        fft_values = np.fft.rfft(segment * window)
        frequencies = np.fft.rfftfreq(len(segment), d=1 / sample_rate)
        magnitudes = np.abs(fft_values)

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(frequencies, magnitudes)
        ax2.set_xlim(0, 2000)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("FFT Magnitude")
        ax2.set_title("Fourier Spectrum of Recorded Voice")
        ax2.grid(True)
        st.pyplot(fig2)
        plt.close(fig2)

        st.subheader("Fundamental Frequency and Harmonics")
        st.write("The fundamental frequency (F₀) is the approximate rate of vocal-fold vibration. Harmonics occur near whole-number multiples of F₀.")

        # Estimate F0 by autocorrelation of a short, voiced window near the
        # center of the selected segment. This avoids assuming the loudest
        # spectral peak is the fundamental (it often is not for vowels).
        frame_length = min(len(segment), int(0.12 * sample_rate))
        center = len(segment) // 2
        frame = segment[center - frame_length // 2:center - frame_length // 2 + frame_length]
        frame = frame * np.hanning(len(frame))
        corr = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
        min_f0, max_f0 = 75.0, 500.0
        min_lag = max(1, int(sample_rate / max_f0))
        max_lag = min(len(corr) - 1, int(sample_rate / min_f0))
        if max_lag <= min_lag or corr[0] <= 0:
            st.warning("Could not estimate pitch from this recording.")
        else:
            candidates, _ = find_peaks(corr[min_lag:max_lag + 1])
            if len(candidates) == 0:
                st.warning("No clear repeating pitch was detected. Try a sustained vowel.")
            else:
                candidate_lags = candidates + min_lag
                best_lag = int(candidate_lags[np.argmax(corr[candidate_lags])])
                confidence = float(corr[best_lag] / corr[0])
                if confidence < 0.2:
                    st.warning("Pitch is unclear. Try a steady, sustained vowel with less background noise.")
                else:
                    # Parabolic interpolation gives a sub-sample lag estimate.
                    refined_lag = float(best_lag)
                    if 1 <= best_lag < len(corr) - 1:
                        left, mid, right = corr[best_lag - 1:best_lag + 2]
                        denominator = left - 2 * mid + right
                        if abs(denominator) > 1e-12:
                            refined_lag += 0.5 * (left - right) / denominator
                    f0 = sample_rate / refined_lag
                    st.metric("Estimated fundamental frequency (F₀)", f"{f0:.2f} Hz")
                    st.caption("Pitch is an estimate from a short section of the recording; a wavering voice or octave ambiguity can affect the result.")

                    # Find the strongest local spectral peak near each
                    # expected harmonic, not simply the globally loudest peak.
                    rows = []
                    harmonic_points = []
                    for n in range(1, int(2000 // f0) + 1):
                        expected = n * f0
                        tolerance = max(12.0, 0.035 * expected)
                        indices = np.where((frequencies >= expected - tolerance) &
                                           (frequencies <= expected + tolerance))[0]
                        if len(indices) == 0:
                            continue
                        peak_index = int(indices[np.argmax(magnitudes[indices])])
                        observed = float(frequencies[peak_index])
                        magnitude = float(magnitudes[peak_index])
                        rows.append({
                            "Harmonic": n,
                            "Expected frequency (Hz)": round(expected, 2),
                            "Observed peak (Hz)": round(observed, 2),
                            "FFT magnitude": round(magnitude, 1),
                        })
                        harmonic_points.append((n, observed, magnitude))

                    if rows:
                        st.write("Harmonic frequencies and magnitudes:")
                        st.dataframe(rows, use_container_width=True, hide_index=True)
                        fig3, ax3 = plt.subplots(figsize=(10, 4))
                        ax3.plot(frequencies, magnitudes, linewidth=1)
                        for n, observed, magnitude in harmonic_points:
                            ax3.plot(observed, magnitude, "ro", markersize=4)
                            ax3.annotate(f"H{n}", (observed, magnitude),
                                         xytext=(0, 7), textcoords="offset points",
                                         ha="center", fontsize=8)
                        ax3.set_xlim(0, 2000)
                        ax3.set_xlabel("Frequency (Hz)")
                        ax3.set_ylabel("FFT Magnitude")
                        ax3.set_title("Labeled Harmonics of Recorded Voice")
                        ax3.grid(True)
                        st.pyplot(fig3)
                        plt.close(fig3)
                        st.caption("FFT magnitudes are unnormalized spectral values, not decibels. Stronger harmonics can reflect vocal-tract resonances (formants).")

                    st.subheader("Build a Voice Waveform from Harmonics")
                    st.caption("Fourier analysis identifies the frequencies and harmonic strengths that make up a voice.")

                    st.write(
                        "Fourier analysis represents a complex sound as a sum of sinusoidal "
                        "components. Choose how many harmonics to keep and watch the "
                        "reconstructed waveform approach the original."
                    )

                    # Use a short, steady section of the same recording. The
                    # inverse FFT preserves each selected component's measured
                    # amplitude and phase, rather than inventing sine waves.
                    reconstruction_length = min(len(segment), int(0.12 * sample_rate))
                    midpoint = len(segment) // 2
                    short = segment[
                        midpoint - reconstruction_length // 2:
                        midpoint - reconstruction_length // 2 + reconstruction_length
                    ].copy()
                    short -= np.mean(short)
                    short_fft = np.fft.rfft(short)
                    short_frequencies = np.fft.rfftfreq(len(short), d=1 / sample_rate)

                    max_harmonics = max(1, min(9, int((sample_rate / 2) // f0)))
                    count = st.slider(
                        "Number of harmonics in reconstruction",
                        min_value=1,
                        max_value=max_harmonics,
                        value=min(4, max_harmonics),
                    )

                    # Keep Fourier bins close to the first N harmonics.
                    # A small band per harmonic accommodates natural pitch
                    # fluctuations and the finite length of the sample.
                    selected_bins = np.zeros(len(short_frequencies), dtype=bool)
                    bandwidth = max(12.0, 1.5 * sample_rate / len(short))
                    for harmonic_number in range(1, count + 1):
                        harmonic_frequency = harmonic_number * f0
                        selected_bins |= (
                            np.abs(short_frequencies - harmonic_frequency) <= bandwidth
                        )
                    reconstructed_fft = np.where(selected_bins, short_fft, 0.0)
                    reconstructed = np.fft.irfft(reconstructed_fft, n=len(short))

                    # Plot only a few pitch periods so the oscillations are visible.
                    display_length = min(len(short), max(2, int(0.035 * sample_rate)))
                    display_time = np.arange(display_length) / sample_rate * 1000.0
                    fig4, ax4 = plt.subplots(figsize=(10, 4))
                    ax4.plot(display_time, short[:display_length],
                             label="Original voice", linewidth=1.5)
                    ax4.plot(display_time, reconstructed[:display_length],
                             label=f"Reconstruction with {count} harmonic(s)",
                             linewidth=1.5, alpha=0.85)
                    ax4.set_xlabel("Time (milliseconds)")
                    ax4.set_ylabel("Amplitude (original recording units)")
                    ax4.set_title("Original Voice vs. Fourier Reconstruction")
                    ax4.legend()
                    ax4.grid(True)
                    st.pyplot(fig4)
                    plt.close(fig4)
                    st.caption(
                        "This is an approximation, not a perfect recreation: it retains "
                        "the measured FFT components near the selected harmonics, "
                        "including their phase. Other harmonics and noise are omitted. "
                        "Try 1 harmonic, then 2, 4, and 9."
                    )


# ---------------------------------------
# SHANNON SPECTRAL ENTROPY
# ---------------------------------------

        st.subheader("Shannon Spectral Entropy")
        st.caption("Measures how widely sound energy is distributed across frequencies.")

        # Calculate energy in each frequency bin
        spectral_energy = np.abs(np.fft.rfft(segment)) ** 2

    # Convert energy into probabilities
    total_energy = np.sum(spectral_energy)

    if total_energy > 0:
        probabilities = spectral_energy / total_energy

        # Remove zero probabilities
        probabilities = probabilities[probabilities > 0]

        # Shannon entropy formula
        entropy = -np.sum(
            probabilities * np.log2(probabilities)
        )

        st.write(f"Spectral Entropy: {entropy:.4f} bits")

        st.write(
            "Shannon entropy measures how spectral energy "
            "is distributed across the frequencies of a voice."
        )

    else:
        st.warning(
            "The recording contains no measurable spectral energy."
        )

# ------------------------------------------
# KL DIVERGENCE COMPARISON
# ------------------------------------------

st.subheader("KL Divergence: Comparing Two Voice Recordings")
st.caption("Measures differences between the frequency-energy distributions of the real and AI-generated voices.")

if uploaded_file is not None and comparison_file is not None:
    st.write("Both recordings are ready for KL divergence analysis.")
    # Read both recordings
    rate1, audio1 = wavfile.read(
        io.BytesIO(uploaded_file.getvalue())
    )
    rate2, audio2 = wavfile.read(
        io.BytesIO(comparison_file.getvalue())
    )
    # Convert stereo recordings to mono
    if audio1.ndim == 2:
        audio1 = audio1.mean(axis=1)

    if audio2.ndim == 2:
        audio2 = audio2.mean(axis=1)

    # Convert audio to floating-point numbers
    audio1 = audio1.astype(np.float64)
    audio2 = audio2.astype(np.float64)

    # Calculate the Fourier Transform of both recordings
    fft1 = np.abs(np.fft.rfft(audio1))
    fft2 = np.abs(np.fft.rfft(audio2))

    # Use the same frequency bins for both recordings
    num_bins = 1000
    freq1 = np.fft.rfftfreq(len(audio1), d=1/rate1)
    freq2 = np.fft.rfftfreq(len(audio2), d=1/rate2)

    # Create a common frequency scale
    max_freq = min(rate1, rate2) / 2
    common_freq = np.linspace(0, max_freq, num_bins)
    spectrum1 = np.interp(common_freq, freq1, fft1)
    spectrum2 = np.interp(common_freq, freq2, fft2)

    # Convert spectral magnitudes into energy
    energy1 = spectrum1 ** 2
    energy2 = spectrum2 ** 2

    # Calculate total spectral energy
    total1 = np.sum(energy1)
    total2 = np.sum(energy2)

    # Convert energy into probability distributions
    if total1 > 0 and total2 > 0:
        p = energy1 / total1
        q = energy2 / total2

        # Prevent zero probabilities
        epsilon = 1e-12
        p = p + epsilon
        q = q + epsilon
        p = p / np.sum(p)
        q = q / np.sum(q)

        # Calculate KL divergence in bits
        kl_divergence = np.sum(
            p * np.log2(p / q)
        )

        # Display KL divergence result
        st.write(f"KL Divergence: {kl_divergence:.4f} bits")

# ----------------------------------------
# PITCH TRACKING ANALYSIS
# ----------------------------------------

st.subheader("Pitch Tracking Over Time")
st.caption("Compares how the musical pitch of the human and AI-generated voices changes over time.")

if uploaded_file is not None:
    st.write("Analyzing how vocal pitch changes over time.")

    # Read the recording for pitch analysis
    pitch_rate, pitch_audio = wavfile.read(
        io.BytesIO(uploaded_file.getvalue())
    )

    # Convert stereo audio to mono
    if pitch_audio.ndim == 2:
        pitch_audio = pitch_audio.mean(axis=1)

    # Convert audio samples to floating-point numbers
    pitch_audio = pitch_audio.astype(np.float64)

    # Set the pitch analysis window and step sizes
    window_size = int(0.04 * pitch_rate)
    hop_size = int(0.01 * pitch_rate)
    pitch_times = []
    # Store the estimated fundamental frequencies
    pitch_values = []
    min_pitch = 150
    max_pitch = 350

    # Calculate the possible pitch periods
    min_period = int(pitch_rate / max_pitch)
    max_period = int(pitch_rate / min_pitch)

    # Analyze the recording in short windows
    for start in range(0, len(pitch_audio) - window_size, hop_size):
        # Extract a short section of audio
        segment = pitch_audio[start:start + window_size]

        # Remove the average amplitude
        segment = segment - np.mean(segment)

        # Measure the loudness of this audio segment
        rms = np.sqrt(np.mean(segment ** 2))
        if rms < 100:
            pitch_values.append(np.nan)
            pitch_times.append(start / pitch_rate)
            continue

        # Calculate autocorrelation
        correlation = np.correlate(
            segment, segment, mode="full"
        )
        correlation = correlation[len(correlation)//2:]

        # Find the strongest repeating pattern
        search = correlation[min_period:max_period]
        best_period = np.argmax(search) + min_period

        # Measure how strongly the sound repeats
        peak_strength = correlation[best_period]
        total_strength = correlation[0]

        # Calculate pitch confidence
        confidence = peak_strength / total_strength
        confidence_threshold = 0.6

        # Reject unreliable pitch measurements
        if confidence < confidence_threshold:
            pitch_values.append(np.nan)
            pitch_times.append(start / pitch_rate)
            continue

        # Calculate fundamental frequency in Hz
        fundamental_frequency = pitch_rate / best_period

        # Store the pitch and its time
        pitch_values.append(fundamental_frequency)
        pitch_times.append(start / pitch_rate)

    # Plot the estimated fundamental frequency
    fig, ax = plt.subplots()
    ax.plot(pitch_times, pitch_values)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Fundamental Frequency (Hz)")
    ax.set_title("Vocal Pitch Over Time")
    st.pyplot(fig)
    plt.close(fig)

# ============================================================
# REAL VOICE VS. AI-GENERATED VOICE — PITCH COMPARISON
# ============================================================

st.subheader("Real Voice vs. AI-Generated Voice — Pitch Comparison")

def extract_pitch_track(audio, sample_rate):
    """Estimate fundamental frequency over time using autocorrelation."""

    audio = np.asarray(audio, dtype=np.float64)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Analyze both recordings at the same sampling rate.
    target_rate = 8000

    if sample_rate != target_rate:
        from scipy.signal import resample_poly
        from math import gcd

        divisor = gcd(int(sample_rate), target_rate)

        audio = resample_poly(
            audio,
            target_rate // divisor,
            int(sample_rate) // divisor
        )

    # Normalize the audio amplitude.
    maximum = np.max(np.abs(audio))

    if maximum > 0:
        audio = audio / maximum

    window_size = int(0.08 * target_rate)
    hop_size = int(0.04 * target_rate)

    min_pitch = 150
    max_pitch = 350

    min_period = int(target_rate / max_pitch)
    max_period = int(target_rate / min_pitch)

    pitch_times = []
    pitch_values = []

    for start in range(0, len(audio) - window_size, hop_size):

        segment = audio[start:start + window_size]
        segment = segment - np.mean(segment)

        rms = np.sqrt(np.mean(segment ** 2))

        pitch_times.append(start / target_rate)

        if rms < 0.01:
            pitch_values.append(np.nan)
            continue

        correlation = np.correlate(
            segment, segment, mode="full"
        )

        correlation = correlation[len(correlation) // 2:]

        search = correlation[min_period:max_period]

        if len(search) == 0:
            pitch_values.append(np.nan)
            continue

        best_period = np.argmax(search) + min_period

        peak_strength = correlation[best_period]
        total_strength = correlation[0]

        if total_strength <= 0:
            pitch_values.append(np.nan)
            continue

        confidence = peak_strength / total_strength

        if confidence < 0.6:
            pitch_values.append(np.nan)
            continue

        fundamental_frequency = target_rate / best_period

        pitch_values.append(fundamental_frequency)

    return np.array(pitch_times), np.array(pitch_values)


if uploaded_file is not None and comparison_file is not None:

    # Analyze the real recording.
    real_rate, real_audio = wavfile.read(
        io.BytesIO(uploaded_file.getvalue())
    )

    real_times, real_pitches = extract_pitch_track(
        real_audio, real_rate
    )

    # Analyze the AI-generated recording.
    ai_rate, ai_audio = wavfile.read(
        io.BytesIO(comparison_file.getvalue())
    )

    ai_times, ai_pitches = extract_pitch_track(
        ai_audio, ai_rate
    )

    # Create the comparison graph.
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        real_times,
        real_pitches,
        color="blue",
        label="REAL VOICE — Human Recording",
        linewidth=1.5
    )

    ax.plot(
        ai_times,
        ai_pitches,
        color="orange",
        label="AI-GENERATED VOICE — Synthetic Recording",
        linewidth=1.5,
        alpha=0.8
    )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Fundamental Frequency (Hz)")

    ax.set_title(
        "Real Voice vs. AI-Generated Voice — Pitch Comparison"
    )

    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

# ============================================================
# SHANNON ENTROPY COMPARISON — REAL VOICE VS. AI VOICE
# ============================================================

st.subheader("Real Voice vs. AI-Generated Voice — Shannon Entropy")

def calculate_voice_entropy(audio, sample_rate):
    """Calculate spectral Shannon entropy from a selected audio segment."""

    audio = np.asarray(audio, dtype=np.float64)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if len(audio) < 256:
        return np.nan

    # Use the same segment-selection rule as our existing analysis.
    duration = len(audio) / sample_rate

    if duration >= 7.0:
        start = int(2.0 * sample_rate)
        end = int(7.0 * sample_rate)
    else:
        start = int(0.2 * len(audio))
        end = int(0.8 * len(audio))

    segment = audio[start:end]

    if len(segment) < 256:
        return np.nan

    # Remove the average amplitude.
    segment = segment - np.mean(segment)

    # Apply a window to reduce spectral leakage.
    segment = segment * np.hanning(len(segment))

    # Fourier transform and spectral energy.
    spectral_energy = np.abs(np.fft.rfft(segment)) ** 2

    total_energy = np.sum(spectral_energy)

    if total_energy <= 0:
        return np.nan

    # Convert energy into probabilities.
    probabilities = spectral_energy / total_energy

    # Remove zero probabilities before calculating logarithms.
    probabilities = probabilities[probabilities > 0]

    # Shannon entropy in bits.
    entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    return entropy


if uploaded_file is not None and comparison_file is not None:

    # Read the real voice recording.
    real_rate, real_audio = wavfile.read(
        io.BytesIO(uploaded_file.getvalue())
    )

    # Read the AI-generated recording.
    ai_rate, ai_audio = wavfile.read(
        io.BytesIO(comparison_file.getvalue())
    )

    # Calculate entropy for each voice.
    real_entropy = calculate_voice_entropy(
        real_audio, real_rate
    )

    ai_entropy = calculate_voice_entropy(
        ai_audio, ai_rate
    )

    if np.isfinite(real_entropy) and np.isfinite(ai_entropy):

        entropy_difference = ai_entropy - real_entropy

        st.write(
            f"REAL VOICE — Shannon Entropy: {real_entropy:.4f} bits"
        )

        st.write(
            f"AI-GENERATED VOICE — Shannon Entropy: {ai_entropy:.4f} bits"
        )

        st.write(
            f"Entropy Difference (AI − Real): "
            f"{entropy_difference:+.4f} bits"
        )

        st.caption(
            "Spectral entropy measures how concentrated or "
            "distributed the frequency energy is. "
            "A higher value does not necessarily indicate "
            "a human or AI-generated voice."
        )

    else:
        st.warning(
            "Unable to calculate Shannon entropy for one "
            "or both recordings."
        )

