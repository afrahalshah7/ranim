from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import numpy as np
import tempfile
import os
import traceback
import librosa
import noisereduce as nr
from scipy.signal import butter, filtfilt

# =========================================================
# APP
# =========================================================

app = FastAPI(title="Ranim Voice Analysis API", version="9.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# UI PAGE
# =========================================================

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Ranim Voice Analysis</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <style>
    body{font-family:Arial,sans-serif;background:#f8fafc;margin:0;padding:0;color:#1f2937;}
    .container{max-width:900px;margin:40px auto;background:white;border-radius:20px;
               box-shadow:0 10px 30px rgba(0,0,0,0.08);padding:30px;}
    h1{text-align:center;margin-bottom:10px;color:#1e3a8a;}
    p{text-align:center;color:#6b7280;}
    .upload-box{margin-top:30px;border:2px dashed #93c5fd;border-radius:16px;
                padding:30px;text-align:center;background:#f8fbff;}
    input[type=file]{margin:15px 0;}
    button{background:#60a5fa;color:white;border:none;padding:12px 24px;
           border-radius:12px;font-size:16px;cursor:pointer;}
    button:hover{background:#3b82f6;}
    .status{margin-top:20px;text-align:center;font-weight:bold;}
    .results{margin-top:30px;display:grid;
             grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;}
    .card{background:#f9fafb;border-radius:16px;padding:18px;
          box-shadow:0 4px 10px rgba(0,0,0,0.04);}
    .label{font-size:13px;color:#6b7280;margin-bottom:8px;}
    .value{font-size:22px;font-weight:bold;color:#111827;}
    .notes{margin-top:25px;background:#eff6ff;border-left:5px solid #60a5fa;
           padding:16px;border-radius:12px;color:#1e3a8a;}
  </style>
</head>
<body>
<div class="container">
  <h1>Ranim Voice Analysis</h1>
  <p>Upload a voice file and get detailed acoustic analysis</p>
  <div class="upload-box">
    <input type="file" id="audioFile" accept=".wav,.mp3,.m4a,.flac,.ogg"/>
    <br/>
    <button onclick="uploadAudio()">Analyze Audio</button>
    <div class="status" id="status"></div>
  </div>
  <div class="results" id="results"></div>
  <div class="notes"   id="notes"   style="display:none;"></div>
</div>
<script>
  async function uploadAudio() {
    const fileInput  = document.getElementById("audioFile");
    const status     = document.getElementById("status");
    const resultsDiv = document.getElementById("results");
    const notesDiv   = document.getElementById("notes");
    resultsDiv.innerHTML = "";
    notesDiv.style.display = "none";
    if (!fileInput.files.length) {
      status.innerText = "Please choose an audio file first."; return;
    }
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    status.innerText = "Analyzing...";
    try {
      const response = await fetch("/analyze_audio",{method:"POST",body:formData});
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Analysis failed");
      status.innerText = "Analysis completed successfully.";
      const r = data.results;
      const items = [
        ["Duration (sec)",       r.duration_sec],
        ["Voiced Segment (sec)", r.voiced_segment_sec],
        ["Pitch Mean (Hz)",      r.pitch_mean_hz],
        ["Pitch Median (Hz)",    r.pitch_median_hz],
        ["Pitch Std (Hz)",       r.pitch_std_hz],
        ["Pitch Min (Hz)",       r.pitch_min_hz],
        ["Pitch Max (Hz)",       r.pitch_max_hz],
        ["Jitter (%)",           r.jitter_local],
        ["Shimmer (%)",          r.shimmer_local],
        ["Shimmer (dB)",         r.shimmer_local_db],
        ["HNR (dB)",             r.hnr_db],
        ["RMS",                  r.rms],
        ["Unvoiced Fraction",    r.fraction_unvoiced_frames],
        ["Intensity Raw",        r.intensity_raw],
        ["Intensity Cleaned",    r.intensity_cleaned],
        ["Intensity Voiced",     r.intensity_voiced],
      ];
      resultsDiv.innerHTML = items.map(([label,value])=>`
        <div class="card">
          <div class="label">${label}</div>
          <div class="value">${fmt(value)}</div>
        </div>`).join("");
      notesDiv.style.display = "block";
      notesDiv.innerHTML =
        "<strong>Notes:</strong><br/>Best result: clean sustained vowel " +
        "<b>aaa</b> for 3–5 seconds.";
    } catch(e){ status.innerText = "Error: " + e.message; }
  }
  function fmt(v){
    if(v===null||v===undefined||Number.isNaN(v)) return "N/A";
    if(typeof v==="number") return v.toFixed(3);
    return v;
  }
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE


# =========================================================
# HELPERS
# =========================================================

def safe_float(value):
    try:
        v = float(value)
        return None if (np.isnan(v) or np.isinf(v)) else v
    except Exception:
        return None


def load_audio(file_path: str, target_sr: int = 44100):
    """
    Load audio at 44100 Hz.
    Fallback for m4a / unusual formats.
    Normalise quiet recordings (peak < 0.1).
    """
    try:
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    except Exception:
        import soundfile as sf
        y, sr = sf.read(file_path, always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != target_sr:
            y = librosa.resample(y.astype(np.float32),
                                 orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    if y is None or len(y) == 0:
        raise ValueError("Audio file is empty")

    y    = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    peak = float(np.max(np.abs(y)))

    if peak <= 1e-12:
        raise ValueError("Audio is silent")
    if peak < 0.1:
        y = y / peak * 0.7

    return y, sr


def remove_dc(y: np.ndarray):
    return (y - np.mean(y)).astype(np.float32)


def bandpass_filter(y: np.ndarray, sr: int,
                    lowcut=70.0, highcut=4000.0, order=4):
    nyq  = 0.5 * sr
    low  = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 0.999)
    if low >= high:
        return y.copy()
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, y).astype(np.float32)


def soft_limit(y: np.ndarray, max_peak=0.98):
    peak = float(np.max(np.abs(y)))
    if peak <= max_peak or peak <= 1e-12:
        return y.astype(np.float32)
    return (y * (max_peak / peak)).astype(np.float32)


def estimate_duration(y, sr):        return safe_float(len(y) / sr)
def estimate_rms(y):
    return safe_float(np.sqrt(np.mean(np.square(y)))) if len(y) else None
def estimate_dbfs(y):
    r = estimate_rms(y)
    return safe_float(20.0 * np.log10(r + 1e-12)) if r else None
def estimate_intensity(y):
    return safe_float(np.mean(np.abs(y))) if len(y) else None


def get_noise_profile(y: np.ndarray, sr: int):
    n = int(sr * 0.25)
    return y[:n] if len(y) >= n > 0 else None


def clean_audio(y: np.ndarray, sr: int):
    """
    DC → bandpass (70–4000 Hz) → gentle NR (prop=0.6) → soft limit.
    prop_decrease = 0.6 preserves micro-variations for Jitter/Shimmer.
    """
    y1 = remove_dc(y)
    y2 = bandpass_filter(y1, sr, lowcut=70.0,
                         highcut=min(4000.0, sr / 2 - 100), order=4)
    noise = get_noise_profile(y2, sr)
    try:
        y3 = nr.reduce_noise(y=y2, sr=sr,
                             y_noise=noise if noise is not None else y2,
                             stationary=True, prop_decrease=0.6)
    except Exception:
        y3 = y2.copy()

    y4 = soft_limit(y3)
    return np.nan_to_num(y4, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# =========================================================
# VOICED SEGMENT
# =========================================================

def moving_rms(y, frame_size, hop_size):
    return np.array([
        float(np.sqrt(np.mean(y[s:s+frame_size]**2) + 1e-12))
        for s in range(0, max(1, len(y)-frame_size+1), hop_size)
    ], dtype=np.float32)


def find_longest_voiced_segment(y: np.ndarray, sr: int):
    """
    Longest voiced segment via RMS energy.
    Threshold: percentile-20 × 0.5 (sensitive for quiet recordings).
    Gap-fill: silent gaps < 0.3 s are bridged.
    """
    frame_size = int(0.03 * sr)
    hop_size   = int(0.01 * sr)

    if len(y) < frame_size:
        return y, 0, len(y)

    rms_arr   = moving_rms(y, frame_size, hop_size)
    threshold = max(float(np.percentile(rms_arr, 20)) * 0.5, 1e-6)
    voiced    = list(rms_arr > threshold)

    # Bridge short gaps
    min_gap = max(1, int(0.3 * sr / hop_size))
    i = 0
    while i < len(voiced):
        if not voiced[i]:
            g0 = i
            while i < len(voiced) and not voiced[i]:
                i += 1
            if (i - g0) < min_gap:
                for j in range(g0, i):
                    voiced[j] = True
        else:
            i += 1

    voiced        = np.array(voiced)
    best_s = best_e = 0
    cur = None

    for i, v in enumerate(voiced):
        if v and cur is None:
            cur = i
        elif not v and cur is not None:
            if (i - cur) > (best_e - best_s):
                best_s, best_e = cur, i
            cur = None

    if cur is not None and (len(voiced) - cur) > (best_e - best_s):
        best_s, best_e = cur, len(voiced)

    if best_e <= best_s:
        return y, 0, len(y)

    s = best_s * hop_size
    e = min(len(y), best_e * hop_size + frame_size)
    seg = y[s:e]

    return (y, 0, len(y)) if len(seg) < int(0.15*sr) else (seg, s, e)


# =========================================================
# F0 ESTIMATION
# =========================================================

def estimate_f0_contour(y: np.ndarray, sr: int,
                         fmin=75.0, fmax=500.0,
                         remove_outliers=True):
    """
    pYIN F0 estimation (Mauch & Dixon 2014).

    remove_outliers=True  → for statistics display (mean/median/std)
    remove_outliers=False → for Jitter calculation (keep real variations)
    """
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, sr=sr, fmin=fmin, fmax=fmax,
            frame_length=2048, hop_length=256)

        if f0 is None:
            return np.array([])

        f0 = np.asarray(f0, dtype=np.float64)
        if voiced_flag is not None:
            f0 = f0[voiced_flag & np.isfinite(f0) & (f0 > 0)]
        else:
            f0 = f0[np.isfinite(f0) & (f0 > 0)]

        if remove_outliers and len(f0) > 5:
            med = float(np.median(f0))
            f0  = f0[(f0 >= med * 0.7) & (f0 <= med * 1.3)]

        return f0
    except Exception:
        return np.array([])


# =========================================================
# GCI DETECTION — GLOTTAL CLOSURE INSTANTS
# =========================================================

def find_glottal_periods(y: np.ndarray, sr: int, f0_median: float):
    """
    Windowed Autocorrelation GCI detection.
    Mimics Praat's PointProcess (periodic, cc).

    FIX v9.1 — Two key improvements:
      1. Tight search window ±20% around expected F0 lag
         (was ±30–40%) to avoid capturing subharmonics and
         octave errors that inflate Jitter dramatically.
      2. Post-detection period filtering: keep only periods
         within ±20% of the expected period (1/F0), then
         apply max_factor. This ensures clean periods even
         for voices with high shimmer or instability.

    Window size stays at 2× period — diagnostic tests showed
    this gives the lowest Jitter error across all F0 ranges.

    Reference: Boersma (1993).
    """
    if f0_median is None or f0_median <= 0:
        return []

    period_s    = int(sr / f0_median)
    expected_p  = 1.0 / f0_median            # expected period in seconds
    window_size = max(period_s * 2, 512)
    hop_size    = max(period_s // 4, 1)
    r_threshold = 0.20
    pulses      = []

    for start in range(0, len(y) - window_size, hop_size):
        frame = y[start: start + window_size].astype(np.float64)
        frame -= np.mean(frame)

        if np.sum(frame**2) < 1e-12:
            continue

        ac = np.correlate(frame, frame, mode='full')
        ac = ac[len(ac)//2:]
        if ac[0] <= 1e-12:
            continue
        ac /= ac[0]

        # FIX v9.1: tight ±20% search avoids subharmonics
        lag_min = max(2, int(sr / (f0_median * 1.2)))
        lag_max = min(len(ac)-1, int(sr / (f0_median * 0.8)))
        if lag_min >= lag_max:
            continue

        region   = ac[lag_min:lag_max]
        peak_idx = int(np.argmax(region))
        peak_lag = lag_min + peak_idx
        r        = float(ac[peak_lag])

        if r < r_threshold:
            continue

        # Parabolic interpolation
        if 0 < peak_lag < len(ac)-1:
            a, b, g = ac[peak_lag-1], ac[peak_lag], ac[peak_lag+1]
            d = a - 2*b + g
            refined = peak_lag - 0.5*(g-a)/d if abs(d) > 1e-12 else float(peak_lag)
        else:
            refined = float(peak_lag)

        pulses.append(((start + window_size/2.0)/sr, refined/sr))

    if len(pulses) < 2:
        return pulses

    filtered = [pulses[0]]
    min_gap  = (1.0 / f0_median) * 0.5
    for pt, ps in pulses[1:]:
        if pt - filtered[-1][0] >= min_gap:
            filtered.append((pt, ps))

    return filtered


# =========================================================
# AMPLITUDE EXTRACTION — RMS per cycle
# =========================================================

def extract_cycle_amplitudes_rms(y: np.ndarray, sr: int,
                                  pulse_data: list, f0_median: float):
    """
    RMS amplitude per glottal cycle.
    Praat uses RMS (not Peak-to-Peak) for Shimmer.
    Reference: Titze (1994).
    """
    if len(pulse_data) < 2 or f0_median <= 0:
        return np.array([])

    hp   = int((sr / f0_median) / 2)
    amps = []

    for i in range(len(pulse_data)-1):
        t1, _ = pulse_data[i]
        t2, _ = pulse_data[i+1]
        s = max(0, int(t1*sr) - hp//4)
        e = min(len(y), int(t2*sr) + hp//4)
        if e - s < 3:
            continue
        rms = float(np.sqrt(np.mean(y[s:e].astype(np.float64)**2)))
        if rms > 1e-10:
            amps.append(rms)

    return np.array(amps, dtype=np.float64)


# =========================================================
# JITTER — Hybrid: GCI periods refined by pYIN
# =========================================================

def compute_jitter_hybrid(pulse_data: list,
                           f0_contour_raw: np.ndarray,
                           sr: int,
                           min_period=0.0001, max_period=0.02,
                           max_factor=1.3):
    """
    Hybrid Jitter — closest to Praat without Parselmouth.

    Strategy:
      1. GCI autocorrelation gives pulse timestamps
      2. pYIN F0 (unsmoothed) gives period per frame
      3. We use GCI periods (refined_lag/sr) as primary source
         but SCALE them so their mean matches pYIN mean period.
         This corrects the systematic bias in GCI period estimation
         while preserving the cycle-to-cycle variation pattern.

    Formula (Boersma 1993):
        Jitter = mean|T_i - T_{i-1}| / mean(T_i)

    References:
    Boersma, P. (1993). Accurate short-term analysis.
    Titze, I.R. (1994). Principles of Voice Production.
    """
    # ── Primary: GCI periods ──────────────────────────────
    if len(pulse_data) >= 3:
        gci_periods = np.array([ps for _, ps in pulse_data])
        valid = (gci_periods >= min_period) & (gci_periods <= max_period)
        gci_periods = gci_periods[valid]

        if len(gci_periods) >= 3:
            # FIX v9.1: narrow periods to ±20% of pYIN expected period
            # This removes subharmonic/octave errors before max_factor runs
            if f0_contour_raw is not None and len(f0_contour_raw) > 3:
                expected_p = float(np.median(
                    1.0 / f0_contour_raw[f0_contour_raw > 0]))
            else:
                expected_p = float(np.median(gci_periods))
            gci_periods = gci_periods[
                (gci_periods >= expected_p * 0.80) &
                (gci_periods <= expected_p * 1.20)]

            if len(gci_periods) < 3:
                gci_periods = np.array([])  # fall through to pYIN fallback

        if len(gci_periods) >= 3:
            # max_factor filter
            clean_gci = [gci_periods[0]]
            for p in gci_periods[1:]:
                lo, hi = min(p, clean_gci[-1]), max(p, clean_gci[-1])
                if hi / (lo + 1e-12) <= max_factor:
                    clean_gci.append(p)
            clean_gci = np.array(clean_gci)

            if len(clean_gci) >= 3:
                mean_gci = float(np.mean(clean_gci))

                # ── Scale correction using pYIN mean ────────────
                if f0_contour_raw is not None and len(f0_contour_raw) > 3:
                    pyin_periods = 1.0 / f0_contour_raw[f0_contour_raw > 0]
                    pyin_valid   = pyin_periods[
                        (pyin_periods >= min_period) &
                        (pyin_periods <= max_period)]

                    if len(pyin_valid) >= 3:
                        mean_pyin = float(np.mean(pyin_valid))
                        if mean_gci > 1e-12:
                            scale      = mean_pyin / mean_gci
                            clean_gci  = clean_gci * scale
                            mean_final = mean_pyin
                        else:
                            mean_final = mean_gci
                    else:
                        mean_final = mean_gci
                else:
                    mean_final = mean_gci

                if mean_final <= 1e-12:
                    return None

                jitter = float(np.mean(np.abs(np.diff(clean_gci)))) / mean_final
                return safe_float(jitter * 100.0)

    # ── Fallback: pure pYIN periods ───────────────────────
    if f0_contour_raw is not None and len(f0_contour_raw) >= 3:
        periods = 1.0 / f0_contour_raw[f0_contour_raw > 0]
        valid   = (periods >= min_period) & (periods <= max_period)
        periods = periods[valid]

        if len(periods) < 3:
            return None

        clean = [periods[0]]
        for p in periods[1:]:
            lo, hi = min(p, clean[-1]), max(p, clean[-1])
            if hi / (lo + 1e-12) <= max_factor:
                clean.append(p)

        clean = np.array(clean)
        if len(clean) < 3:
            return None

        mean_p = float(np.mean(clean))
        if mean_p <= 1e-12:
            return None

        return safe_float(float(np.mean(np.abs(np.diff(clean)))) / mean_p * 100.0)

    return None


# =========================================================
# SHIMMER — Praat-equivalent formula
# =========================================================

def compute_shimmer_local(amplitudes: np.ndarray, max_factor=1.6):
    """
    Shimmer (local) = mean|ΔA| / mean A
    Shimmer (dB)    = mean|20·log10(A_{i+1}/A_i)|

    Titze (1994): max_factor = 1.6
    """
    if len(amplitudes) < 3:
        return None, None

    clean = [amplitudes[0]]
    for a in amplitudes[1:]:
        lo, hi = min(a, clean[-1]), max(a, clean[-1])
        if hi / (lo + 1e-12) <= max_factor:
            clean.append(a)

    clean = np.array(clean)
    if len(clean) < 3:
        return None, None

    mean_a = float(np.mean(clean))
    if mean_a <= 1e-12:
        return None, None

    shimmer_pct = float(np.mean(np.abs(np.diff(clean)))) / mean_a * 100.0

    a1, a2 = clean[:-1], clean[1:]
    valid  = (a1 > 1e-12) & (a2 > 1e-12)
    shimmer_db = (float(np.mean(np.abs(20.0 * np.log10(a2[valid]/a1[valid]))))
                  if np.any(valid) else None)

    return safe_float(shimmer_pct), safe_float(shimmer_db)


# =========================================================
# HNR — Cross-Correlation with Window Correction (Praat cc)
# =========================================================

def compute_hnr_cc(y: np.ndarray, sr: int, f0_median: float,
                   time_step=0.01, silence_threshold=0.1,
                   periods_per_window=4.5):
    """
    HNR via Cross-Correlation with Window Autocorrelation Correction.

    This replicates Praat's "To Harmonicity (cc)" more accurately
    than simple autocorrelation by correcting for the Hanning window
    effect — the key difference from earlier versions.

    Algorithm:
      1. Hanning-windowed frames (4.5 periods)
      2. AC of signal   → ac_signal
      3. AC of window   → ac_window
      4. r_corrected    = ac_signal / (ac_window · ac_signal[0]/ac_window[0])
         This removes the window's distortion of the peak shape.
      5. Find peak near F0 lag (±40%)
      6. Parabolic interpolation refines r
      7. HNR_frame = 10·log10(r / (1−r))
      8. Final HNR  = MEDIAN across voiced frames

    Reference:
    Boersma, P. (1993). Accurate short-term analysis of the fundamental
    frequency and the harmonics-to-noise ratio of a sampled sound.
    Proceedings of the Institute of Phonetic Sciences, 17, 97–110.
    """
    if f0_median is None or f0_median <= 0 or len(y) < 512:
        return None

    period_s    = int(sr / f0_median)
    window_size = max(int(periods_per_window * period_s * 2), 2048)
    hop_size    = max(1, int(time_step * sr))
    hnr_vals    = []

    for start in range(0, len(y) - window_size, hop_size):
        frame = y[start: start + window_size].astype(np.float64)
        frame -= np.mean(frame)

        window   = np.hanning(len(frame))
        windowed = frame * window

        if np.sum(windowed**2) < 1e-12:
            continue

        # AC of signal
        ac_sig = np.correlate(windowed, windowed, mode='full')
        ac_sig = ac_sig[len(ac_sig)//2:]

        # AC of window (Praat's cc correction)
        ac_win = np.correlate(window, window, mode='full')
        ac_win = ac_win[len(ac_win)//2:]

        if ac_sig[0] <= 1e-12 or ac_win[0] <= 1e-12:
            continue

        # Window-corrected normalised AC
        scale = ac_sig[0] / ac_win[0]
        with np.errstate(divide='ignore', invalid='ignore'):
            r_norm = np.where(
                ac_win > 1e-12,
                ac_sig / (ac_win * scale),
                0.0
            )

        # Search ±40%
        lag_min = max(2, int(sr / (f0_median * 1.4)))
        lag_max = min(len(r_norm)-1, int(sr / (f0_median * 0.6)))
        if lag_min >= lag_max:
            continue

        region   = r_norm[lag_min:lag_max]
        peak_lag = lag_min + int(np.argmax(region))
        r        = float(r_norm[peak_lag])

        if r < silence_threshold:
            continue

        # Parabolic refinement
        if 0 < peak_lag < len(r_norm)-1:
            a, b, g = r_norm[peak_lag-1], r_norm[peak_lag], r_norm[peak_lag+1]
            d = a - 2*b + g
            if abs(d) > 1e-12:
                delta = float(np.clip(-0.5*(g-a)/d, -0.5, 0.5))
                r = float(b - 0.25*(g-a)*delta)

        r   = float(np.clip(r, 1e-6, 0.9999))
        hnr = 10.0 * np.log10(r / (1.0 - r))

        if hnr < -20.0:
            continue

        hnr_vals.append(hnr)

    if not hnr_vals:
        return None

    return safe_float(float(np.median(hnr_vals)))


# =========================================================
# MAIN ANALYSIS PIPELINE
# =========================================================

def analyze_signal(y: np.ndarray, sr: int):
    """
    Full acoustic analysis pipeline — v9.0

     1.  Duration + raw intensity
     2.  DC removal → bandpass → gentle NR → soft limit
     3.  Longest voiced segment (gap-fill)
     4.  pYIN F0 (outlier-filtered)  → statistics
     5.  pYIN F0 (raw, no filter)    → Jitter input
     6.  GCI via windowed AC (thr 0.2)
     7.  RMS amplitudes per cycle
     8.  Jitter HYBRID (GCI + pYIN scale correction)   — Boersma 1993
     9.  Shimmer (local / dB)                           — Titze 1994
    10.  HNR  cross-correlation + window correction     — Boersma 1993
    11.  Result dict
    """
    duration   = estimate_duration(y, sr)
    raw_dbfs   = estimate_dbfs(y)
    raw_intens = estimate_intensity(y)

    y_raw   = remove_dc(y)
    y_clean = clean_audio(y_raw, sr)

    clean_dbfs   = estimate_dbfs(y_clean)
    clean_intens = estimate_intensity(y_clean)

    voiced_proc, seg_s, seg_e = find_longest_voiced_segment(y_clean, sr)
    voiced_raw  = y_raw[seg_s:seg_e]
    voiced_int  = estimate_intensity(voiced_proc)

    # ── F0 (outlier-filtered) for statistics ─────────────
    f0s = estimate_f0_contour(voiced_proc, sr,
                               fmin=75.0, fmax=500.0,
                               remove_outliers=True)

    if len(f0s) == 0:
        mean_f0 = median_f0 = std_f0 = min_f0 = max_f0 = None
    else:
        mean_f0   = safe_float(np.mean(f0s))
        median_f0 = safe_float(np.median(f0s))
        std_f0    = safe_float(np.std(f0s))
        min_f0    = safe_float(np.min(f0s))
        max_f0    = safe_float(np.max(f0s))

    # ── F0 (raw, no outlier removal) for Jitter ──────────
    f0_raw = estimate_f0_contour(voiced_proc, sr,
                                  fmin=75.0, fmax=500.0,
                                  remove_outliers=False)

    jitter_pct = shimmer_pct = shimmer_db = hnr_db = None

    if median_f0 is not None and median_f0 > 0:

        # ── GCI detection ────────────────────────────────
        pulse_data = find_glottal_periods(voiced_proc, sr, median_f0)

        # ── Jitter HYBRID ────────────────────────────────
        jitter_pct = compute_jitter_hybrid(
            pulse_data, f0_raw, sr)

        # ── Shimmer from RMS amplitudes ───────────────────
        if len(pulse_data) >= 3:
            amps = extract_cycle_amplitudes_rms(
                voiced_raw, sr, pulse_data, median_f0)
            if len(amps) >= 3:
                shimmer_pct, shimmer_db = compute_shimmer_local(amps)

        # ── HNR cross-correlation + window correction ─────
        hnr_db = compute_hnr_cc(
            voiced_raw, sr, median_f0,
            time_step=0.01,
            silence_threshold=0.1,
            periods_per_window=4.5)

    voiced_dur    = safe_float(len(voiced_proc) / sr)
    frac_unvoiced = None
    if duration and voiced_dur is not None and duration > 0:
        frac_unvoiced = safe_float(max(0.0, 1.0 - voiced_dur / duration))

    return {
        "sample_rate":            sr,
        "duration_seconds":       duration,
        "voiced_segment_seconds": voiced_dur,

        "raw_relative_dbfs":          raw_dbfs,
        "cleaned_relative_dbfs":      clean_dbfs,
        "raw_intensity_mean_abs":     raw_intens,
        "cleaned_intensity_mean_abs": clean_intens,
        "voiced_intensity_mean_abs":  voiced_int,

        "mean_f0_hz":   mean_f0,
        "median_f0_hz": median_f0,
        "std_f0_hz":    std_f0,
        "min_f0_hz":    min_f0,
        "max_f0_hz":    max_f0,

        "jitter_local_pct":  jitter_pct,
        "shimmer_local_pct": shimmer_pct,
        "shimmer_local_db":  shimmer_db,
        "hnr_db":            hnr_db,

        "rms":                               estimate_rms(y_raw),
        "fraction_unvoiced_frames_estimate": frac_unvoiced,
    }


def build_results(analysis: dict):
    return {
        "duration_sec":       analysis.get("duration_seconds"),
        "voiced_segment_sec": analysis.get("voiced_segment_seconds"),

        "pitch_mean_hz":   analysis.get("mean_f0_hz"),
        "pitch_median_hz": analysis.get("median_f0_hz"),
        "pitch_std_hz":    analysis.get("std_f0_hz"),
        "pitch_min_hz":    analysis.get("min_f0_hz"),
        "pitch_max_hz":    analysis.get("max_f0_hz"),

        "jitter_local":     analysis.get("jitter_local_pct"),
        "shimmer_local":    analysis.get("shimmer_local_pct"),
        "shimmer_local_db": analysis.get("shimmer_local_db"),
        "hnr_db":           analysis.get("hnr_db"),

        "rms":                      analysis.get("rms"),
        "fraction_unvoiced_frames": analysis.get(
            "fraction_unvoiced_frames_estimate"),

        "relative_dbfs_raw":     analysis.get("raw_relative_dbfs"),
        "relative_dbfs_cleaned": analysis.get("cleaned_relative_dbfs"),
        "intensity_raw":         analysis.get("raw_intensity_mean_abs"),
        "intensity_cleaned":     analysis.get("cleaned_intensity_mean_abs"),
        "intensity_voiced":      analysis.get("voiced_intensity_mean_abs"),
    }


# =========================================================
# ENDPOINT
# =========================================================

@app.post("/analyze_audio")
async def analyze_audio(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".wav", ".flac", ".ogg", ".mp3", ".m4a"}:
        raise HTTPException(400,
            "Unsupported file type. Use WAV, FLAC, OGG, MP3, or M4A.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name

        y, sr    = load_audio(temp_path, target_sr=44100)
        analysis = analyze_signal(y, sr)
        results  = build_results(analysis)

        return {
            "success":     True,
            "app_version": "v9.1.0",
            "filename":    file.filename,
            "results":     results,
            "notes": {
                "best_input":     "Clean sustained vowel aaa for 3–5 seconds.",
                "important_note": "Acoustic analysis aid — not a medical diagnosis.",
                "pipeline": (
                    "DC → bandpass → gentle NR → soft limit → "
                    "voiced segment (gap-fill) → "
                    "pYIN F0 (filtered for stats, raw for Jitter) → "
                    "GCI adaptive window (2–6× period, F0-adaptive thr) → RMS amplitudes → "
                    "Jitter hybrid GCI+pYIN scale (Boersma 1993) → "
                    "Shimmer RMS (Titze 1994) → "
                    "HNR cross-correlation + window correction (Boersma 1993)"
                ),
                "references": [
                    "Boersma, P. (1993). Accurate short-term analysis of the "
                    "fundamental frequency and the HNR. "
                    "Proc. Institute of Phonetic Sciences, 17, 97-110.",
                    "Titze, I.R. (1994). Principles of Voice Production. "
                    "Prentice Hall.",
                    "Mauch & Dixon (2014). pYIN. ICASSP."
                ]
            }
        }

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(500, f"Audio analysis failed: {str(e)}")

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass