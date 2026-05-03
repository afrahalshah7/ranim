from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import numpy as np
import tempfile
import os
import traceback
import librosa
import noisereduce as nr

from scipy.signal import butter, filtfilt, correlate

app = FastAPI(
    title="Ranim Voice Analysis API",
    version="5.0.0"
)

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
  <meta charset="UTF-8" />
  <title>Ranim Voice Analysis</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <style>
    body{
      font-family: Arial, sans-serif;
      background:#f8fafc;
      margin:0;
      padding:0;
      color:#1f2937;
    }
    .container{
      max-width:900px;
      margin:40px auto;
      background:white;
      border-radius:20px;
      box-shadow:0 10px 30px rgba(0,0,0,0.08);
      padding:30px;
    }
    h1{
      text-align:center;
      margin-bottom:10px;
      color:#1e3a8a;
    }
    p{
      text-align:center;
      color:#6b7280;
    }
    .upload-box{
      margin-top:30px;
      border:2px dashed #93c5fd;
      border-radius:16px;
      padding:30px;
      text-align:center;
      background:#f8fbff;
    }
    input[type=file]{
      margin:15px 0;
    }
    button{
      background:#60a5fa;
      color:white;
      border:none;
      padding:12px 24px;
      border-radius:12px;
      font-size:16px;
      cursor:pointer;
    }
    button:hover{
      background:#3b82f6;
    }
    .status{
      margin-top:20px;
      text-align:center;
      font-weight:bold;
    }
    .results{
      margin-top:30px;
      display:grid;
      grid-template-columns:repeat(auto-fit, minmax(220px, 1fr));
      gap:16px;
    }
    .card{
      background:#f9fafb;
      border-radius:16px;
      padding:18px;
      box-shadow:0 4px 10px rgba(0,0,0,0.04);
    }
    .label{
      font-size:13px;
      color:#6b7280;
      margin-bottom:8px;
    }
    .value{
      font-size:22px;
      font-weight:bold;
      color:#111827;
    }
    .notes{
      margin-top:25px;
      background:#eff6ff;
      border-left:5px solid #60a5fa;
      padding:16px;
      border-radius:12px;
      color:#1e3a8a;
    }
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
    <div class="notes" id="notes" style="display:none;"></div>
  </div>

  <script>
    async function uploadAudio() {
      const fileInput = document.getElementById("audioFile");
      const status = document.getElementById("status");
      const resultsDiv = document.getElementById("results");
      const notesDiv = document.getElementById("notes");

      resultsDiv.innerHTML = "";
      notesDiv.style.display = "none";

      if (!fileInput.files.length) {
        status.innerText = "Please choose an audio file first.";
        return;
      }

      const formData = new FormData();
      formData.append("file", fileInput.files[0]);

      status.innerText = "Analyzing...";

      try {
        const response = await fetch("/analyze_audio", {
          method: "POST",
          body: formData
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || "Analysis failed");
        }

        status.innerText = "Analysis completed successfully.";

        const r = data.results;

        const items = [
          ["Duration (sec)", r.duration_sec],
          ["Voiced Segment (sec)", r.voiced_segment_sec],
          ["Pitch Mean (Hz)", r.pitch_mean_hz],
          ["Pitch Median (Hz)", r.pitch_median_hz],
          ["Pitch Std (Hz)", r.pitch_std_hz],
          ["Pitch Min (Hz)", r.pitch_min_hz],
          ["Pitch Max (Hz)", r.pitch_max_hz],
          ["Jitter (%)", r.jitter_local],
          ["Shimmer (%)", r.shimmer_local],
          ["Shimmer (dB)", r.shimmer_local_db],
          ["HNR (dB)", r.hnr_db],
          ["RMS", r.rms],
          ["Unvoiced Fraction", r.fraction_unvoiced_frames],
          ["Intensity Raw", r.intensity_raw],
          ["Intensity Cleaned", r.intensity_cleaned],
          ["Intensity Voiced", r.intensity_voiced]
        ];

        resultsDiv.innerHTML = items.map(([label, value]) => `
          <div class="card">
            <div class="label">${label}</div>
            <div class="value">${formatValue(value)}</div>
          </div>
        `).join("");

        notesDiv.style.display = "block";
        notesDiv.innerHTML = `
          <strong>Notes:</strong><br/>
          Best result usually comes from a clean sustained vowel like <b>aaa</b> for 3 to 5 seconds.
        `;
      } catch (error) {
        status.innerText = "Error: " + error.message;
      }
    }

    function formatValue(value) {
      if (value === null || value === undefined || Number.isNaN(value)) {
        return "N/A";
      }
      if (typeof value === "number") {
        return value.toFixed(3);
      }
      return value;
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
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    except Exception:
        return None


def load_audio(file_path: str, target_sr: int = 16000):
    """
    More flexible loading for wav/mp3/m4a/flac/ogg.
    librosa handles more formats than soundfile alone.
    """
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    if y is None or len(y) == 0:
        raise ValueError("Audio file is empty")

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if np.max(np.abs(y)) <= 1e-12:
        raise ValueError("Audio is silent")

    return y, sr


def remove_dc(y: np.ndarray):
    return y - np.mean(y)


def bandpass_filter(y: np.ndarray, sr: int, lowcut=70.0, highcut=4000.0, order=4):
    nyq = 0.5 * sr
    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 0.999)

    if low >= high:
        return y.copy()

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, y).astype(np.float32)


def soft_limit(y: np.ndarray, max_peak=0.98):
    peak = np.max(np.abs(y))
    if peak <= max_peak or peak <= 1e-12:
        return y.astype(np.float32)
    scale = max_peak / peak
    return (y * scale).astype(np.float32)


def estimate_duration(y: np.ndarray, sr: int):
    return safe_float(len(y) / sr)


def estimate_rms(y: np.ndarray):
    if len(y) == 0:
        return None
    return safe_float(np.sqrt(np.mean(np.square(y))))


def estimate_relative_dbfs(y: np.ndarray):
    rms = estimate_rms(y)
    if rms is None or rms <= 0:
        return None
    return safe_float(20.0 * np.log10(rms + 1e-12))


def estimate_intensity_mean_abs(y: np.ndarray):
    if len(y) == 0:
        return None
    return safe_float(np.mean(np.abs(y)))


def get_noise_profile(y: np.ndarray, sr: int):
    n = int(sr * 0.25)
    if len(y) < n or n <= 0:
        return None
    return y[:n]


def clean_audio(y: np.ndarray, sr: int):
    """
    Hybrid cleaning:
    - remove DC
    - band-pass for human voice
    - noise reduction
    - soft limiting if clipping
    """
    y1 = remove_dc(y)
    y2 = bandpass_filter(y1, sr, lowcut=70.0, highcut=min(4000.0, sr / 2 - 100), order=4)

    noise_profile = get_noise_profile(y2, sr)

    try:
        if noise_profile is not None:
            y3 = nr.reduce_noise(
                y=y2,
                sr=sr,
                y_noise=noise_profile,
                stationary=False,
                prop_decrease=0.85
            )
        else:
            y3 = nr.reduce_noise(
                y=y2,
                sr=sr,
                stationary=False,
                prop_decrease=0.85
            )
    except Exception:
        y3 = y2.copy()

    y4 = soft_limit(y3, max_peak=0.98)
    y4 = np.nan_to_num(y4, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return y4


# =========================================================
# VOICED SEGMENT
# =========================================================

def moving_rms(y: np.ndarray, frame_size: int, hop_size: int):
    values = []
    for start in range(0, max(1, len(y) - frame_size + 1), hop_size):
        frame = y[start:start + frame_size]
        rms = np.sqrt(np.mean(frame * frame) + 1e-12)
        values.append(rms)
    return np.array(values, dtype=np.float32)


def find_longest_voiced_segment(y: np.ndarray, sr: int):
    frame_size = int(0.03 * sr)
    hop_size = int(0.01 * sr)

    if len(y) < frame_size:
        return y, 0, len(y)

    rms = moving_rms(y, frame_size, hop_size)
    if len(rms) == 0:
        return y, 0, len(y)

    threshold = max(np.percentile(rms, 40) * 0.75, 1e-5)
    voiced = rms > threshold

    best_start = 0
    best_end = 0
    current_start = None

    for i, v in enumerate(voiced):
        if v and current_start is None:
            current_start = i
        elif not v and current_start is not None:
            if (i - current_start) > (best_end - best_start):
                best_start, best_end = current_start, i
            current_start = None

    if current_start is not None:
        if (len(voiced) - current_start) > (best_end - best_start):
            best_start, best_end = current_start, len(voiced)

    if best_end <= best_start:
        return y, 0, len(y)

    start_sample = best_start * hop_size
    end_sample = min(len(y), best_end * hop_size + frame_size)

    seg = y[start_sample:end_sample]
    if len(seg) < int(0.15 * sr):
        return y, 0, len(y)

    return seg, start_sample, end_sample


# =========================================================
# F0 + PERIODS
# =========================================================

def estimate_f0_contour(y: np.ndarray, sr: int, fmin=75.0, fmax=500.0):
    """
    pYIN is usually more stable for human sustained vowels.
    """
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            sr=sr,
            fmin=fmin,
            fmax=fmax,
            frame_length=1024,
            hop_length=160
        )

        if f0 is None:
            return np.array([])

        f0 = np.asarray(f0, dtype=np.float64)
        f0 = f0[np.isfinite(f0) & (f0 > 0)]
        return f0
    except Exception:
        return np.array([])


def rising_zero_crossings(y: np.ndarray, sr: int):
    crossings = []

    for i in range(len(y) - 1):
        y1 = y[i]
        y2 = y[i + 1]

        if y1 <= 0 < y2:
            denom = (y2 - y1)
            frac = 0.0 if abs(denom) < 1e-12 else (-y1 / denom)
            t = (i + frac) / sr
            crossings.append(t)

    return np.array(crossings, dtype=np.float64)


def extract_periods_and_amplitudes(y_raw: np.ndarray, y_proc: np.ndarray, sr: int, f0_median=None):
    crossings = rising_zero_crossings(y_proc, sr)

    if len(crossings) < 3:
        return np.array([]), np.array([])

    periods = np.diff(crossings)

    if f0_median is not None and f0_median > 0:
        expected = 1.0 / f0_median
        min_period = max(0.002, expected * 0.5)
        max_period = min(0.02, expected * 1.5)
    else:
        min_period = 1.0 / 500.0
        max_period = 1.0 / 75.0

    valid_periods = []
    valid_amplitudes = []
    prev_period = None

    for i in range(len(periods)):
        p = periods[i]

        if not (min_period <= p <= max_period):
            prev_period = None
            continue

        if prev_period is not None:
            factor = max(p, prev_period) / max(min(p, prev_period), 1e-12)
            if factor > 1.3:
                prev_period = p
                continue

        start_sample = max(0, int(crossings[i] * sr))
        end_sample = min(len(y_raw), int(crossings[i + 1] * sr))

        if end_sample <= start_sample + 2:
            prev_period = p
            continue

        cycle = y_raw[start_sample:end_sample]
        amp = float(np.max(cycle) - np.min(cycle))

        if amp <= 1e-8:
            prev_period = p
            continue

        valid_periods.append(p)
        valid_amplitudes.append(amp)
        prev_period = p

    return np.array(valid_periods, dtype=np.float64), np.array(valid_amplitudes, dtype=np.float64)


# =========================================================
# JITTER / SHIMMER / HNR
# =========================================================

def jitter_local(periods: np.ndarray):
    if len(periods) < 3:
        return None

    mean_period = np.mean(periods)
    if mean_period <= 0:
        return None

    value = np.mean(np.abs(np.diff(periods))) / mean_period
    return safe_float(value * 100.0)


def shimmer_local(amplitudes: np.ndarray):
    if len(amplitudes) < 3:
        return None

    mean_amp = np.mean(amplitudes)
    if mean_amp <= 0:
        return None

    value = np.mean(np.abs(np.diff(amplitudes))) / mean_amp
    return safe_float(value * 100.0)


def shimmer_local_db(amplitudes: np.ndarray):
    if len(amplitudes) < 3:
        return None

    a1 = amplitudes[:-1]
    a2 = amplitudes[1:]

    valid = (a1 > 1e-12) & (a2 > 1e-12)
    if not np.any(valid):
        return None

    value = np.mean(np.abs(20.0 * np.log10(a2[valid] / a1[valid])))
    return safe_float(value)


def estimate_hnr(y: np.ndarray, sr: int, f0_median: float | None):
    """
    Simple autocorrelation-based HNR estimate on middle voiced signal.
    """
    if f0_median is None or f0_median <= 0 or len(y) < 512:
        return None

    lag = int(sr / f0_median)
    if lag <= 1 or lag >= len(y) // 2:
        return None

    y = y - np.mean(y)
    ac = correlate(y, y, mode="full")
    ac = ac[len(ac) // 2:]

    if ac[0] <= 1e-12 or lag >= len(ac):
        return None

    r = ac[lag] / (ac[0] + 1e-12)
    r = np.clip(r, 1e-6, 1 - 1e-6)

    return safe_float(10.0 * np.log10(r / (1.0 - r)))


# =========================================================
# MAIN ANALYSIS
# =========================================================

def analyze_signal(y: np.ndarray, sr: int):
    duration = estimate_duration(y, sr)

    raw_dbfs = estimate_relative_dbfs(y)
    raw_intensity = estimate_intensity_mean_abs(y)

    y_raw = remove_dc(y)
    y_clean = clean_audio(y_raw, sr)

    clean_dbfs = estimate_relative_dbfs(y_clean)
    clean_intensity = estimate_intensity_mean_abs(y_clean)

    voiced_seg_proc, seg_start, seg_end = find_longest_voiced_segment(y_clean, sr)
    voiced_seg_raw = y_raw[seg_start:seg_end]

    voiced_intensity = estimate_intensity_mean_abs(voiced_seg_proc)

    f0s = estimate_f0_contour(voiced_seg_proc, sr, fmin=75.0, fmax=500.0)

    if len(f0s) == 0:
        mean_f0 = None
        median_f0 = None
        std_f0 = None
        min_f0 = None
        max_f0 = None
    else:
        mean_f0 = safe_float(np.mean(f0s))
        median_f0 = safe_float(np.median(f0s))
        std_f0 = safe_float(np.std(f0s))
        min_f0 = safe_float(np.min(f0s))
        max_f0 = safe_float(np.max(f0s))

    periods, amplitudes = extract_periods_and_amplitudes(
        y_raw=voiced_seg_raw,
        y_proc=voiced_seg_proc,
        sr=sr,
        f0_median=median_f0
    )

    jitter_pct = jitter_local(periods)
    shimmer_pct = shimmer_local(amplitudes)
    shimmer_db = shimmer_local_db(amplitudes)
    hnr_db = estimate_hnr(voiced_seg_proc, sr, median_f0)

    voiced_duration = safe_float(len(voiced_seg_proc) / sr)
    fraction_unvoiced = None
    if duration and voiced_duration is not None and duration > 0:
        fraction_unvoiced = safe_float(max(0.0, 1.0 - (voiced_duration / duration)))

    rms = estimate_rms(y_raw)

    analysis = {
        "sample_rate": sr,
        "duration_seconds": duration,
        "voiced_segment_seconds": voiced_duration,

        "raw_relative_dbfs": raw_dbfs,
        "cleaned_relative_dbfs": clean_dbfs,

        "raw_intensity_mean_abs": raw_intensity,
        "cleaned_intensity_mean_abs": clean_intensity,
        "voiced_intensity_mean_abs": voiced_intensity,

        "mean_f0_hz": mean_f0,
        "median_f0_hz": median_f0,
        "std_f0_hz": std_f0,
        "min_f0_hz": min_f0,
        "max_f0_hz": max_f0,

        "jitter_local_pct": jitter_pct,
        "shimmer_local_pct": shimmer_pct,
        "shimmer_local_db": shimmer_db,
        "hnr_db": hnr_db,

        "rms": rms,
        "fraction_unvoiced_frames_estimate": fraction_unvoiced,
    }

    return analysis


def build_frontend_results(analysis: dict):
    return {
        "duration_sec": analysis.get("duration_seconds"),
        "voiced_segment_sec": analysis.get("voiced_segment_seconds"),

        "pitch_mean_hz": analysis.get("mean_f0_hz"),
        "pitch_median_hz": analysis.get("median_f0_hz"),
        "pitch_std_hz": analysis.get("std_f0_hz"),
        "pitch_min_hz": analysis.get("min_f0_hz"),
        "pitch_max_hz": analysis.get("max_f0_hz"),

        "jitter_local": analysis.get("jitter_local_pct"),
        "shimmer_local": analysis.get("shimmer_local_pct"),
        "shimmer_local_db": analysis.get("shimmer_local_db"),
        "hnr_db": analysis.get("hnr_db"),

        "rms": analysis.get("rms"),
        "fraction_unvoiced_frames": analysis.get("fraction_unvoiced_frames_estimate"),

        "relative_dbfs_raw": analysis.get("raw_relative_dbfs"),
        "relative_dbfs_cleaned": analysis.get("cleaned_relative_dbfs"),

        "intensity_raw": analysis.get("raw_intensity_mean_abs"),
        "intensity_cleaned": analysis.get("cleaned_intensity_mean_abs"),
        "intensity_voiced": analysis.get("voiced_intensity_mean_abs"),
    }


# =========================================================
# ENDPOINT
# =========================================================

@app.post("/analyze_audio")
async def analyze_audio(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    allowed_extensions = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload WAV, FLAC, OGG, MP3, or M4A."
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        y, sr = load_audio(temp_path, target_sr=16000)
        analysis = analyze_signal(y, sr)
        results = build_frontend_results(analysis)

        return {
            "success": True,
            "app_version": "v6.0.0",
            "filename": file.filename,
            "results": results,
            "notes": {
                "best_input": "Use a clean sustained vowel like aaa for 3 to 5 seconds.",
                "important_note": "This is an acoustic analysis aid, not a medical diagnosis.",
                "pipeline": "DC removal -> band-pass filter -> noise reduction -> soft limiting -> voiced segment -> F0/jitter/shimmer/HNR"
            }
        }

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(e)}")

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
