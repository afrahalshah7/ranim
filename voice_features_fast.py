import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, butter, filtfilt, get_window

def load_mono(path, target_sr=16000):
    x, sr = sf.read(path, always_2d=False)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    x = x.astype(np.float64)
    if sr != target_sr:
        g = np.gcd(sr, target_sr)
        x = resample_poly(x, target_sr // g, sr // g)
        sr = target_sr
    x = x / (np.max(np.abs(x)) + 1e-12)
    return x, sr

def bandpass(x, sr, low=70.0, high=3500.0, order=4):
    nyq = 0.5 * sr
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x)

def frame_signal(x, frame_len, hop_len):
    n = len(x)
    if n < frame_len:
        x = np.pad(x, (0, frame_len - n))
        n = len(x)

    n_frames = 1 + (n - frame_len) // hop_len
    return np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, frame_len),
        strides=(x.strides[0] * hop_len, x.strides[0]),
        writeable=False
    )

def pitch_ac_track(x, sr, frame_ms=40, hop_ms=10, fmin=60, fmax=400, r_thresh=0.45):
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    frames = frame_signal(x, frame_len, hop_len)
    win = get_window("hann", frame_len, fftbins=True)

    f0s = np.zeros(len(frames))
    rmaxs = np.zeros(len(frames))

    tau_min = int(sr / fmax)
    tau_max = int(sr / fmin)

    for i, fr in enumerate(frames):
        fr = (fr - np.mean(fr)) * win
        ac = np.correlate(fr, fr, mode="full")
        ac = ac[len(ac) // 2:]

        if ac[0] <= 1e-12:
            continue

        nac = ac / (ac[0] + 1e-12)
        tmax = min(tau_max, len(nac) - 1)

        if tau_min >= tmax:
            continue

        seg = nac[tau_min:tmax + 1]
        idx = np.argmax(seg) + tau_min
        r = nac[idx]

        if r >= r_thresh:
            f0s[i] = sr / idx
            rmaxs[i] = r

    voiced = f0s > 0
    return f0s, voiced, rmaxs, hop_len

def jitter_shimmer_fast(x, sr, f0s, voiced, hop_len):
    Tv = 1.0 / (f0s[voiced] + 1e-12)

    if len(Tv) >= 3:
        jitter_local = float(np.mean(np.abs(np.diff(Tv))) / (np.mean(Tv) + 1e-12))
    else:
        jitter_local = np.nan

    amps = []
    frame_win = int(0.03 * sr)

    for i in np.where(voiced)[0]:
        a = i * hop_len
        b = min(len(x), a + frame_win)
        fr = x[a:b]
        if len(fr) > 5:
            amps.append(np.max(fr) - np.min(fr))

    amps = np.array(amps)

    if len(amps) >= 3:
        shimmer_local = float(np.mean(np.abs(np.diff(amps))) / (np.mean(amps) + 1e-12))
    else:
        shimmer_local = np.nan

    return jitter_local, shimmer_local

def hnr_voiced_only(rmaxs, voiced):
    rv = rmaxs[voiced]
    if len(rv) == 0:
        return np.nan

    r = np.clip(rv, 1e-6, 1 - 1e-6)
    hnr = 10.0 * np.log10(r / (1.0 - r))
    return float(np.mean(hnr))

def analyze_voice_fast(path, fmin=60, fmax=400, r_thresh=0.45):
    x, sr = load_mono(path, 16000)
    xb = bandpass(x, sr, 70, 3500)

    f0s, voiced, rmaxs, hop_len = pitch_ac_track(
        xb, sr, fmin=fmin, fmax=fmax, r_thresh=r_thresh
    )

    voiced_f0 = f0s[voiced]
    pitch_stats = {
        "mean": float(np.mean(voiced_f0)) if len(voiced_f0) else np.nan,
        "median": float(np.median(voiced_f0)) if len(voiced_f0) else np.nan,
        "std": float(np.std(voiced_f0)) if len(voiced_f0) else np.nan,
        "min": float(np.min(voiced_f0)) if len(voiced_f0) else np.nan,
        "max": float(np.max(voiced_f0)) if len(voiced_f0) else np.nan,
    }

    jitter, shimmer = jitter_shimmer_fast(xb, sr, f0s, voiced, hop_len)
    hnr = hnr_voiced_only(rmaxs, voiced)

    return {
        "sr": sr,
        "duration_sec": float(len(x) / sr),
        "pitch_hz": pitch_stats,
        "voicing": {
            "fraction_unvoiced_frames": float(np.mean(~voiced))
        },
        "jitter_local": jitter,
        "shimmer_local": shimmer,
        "hnr_db_voiced_only": hnr,
        "rms": float(np.sqrt(np.mean(x * x)))
    }