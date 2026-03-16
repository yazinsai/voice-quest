"""
Voice Gender Detection Demo — F0 pitch analysis using Praat (parselmouth).
Records audio in the browser, analyzes fundamental frequency, returns gender probability.
"""

import io
import tempfile
import numpy as np
import parselmouth
from pydub import AudioSegment
from scipy.stats import norm
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from pathlib import Path

app = FastAPI()

# F0 ranges (Hz) — based on phonetics research
MALE_MEAN = 120.0
FEMALE_MEAN = 210.0
# Standard deviations (approximate)
MALE_STD = 25.0
FEMALE_STD = 25.0


def analyze_pitch(audio_bytes: bytes) -> dict:
    """
    Analyze audio bytes and return gender probability based on F0 pitch.

    Uses Praat's pitch detection algorithm via parselmouth.
    Returns probability scores and raw F0 statistics.
    """
    # Convert webm/ogg to WAV (parselmouth only reads WAV/AIFF natively)
    webm_path = None
    wav_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            f.write(audio_bytes)
            webm_path = f.name

        audio_seg = AudioSegment.from_file(webm_path)
        wav_path = webm_path.replace(".webm", ".wav")
        audio_seg.export(wav_path, format="wav")

        sound = parselmouth.Sound(wav_path)
    except Exception as e:
        return {"error": f"Could not process audio: {e}"}
    finally:
        if webm_path:
            Path(webm_path).unlink(missing_ok=True)
        if wav_path:
            Path(wav_path).unlink(missing_ok=True)

    # Extract pitch using Praat's autocorrelation method
    pitch = sound.to_pitch(time_step=0.01, pitch_floor=50, pitch_ceiling=400)
    pitch_values = pitch.selected_array["frequency"]

    # Filter out unvoiced frames (0 Hz)
    voiced = pitch_values[pitch_values > 0]

    if len(voiced) < 5:
        return {"error": "Not enough voiced speech detected. Try speaking louder or longer."}

    mean_f0 = float(np.mean(voiced))
    median_f0 = float(np.median(voiced))
    std_f0 = float(np.std(voiced))
    min_f0 = float(np.min(voiced))
    max_f0 = float(np.max(voiced))

    # Compute probability using simple Gaussian likelihood ratio
    # P(female | f0) ∝ P(f0 | female) * P(female)
    # Assuming equal priors
    p_male = norm.pdf(median_f0, MALE_MEAN, MALE_STD)
    p_female = norm.pdf(median_f0, FEMALE_MEAN, FEMALE_STD)

    total = p_male + p_female
    if total == 0:
        male_prob = 0.5
        female_prob = 0.5
    else:
        male_prob = p_male / total
        female_prob = p_female / total

    # Classification
    if female_prob > 0.7:
        gender = "female"
        confidence = "high"
    elif female_prob > 0.5:
        gender = "female"
        confidence = "low"
    elif male_prob > 0.7:
        gender = "male"
        confidence = "high"
    else:
        gender = "male"
        confidence = "low"

    return {
        "gender": gender,
        "confidence": confidence,
        "male_probability": round(male_prob * 100, 1),
        "female_probability": round(female_prob * 100, 1),
        "f0_mean": round(mean_f0, 1),
        "f0_median": round(median_f0, 1),
        "f0_std": round(std_f0, 1),
        "f0_min": round(min_f0, 1),
        "f0_max": round(max_f0, 1),
        "voiced_frames": len(voiced),
        "duration_seconds": round(sound.duration, 2),
    }


@app.post("/analyze")
async def analyze(audio: UploadFile = File(...)):
    """Accept audio upload and return gender analysis."""
    audio_bytes = await audio.read()
    if len(audio_bytes) < 1000:
        return {"error": "Audio too short. Please record at least 2 seconds."}
    result = analyze_pitch(audio_bytes)
    return result


@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "index.html").read_text()
