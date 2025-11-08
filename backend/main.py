# main.py — OneVoice Backend (Windows-safe + optional Gemini)
import os
import uuid
import time
import json
import shutil
import asyncio
import tempfile
import subprocess
from typing import Optional, Dict
import base64
import contextlib

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx

# Try to load Gemini SDK if present
with contextlib.suppress(ImportError):
    import google.generativeai as genai  # type: ignore

# Try to load Google Cloud Speech-to-Text
try:
    from google.cloud import speech
    from google.cloud.speech import SpeechAdaptation, PhraseSet, PhraseSetReference
    GOOGLE_CLOUD_SPEECH_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_SPEECH_AVAILABLE = False
    print("Warning: Google Cloud Speech-to-Text not available. Install google-cloud-speech package.")

# Try to load OpenCV for sign language recognition
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("Warning: OpenCV not available. Sign language recognition will be disabled.")

# Try to load pickle for ML model
try:
    import pickle
    PICKLE_AVAILABLE = True
except ImportError:
    PICKLE_AVAILABLE = False

# Try to load noisereduce for audio processing
try:
    import noisereduce as nr
    from scipy import signal
    NOISE_REDUCE_AVAILABLE = True
except ImportError:
    NOISE_REDUCE_AVAILABLE = False
    print("Warning: noisereduce not available. Audio noise reduction will be disabled.")

from dotenv import load_dotenv
load_dotenv()  # <-- Add this line

# -----------------------------
# Environment (no hard-coded keys)
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_TRANSLATE_API_KEY = os.getenv("GOOGLE_TRANSLATE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Only use if actually set

# -----------------------------
# App setup
# -----------------------------
app = FastAPI(title="OneVoice Backend", version="0.1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

ALLOWED_TARGETS = {"en": "English", "hi": "Hindi", "es": "Spanish"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB
MAX_DURATION_SECONDS = 10

# -----------------------------
# Models
# -----------------------------
class TranslateResponse(BaseModel):
    sourceText: str
    translatedText: str
    audioUrl: str
    language: str
    latencyMs: int
    stageTimings: Dict[str, int]

class ErrorEnvelope(BaseModel):
    error: Dict[str, str]

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None

class TTSResponse(BaseModel):
    audioBase64: str
    sampleRate: int = 24000
    channels: int = 1

class TranslateTextRequest(BaseModel):
    text: str
    targetLanguage: str

class TranslateTextResponse(BaseModel):
    translatedText: str

class SignLanguageRequest(BaseModel):
    pass  # Video will be uploaded as multipart/form-data

class SignLanguageResponse(BaseModel):
    recognizedText: str
    confidence: float
    processingTimeMs: int

# -----------------------------
# Health
# -----------------------------
@app.get("/healthz")
async def healthz():
    return {"ok": True, "version": app.version}

# -----------------------------
# Windows-safe subprocess helpers
# -----------------------------
def _which(tool: str) -> Optional[str]:
    for p in os.environ.get("PATH", "").split(os.pathsep):
        base = os.path.join(p, tool)
        if os.path.isfile(base) and os.access(base, os.X_OK):
            return base
        if os.name == "nt":
            for ext in (".exe", ".bat", ".cmd"):
                cand = base + ext
                if os.path.isfile(cand) and os.access(cand, os.X_OK):
                    return cand
    return None

async def _run_subprocess(cmd: list[str]) -> subprocess.CompletedProcess:
    def _runner():
        return subprocess.run(cmd, capture_output=True, text=False)
    return await asyncio.to_thread(_runner)

async def _run_ffprobe_duration(path: str) -> float:
    if _which("ffprobe") is None:
        return -1.0
    try:
        proc = await _run_subprocess([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path
        ])
        if proc.returncode != 0:
            return -1.0
        out = (proc.stdout or b"").strip()
        return float(out.decode("utf-8")) if out else -1.0
    except Exception:
        return -1.0

async def _convert_to_wav_16k_mono(src_path: str, dst_path: str) -> None:
    if _which("ffmpeg") is None:
        raise HTTPException(status_code=500, detail="ffmpeg not found on PATH")
    proc = await _run_subprocess(["ffmpeg", "-y", "-i", src_path, "-ac", "1", "-ar", "16000", dst_path])
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {proc.stderr!r}")

async def _convert_audio_to_pcm_base64(src_path: str, sample_rate: int = 24000, channels: int = 1) -> str:
    if _which("ffmpeg") is None:
        raise HTTPException(status_code=500, detail="ffmpeg not found on PATH")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pcm") as tmp_pcm:
        pcm_path = tmp_pcm.name
    try:
        proc = await _run_subprocess([
            "ffmpeg", "-y", "-i", src_path,
            "-f", "s16le", "-acodec", "pcm_s16le",
            "-ac", str(channels), "-ar", str(sample_rate), pcm_path
        ])
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg pcm conversion failed: {proc.stderr!r}")
        with open(pcm_path, "rb") as f:
            raw = f.read()
        return base64.b64encode(raw).decode("ascii")
    finally:
        with contextlib.suppress(Exception):
            os.unlink(pcm_path)

# -----------------------------
# External services
# -----------------------------
async def transcribe_with_whisper(wav_path: str, timeout: float = 15.0) -> str:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    form = {"model": (None, "whisper-1")}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            with open(wav_path, "rb") as f:
                files = {"file": (os.path.basename(wav_path), f, "audio/wav")}
                resp = await client.post(url, headers=headers, data=form, files=files)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Whisper error: {resp.text}")
        data = resp.json()
        return data.get("text", "")
    except httpx.RequestError as e:
        raise HTTPException(status_code=504, detail=f"Whisper timeout/network: {str(e)}")

async def translate_with_google(text: str, target_lang_code: str, timeout: float = 10.0) -> str:
    if not GOOGLE_TRANSLATE_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_TRANSLATE_API_KEY not set")
    url = f"https://translation.googleapis.com/language/translate/v2?key={GOOGLE_TRANSLATE_API_KEY}"
    payload = {"q": text, "target": target_lang_code}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Translate error: {resp.text}")
        data = resp.json()
        return data.get("data", {}).get("translations", [{}])[0].get("translatedText", "")
    except httpx.RequestError as e:
        raise HTTPException(status_code=504, detail=f"Translate timeout/network: {str(e)}")

def _gemini_client():
    if not GOOGLE_API_KEY:
        return None
    if 'genai' not in globals():
        return None
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        return genai.GenerativeModel("gemini-2.0-flash")
    except Exception:
        return None

async def translate_with_gemini(text: str, target_language_name: str) -> str:
    model = _gemini_client()
    if model is None:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not set or Gemini client unavailable")
    prompt = (
        f"Translate the following text to {target_language_name}. "
        f"Return only the translated text without extra commentary.\n\n{text}"
    )
    try:
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, lambda: model.generate_content(prompt))
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()
        return ""
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini translate error: {str(e)}")

# -----------------------------
# Sign Language Recognition helpers
# -----------------------------
class SignLanguageRecognizer:
    """Sign language recognizer using OpenCV and optional ML model."""
    
    def __init__(self):
        # Simple gesture mappings for basic detection
        self.gesture_map = self._initialize_gesture_map()
        
        # Try to load trained ML model
        self.ml_model = None
        self.label_map = None
        self._load_ml_model()
    
    def _load_ml_model(self):
        """Load trained ML model if available."""
        model_path = os.path.join(BASE_DIR, "sign_language_model.pkl")
        if os.path.exists(model_path) and PICKLE_AVAILABLE:
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                self.ml_model = model_data['model']
                self.label_map = model_data['label_map']
                print(f"Loaded ML model with {model_data.get('accuracy', 0)*100:.1f}% accuracy")
            except Exception as e:
                print(f"Could not load ML model: {e}")
                self.ml_model = None
    
    def _initialize_gesture_map(self) -> Dict[str, str]:
        """Initialize basic gesture to letter/word mappings."""
        return {
            "hand_detected": "hello",
            "motion_detected": "gesture detected",
            "static_hand": "sign language",
        }
    
    def _detect_skin(self, frame):
        """Detect skin color regions in the frame."""
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask for skin color
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _analyze_contours(self, mask):
        """Analyze contours to detect hand gestures."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the largest contour (assumed to be the hand)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Filter out small contours (noise)
        if area < 5000:
            return None
        
        # Calculate convex hull and defects for finger counting
        hull = cv2.convexHull(largest_contour, returnPoints=False)
        
        if len(hull) > 3 and len(largest_contour) > 3:
            defects = cv2.convexityDefects(largest_contour, hull)
            
            if defects is not None:
                # Count the number of defects (spaces between fingers)
                finger_count = 0
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    if d > 10000:  # Threshold for significant defects
                        finger_count += 1
                
                # Simple gesture classification based on finger count
                if finger_count == 0:
                    return "closed_fist"  # Fist or thumbs up
                elif finger_count == 1:
                    return "one_finger"  # Pointing
                elif finger_count == 2:
                    return "peace_sign"  # Peace or victory
                elif finger_count >= 4:
                    return "open_palm"  # Open hand
        
        return "hand_detected"
    
    def _extract_hand_region(self, frame):
        """Extract and preprocess hand region for ML model."""
        # Detect skin regions
        mask = self._detect_skin(frame)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest contour (hand)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Extract hand region
        hand_region = frame[y:y+h, x:x+w]
        
        # Convert to grayscale
        gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
        
        # Resize to 28x28 (model input size)
        resized = cv2.resize(gray, (28, 28))
        
        # Normalize to [0, 1]
        normalized = resized.astype('float32') / 255.0
        
        # Flatten to 784 features
        flattened = normalized.reshape(1, -1)
        
        return flattened
    
    def _predict_with_ml(self, hand_features):
        """Predict sign using ML model."""
        if self.ml_model is None or hand_features is None:
            return None
        
        try:
            prediction = self.ml_model.predict(hand_features)[0]
            # Get probability/confidence
            if hasattr(self.ml_model, 'predict_proba'):
                probas = self.ml_model.predict_proba(hand_features)[0]
                confidence = float(np.max(probas))
            else:
                confidence = 0.8  # Default confidence
            
            # Map to letter
            letter = self.label_map.get(prediction, '?')
            return letter, confidence
        except Exception as e:
            print(f"ML prediction error: {e}")
            return None
    
    async def process_video(self, video_path: str) -> tuple[str, float]:
        """Process video and recognize sign language gestures."""
        def _process():
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError("Could not open video file")
            
            recognized_gestures = []
            ml_predictions = []
            frame_count = 0
            total_frames = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                total_frames += 1
                frame_count += 1
                
                # Process every 5th frame to reduce computation
                if frame_count % 5 != 0:
                    continue
                
                # Try ML model first if available
                if self.ml_model is not None:
                    hand_features = self._extract_hand_region(frame)
                    ml_result = self._predict_with_ml(hand_features)
                    if ml_result:
                        letter, conf = ml_result
                        ml_predictions.append((letter, conf))
                else:
                    # Fallback to basic gesture detection
                    mask = self._detect_skin(frame)
                    gesture = self._analyze_contours(mask)
                    if gesture:
                        recognized_gestures.append(gesture)
            
            cap.release()
            
            # Use ML predictions if available
            if ml_predictions:
                # Get most common prediction
                from collections import Counter
                letters = [p[0] for p in ml_predictions]
                letter_counts = Counter(letters)
                most_common_letter = letter_counts.most_common(1)[0][0]
                
                # Average confidence for that letter
                confidences = [p[1] for p in ml_predictions if p[0] == most_common_letter]
                avg_confidence = sum(confidences) / len(confidences)
                
                return f"Letter: {most_common_letter}", avg_confidence
            
            # Fallback to basic gesture recognition
            if not recognized_gestures:
                return "No hand detected in video", 0.0
            
            from collections import Counter
            gesture_counts = Counter(recognized_gestures)
            most_common_gesture = gesture_counts.most_common(1)[0][0]
            confidence = gesture_counts[most_common_gesture] / len(recognized_gestures)
            
            gesture_text_map = {
                "closed_fist": "stop",
                "one_finger": "one",
                "peace_sign": "peace",
                "open_palm": "hello",
                "hand_detected": "sign language gesture",
            }
            text = gesture_text_map.get(most_common_gesture, "gesture detected")
            
            return text, confidence
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _process)
    
    def cleanup(self):
        """Clean up resources."""
        pass  # No cleanup needed for OpenCV-only approach

# Global sign language recognizer instance
_sign_language_recognizer: Optional[SignLanguageRecognizer] = None

def get_sign_language_recognizer() -> Optional[SignLanguageRecognizer]:
    """Get or create sign language recognizer instance."""
    global _sign_language_recognizer
    
    # Check if OpenCV is available
    if not OPENCV_AVAILABLE:
        return None
    
    if _sign_language_recognizer is None:
        try:
            _sign_language_recognizer = SignLanguageRecognizer()
        except Exception as e:
            print(f"Failed to initialize sign language recognizer: {e}")
            return None
    
    return _sign_language_recognizer

# -----------------------------
# TTS helpers
# -----------------------------
def _lang_name_to_code(name: str) -> str:
    m = {
        "english": "en", "hindi": "hi", "spanish": "es",
        "odia": "or", "telugu": "te", "gujarati": "gu",
        "french": "fr", "français": "fr", "francais": "fr",
        "german": "de", "deutsch": "de",
        "tamil": "ta", "தமிழ்": "ta",
        "bengali": "bn", "বাংলা": "bn",
    }
    return m.get(name.strip().lower(), name.strip().lower())

async def tts_with_gtts(text: str, lang: str) -> str:
    from gtts import gTTS
    gtts_lang = {"en": "en", "hi": "hi", "es": "es"}.get(lang, "en")
    out_name = f"{uuid.uuid4().hex}.mp3"
    out_path = os.path.join(STATIC_DIR, out_name)
    def _save():
        gTTS(text=text, lang=gtts_lang).save(out_path)
    await asyncio.to_thread(_save)
    return f"/static/{out_name}"

async def tts_gtts_pcm_base64(text: str, lang_code: str) -> str:
    from gtts import gTTS
    # Expanded language support for gTTS
    gtts_lang = {
        "en": "en",  # English
        "hi": "hi",  # Hindi
        "es": "es",  # Spanish
        "fr": "fr",  # French
        "de": "de",  # German
        "te": "te",  # Telugu
        "ta": "ta",  # Tamil
        "bn": "bn",  # Bengali
        "or": "or",  # Odia
        "gu": "gu",  # Gujarati
    }.get(lang_code, "en")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_mp3:
        mp3_path = tmp_mp3.name
    try:
        def _save():
            gTTS(text=text, lang=gtts_lang).save(mp3_path)
        await asyncio.to_thread(_save)
        return await _convert_audio_to_pcm_base64(mp3_path, sample_rate=24000, channels=1)
    finally:
        with contextlib.suppress(Exception):
            os.unlink(mp3_path)

# -----------------------------
# Routes
# -----------------------------
@app.post("/translate_audio", response_model=TranslateResponse, responses={400: {"model": ErrorEnvelope}})
async def translate_audio(file: UploadFile = File(...), target_lang: str = Form(...)):
    start_ts = time.perf_counter()

    if target_lang not in ALLOWED_TARGETS:
        raise HTTPException(status_code=400, detail=json.dumps({
            "code": "invalid_target", "message": "Unsupported target language",
            "hint": ",".join(ALLOWED_TARGETS.keys())
        }))

    # Size checks
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail=json.dumps({"code": "file_too_large", "message": "Max 10MB"}))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_in:
        in_path = tmp_in.name
        tmp_in.write(content)

    try:
        dur = await _run_ffprobe_duration(in_path)
        if dur < 0 or dur > MAX_DURATION_SECONDS:
            raise HTTPException(status_code=400, detail=json.dumps({"code": "too_long", "message": "Clip too long (max 10s)"}))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            wav_path = tmp_out.name
        await _convert_to_wav_16k_mono(in_path, wav_path)

        t0 = time.perf_counter()
        source_text = await transcribe_with_whisper(wav_path)
        t1 = time.perf_counter()

        target_code = target_lang
        translated_text = await translate_with_google(source_text, target_code)
        t2 = time.perf_counter()

        audio_url = await tts_with_gtts(translated_text, target_code)
        t3 = time.perf_counter()

        return TranslateResponse(
            sourceText=source_text,
            translatedText=translated_text,
            audioUrl=audio_url,
            language=target_lang,
            latencyMs=int((t3 - start_ts) * 1000),
            stageTimings={
                "asrMs": int((t1 - t0) * 1000),
                "translateMs": int((t2 - t1) * 1000),
                "ttsMs": int((t3 - t2) * 1000),
            },
        )
    finally:
        with contextlib.suppress(Exception):
            os.unlink(in_path)
        with contextlib.suppress(Exception):
            os.unlink(wav_path)

@app.post("/tts", response_model=TTSResponse, responses={400: {"model": ErrorEnvelope}})
async def tts_api(payload: TTSRequest):
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail=json.dumps({"code": "invalid_text", "message": "Text required"}))
    
    # Extract language code from voice parameter (format: "lang-voice" or just language name)
    lang_code = "en"
    if payload.voice:
        vc = payload.voice.strip()
        lang_code = vc.split("-")[0].lower() if "-" in vc else _lang_name_to_code(vc)
        if len(lang_code) > 2:
            lang_code = "en"
    
    text_to_speak = payload.text.strip()
    
    # Translate text to the selected language if not already in that language
    # Only translate if we have translation services available and language is not English
    if lang_code != "en" and (GOOGLE_TRANSLATE_API_KEY or GOOGLE_API_KEY):
        try:
            # Get language name for translation
            lang_name_map = {
                "en": "English", "hi": "Hindi", "es": "Spanish", "fr": "French",
                "de": "German", "te": "Telugu", "ta": "Tamil", "bn": "Bengali",
                "or": "Odia", "gu": "Gujarati"
            }
            target_language_name = lang_name_map.get(lang_code, "English")
            
            # Try Gemini first if available, then fall back to Google Translate
            use_gemini = bool(GOOGLE_API_KEY) and ('genai' in globals())
            if use_gemini:
                try:
                    text_to_speak = await translate_with_gemini(text_to_speak, target_language_name)
                except Exception:
                    # Fall back to Google Translate if Gemini fails
                    if GOOGLE_TRANSLATE_API_KEY:
                        text_to_speak = await translate_with_google(text_to_speak, lang_code)
            elif GOOGLE_TRANSLATE_API_KEY:
                text_to_speak = await translate_with_google(text_to_speak, lang_code)
        except Exception as e:
            # If translation fails, log but continue with original text
            print(f"Translation failed, using original text: {e}")
    
    # Generate TTS in the selected language
    audio_b64 = await tts_gtts_pcm_base64(text_to_speak, lang_code)
    return TTSResponse(audioBase64=audio_b64)

@app.post("/translate_text", response_model=TranslateTextResponse, responses={400: {"model": ErrorEnvelope}})
async def translate_text_api(payload: TranslateTextRequest):
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail=json.dumps({"code": "invalid_text", "message": "Text required"}))

    text_in = payload.text.strip()
    target_name = payload.targetLanguage

    # Prefer Gemini only if properly configured; otherwise use Google Translate
    use_gemini = bool(GOOGLE_API_KEY) and ('genai' in globals())
    if use_gemini:
        try:
            translated = await translate_with_gemini(text_in, target_name)
            if translated:
                return TranslateTextResponse(translatedText=translated)
        except HTTPException:
            raise
        except Exception:
            pass  # fall through to Google Translate

    if not GOOGLE_TRANSLATE_API_KEY:
        # Neither Gemini nor Google Translate available
        raise HTTPException(status_code=500, detail="No translation service configured. Set GOOGLE_API_KEY or GOOGLE_TRANSLATE_API_KEY")

    target_code = _lang_name_to_code(target_name)
    translated = await translate_with_google(text_in, target_code)
    return TranslateTextResponse(translatedText=translated)

@app.post("/sign_language_to_text", response_model=SignLanguageResponse, responses={400: {"model": ErrorEnvelope}})
async def sign_language_to_text(file: UploadFile = File(...)):
    """Convert sign language video to text."""
    start_ts = time.perf_counter()
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=400,
            detail=json.dumps({"code": "invalid_file_type", "message": "Video file required"})
        )
    
    # Check if sign language recognizer is available
    recognizer = get_sign_language_recognizer()
    if recognizer is None:
        raise HTTPException(
            status_code=500,
            detail="Sign language recognition not available. Install opencv-python and numpy."
        )
    
    # Read upload to temp file
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=json.dumps({"code": "file_too_large", "message": "Max 10MB"})
        )
    
    # Determine file extension from content type
    ext_map = {
        "video/mp4": ".mp4",
        "video/webm": ".webm",
        "video/quicktime": ".mov",
        "video/x-msvideo": ".avi",
    }
    ext = ext_map.get(file.content_type, ".mp4")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_video:
        video_path = tmp_video.name
        tmp_video.write(content)
    
    try:
        # Process video
        recognized_text, confidence = await recognizer.process_video(video_path)
        
        end_ts = time.perf_counter()
        processing_time_ms = int((end_ts - start_ts) * 1000)
        
        return SignLanguageResponse(
            recognizedText=recognized_text,
            confidence=confidence,
            processingTimeMs=processing_time_ms
        )
    finally:
        with contextlib.suppress(Exception):
            os.unlink(video_path)

# -----------------------------
# WebSocket Voice Streaming
# -----------------------------

# Comprehensive phrase sets for voice commands with multiple variations
# Higher boost values for more important commands

# Navigation commands - Highest priority
NAVIGATION_PHRASES = [
    # Hearing mode variations
    "go to hearing", "go to hearing mode", "go to hearing panel", "open hearing", "open hearing mode",
    "switch to hearing", "switch to hearing mode", "switch hearing", "change to hearing",
    "show hearing", "show hearing mode", "navigate to hearing", "navigate hearing",
    "hearing mode", "hearing panel", "hearing section",
    
    # Speech mode variations
    "go to speech", "go to speech mode", "go to speech panel", "open speech", "open speech mode",
    "switch to speech", "switch to speech mode", "switch speech", "change to speech",
    "show speech", "show speech mode", "navigate to speech", "navigate speech",
    "speech mode", "speech panel", "speech section", "speech impaired",
    
    # Sign language mode variations
    "go to sign language", "go to sign language mode", "go to sign language panel",
    "open sign language", "open sign language mode", "open sign", "open signs",
    "switch to sign language", "switch to sign language mode", "switch sign language",
    "change to sign language", "show sign language", "show sign language mode",
    "navigate to sign language", "navigate sign language",
    "sign language mode", "sign language panel", "sign language section", "sign mode",
]

# Voice control commands
VOICE_CONTROL_PHRASES = [
    "enable voice", "enable voice commands", "enable voice control", "enable voice mode",
    "turn on voice", "turn on voice commands", "turn on voice control",
    "start voice", "start voice commands", "start voice control", "start listening for commands",
    "activate voice", "activate voice commands", "activate voice control",
    
    "disable voice", "disable voice commands", "disable voice control", "disable voice mode",
    "turn off voice", "turn off voice commands", "turn off voice control",
    "stop voice", "stop voice commands", "stop voice control", "stop listening for commands",
    "deactivate voice", "deactivate voice commands", "deactivate voice control",
]

# Hearing panel commands
HEARING_PHRASES = [
    # Listening commands
    "start listening", "stop listening", "begin listening", "end listening",
    "start hearing", "stop hearing", "start capture", "stop capture",
    "listen now", "stop listen", "begin capture", "end capture",
    "start transcription", "stop transcription", "start transcribing", "stop transcribing",
    
    # Translation commands
    "enable translation", "disable translation", "turn on translation", "turn off translation",
    "enable translate", "disable translate", "turn on translate", "turn off translate",
    "start translation", "stop translation", "activate translation", "deactivate translation",
    
    # Language selection
    "set language to english", "set language to hindi", "set language to odia",
    "set language to telugu", "set language to gujarati", "set language to spanish",
    "set language to french", "set language to german", "set language to tamil",
    "set language to bengali", "change language to english", "change language to hindi",
    "language english", "language hindi", "language spanish", "language french",
    "translate to english", "translate to hindi", "translate to spanish",
    
    # Clear commands
    "clear transcript", "reset transcript", "erase transcript", "clear text",
    "clear captions", "reset captions", "erase captions", "clear all",
    "delete transcript", "delete text", "clear hearing", "reset hearing",
]

# Speech panel commands
SPEECH_PHRASES = [
    # Speak commands
    "speak", "say", "read", "pronounce", "speak now", "say now", "read now",
    "speak this", "say this", "read this", "pronounce this",
    "speak text", "say text", "read text", "pronounce text",
    "speak the text", "say the text", "read the text",
    "read aloud", "speak aloud", "say aloud", "read out loud",
    
    # Dictation commands
    "start dictation", "stop dictation", "begin dictation", "end dictation",
    "start typing", "stop typing", "start voice typing", "stop voice typing",
    "start speaking", "stop speaking", "listen for text", "stop listening for text",
    
    # Language selection
    "set language to spanish", "set language to french", "set language to german",
    "set language to tamil", "set language to bengali", "change language to spanish",
    "language spanish", "language french", "language german",
    "speak in spanish", "speak in french", "speak in german",
    
    # Voice selection
    "set voice to kore", "set voice to puck", "set voice to charon",
    "set voice to fenrir", "set voice to zephyr", "change voice to kore",
    "use kore", "use puck", "use charon", "use fenrir", "use zephyr",
    "voice kore", "voice puck", "voice charon", "voice fenrir", "voice zephyr",
    
    # Phrase selection
    "use phrase one", "use phrase two", "use phrase three",
    "use phrase four", "use phrase five", "use phrase six",
    "phrase one", "phrase two", "phrase three", "phrase four", "phrase five", "phrase six",
    "select phrase one", "select phrase two", "select phrase three",
    
    # Clear commands
    "clear text", "reset text", "erase text", "clear input", "delete text",
    "clear speech", "reset speech", "erase speech", "clear all text",
]

# Sign language panel commands
SIGN_LANGUAGE_PHRASES = [
    # Recording commands
    "start recording", "stop recording", "begin recording", "end recording",
    "start video", "stop video", "begin video", "end video",
    "start capture", "stop capture", "begin capture", "end capture",
    "record now", "stop record", "start record", "end record",
    "start camera", "stop camera", "turn on camera", "turn off camera",
    
    # Recognition commands
    "recognize signs", "process video", "analyze signs", "analyze video",
    "process signs", "recognize video", "detect signs", "identify signs",
    "process recording", "analyze recording", "recognize recording",
    "what are the signs", "what signs", "show signs", "display signs",
    
    # Clear commands
    "clear result", "reset result", "erase result", "clear video",
    "clear recording", "reset recording", "erase recording", "clear all",
    "delete result", "delete video", "delete recording",
]

# Combine all phrases
VOICE_COMMAND_PHRASES = (
    NAVIGATION_PHRASES +
    VOICE_CONTROL_PHRASES +
    HEARING_PHRASES +
    SPEECH_PHRASES +
    SIGN_LANGUAGE_PHRASES
)

GOOGLE_CLOUD_SPEECH_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
GOOGLE_CLOUD_SPEECH_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")

@app.websocket("/ws/voice")
async def websocket_voice(websocket: WebSocket):
    """
    WebSocket endpoint for streaming voice recognition.
    Receives audio chunks and returns transcripts using Google Cloud Speech-to-Text.
    """
    await websocket.accept()
    
    if not GOOGLE_CLOUD_SPEECH_AVAILABLE:
        await websocket.send_json({
            "type": "error",
            "message": "Google Cloud Speech-to-Text not available. Please install google-cloud-speech."
        })
        await websocket.close()
        return
    
    # Check for Google Cloud credentials
    if not GOOGLE_CLOUD_SPEECH_CREDENTIALS and not GOOGLE_CLOUD_SPEECH_PROJECT:
        await websocket.send_json({
            "type": "error",
            "message": "Google Cloud credentials not configured. Set GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_CLOUD_PROJECT."
        })
        await websocket.close()
        return
    
    try:
        # Initialize Google Cloud Speech client
        client = speech.SpeechClient()
        
        # Create multiple phrase sets with different boost values for better accuracy
        # Navigation commands get highest boost
        navigation_phrase_set = speech.PhraseSet(
            phrases=[
                speech.PhraseSet.Phrase(value=phrase, boost=20.0)
                for phrase in NAVIGATION_PHRASES
            ]
        )
        
        # Voice control commands get high boost
        voice_control_phrase_set = speech.PhraseSet(
            phrases=[
                speech.PhraseSet.Phrase(value=phrase, boost=18.0)
                for phrase in VOICE_CONTROL_PHRASES
            ]
        )
        
        # Panel-specific commands get medium-high boost
        hearing_phrase_set = speech.PhraseSet(
            phrases=[
                speech.PhraseSet.Phrase(value=phrase, boost=15.0)
                for phrase in HEARING_PHRASES
            ]
        )
        
        speech_phrase_set = speech.PhraseSet(
            phrases=[
                speech.PhraseSet.Phrase(value=phrase, boost=15.0)
                for phrase in SPEECH_PHRASES
            ]
        )
        
        sign_language_phrase_set = speech.PhraseSet(
            phrases=[
                speech.PhraseSet.Phrase(value=phrase, boost=15.0)
                for phrase in SIGN_LANGUAGE_PHRASES
            ]
        )
        
        # Create speech adaptation with all phrase sets
        adaptation = speech.SpeechAdaptation(
            phrase_sets=[
                navigation_phrase_set,
                voice_control_phrase_set,
                hearing_phrase_set,
                speech_phrase_set,
                sign_language_phrase_set,
            ]
        )
        
        # Configure recognition with automatic language detection
        # Support multiple languages for detection
        alternative_language_codes = [
            "en-US", "hi-IN", "es-ES", "fr-FR", "de-DE", 
            "te-IN", "ta-IN", "bn-IN", "or-IN", "gu-IN"
        ]
        
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",  # Primary language
            alternative_language_codes=alternative_language_codes,  # Auto-detect from these
            enable_automatic_punctuation=True,
            enable_interim_results=True,
            enable_word_time_offsets=True,  # Better accuracy tracking
            enable_word_confidence=True,  # Word-level confidence scores
            adaptation=adaptation,
            model="latest_long",  # Best for continuous speech
            use_enhanced=True,  # Use enhanced model for better accuracy
            audio_channel_count=1,  # Mono audio
            enable_separate_recognition_per_channel=False,
        )
        
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True,
            single_utterance=False,  # Allow continuous recognition
        )
        
        # Use a queue to collect audio chunks
        import queue
        import threading
        audio_queue = queue.Queue()
        response_queue = queue.Queue()
        stream_closed = False
        
        async def receive_audio():
            """Receive audio chunks from WebSocket and add to queue"""
            try:
                while True:
                    data = await websocket.receive_bytes()
                    if not data:
                        break
                    audio_queue.put(data)
            except WebSocketDisconnect:
                pass
            finally:
                nonlocal stream_closed
                stream_closed = True
                audio_queue.put(None)  # Signal end of stream
        
        def process_audio_chunk(audio_bytes: bytes) -> bytes:
            """Apply noise reduction and normalization to audio chunk"""
            if not NOISE_REDUCE_AVAILABLE:
                return audio_bytes
            
            try:
                # Convert bytes to numpy array (Int16)
                audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                
                # Convert to float32 for processing (-1.0 to 1.0 range)
                audio_float = audio_int16.astype(np.float32) / 32768.0
                
                # Apply noise reduction with aggressive settings for voice commands
                audio_reduced = nr.reduce_noise(
                    y=audio_float, 
                    sr=16000, 
                    stationary=False,
                    prop_decrease=0.8,  # More aggressive noise reduction
                    n_fft=2048,  # Better frequency resolution
                    win_length=2048,
                    hop_length=512
                )
                
                # Apply high-pass filter to remove low-frequency noise
                try:
                    from scipy.signal import butter, filtfilt
                    nyquist = 16000 / 2
                    low_cutoff = 80 / nyquist  # Remove frequencies below 80 Hz
                    b, a = butter(4, low_cutoff, btype='high')
                    audio_filtered = filtfilt(b, a, audio_reduced)
                except Exception:
                    audio_filtered = audio_reduced
                
                # Normalize audio (RMS normalization for better consistency)
                rms = np.sqrt(np.mean(audio_filtered**2))
                if rms > 0:
                    target_rms = 0.15  # Slightly higher target for better clarity
                    audio_normalized = audio_filtered * (target_rms / rms)
                    # Clip to prevent distortion but allow more dynamic range
                    audio_normalized = np.clip(audio_normalized, -0.98, 0.98)
                else:
                    audio_normalized = audio_filtered
                
                # Convert back to Int16
                audio_processed = (audio_normalized * 32767.0).astype(np.int16)
                
                # Convert back to bytes
                return audio_processed.tobytes()
            except Exception as e:
                print(f"Audio processing error: {e}")
                return audio_bytes  # Return original if processing fails
        
        def audio_generator():
            """Generator that yields processed audio chunks from queue"""
            while True:
                try:
                    chunk = audio_queue.get(timeout=1)
                    if chunk is None:  # End of stream signal
                        break
                    # Process audio chunk (noise reduction + normalization)
                    processed_chunk = process_audio_chunk(chunk)
                    yield speech.StreamingRecognizeRequest(audio_content=processed_chunk)
                except queue.Empty:
                    if stream_closed:
                        break
                    continue
        
        def run_recognition():
            """Run Google Cloud Speech recognition in a separate thread"""
            try:
                responses = client.streaming_recognize(streaming_config, audio_generator())
                for response in responses:
                    response_queue.put(response)
            except Exception as e:
                response_queue.put({"error": str(e)})
            finally:
                response_queue.put(None)  # Signal end
        
        # Start receiving audio in background
        receive_task = asyncio.create_task(receive_audio())
        
        # Start recognition in a separate thread
        recognition_thread = threading.Thread(target=run_recognition, daemon=True)
        recognition_thread.start()
        
        try:
            # Process responses from the queue
            while True:
                try:
                    response = response_queue.get(timeout=0.5)
                    if response is None:  # End of stream
                        break
                    
                    if isinstance(response, dict) and "error" in response:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Recognition error: {response['error']}"
                        })
                        break
                    
                    if response.error:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Recognition error: {response.error.message}"
                        })
                        continue
                    
                    for result in response.results:
                        if not result.alternatives:
                            continue
                        
                        alternative = result.alternatives[0]
                        transcript = alternative.transcript
                        confidence = alternative.confidence if hasattr(alternative, 'confidence') else None
                        
                        # Extract detected language code from result
                        # Google Cloud Speech returns language_code in the result if language detection is enabled
                        detected_language = "en-US"  # Default
                        if hasattr(result, 'language_code') and result.language_code:
                            detected_language = result.language_code
                        elif hasattr(alternative, 'language_code') and alternative.language_code:
                            detected_language = alternative.language_code
                        
                        # Convert language code to our format (e.g., "en-US" -> "en", "hi-IN" -> "hi")
                        lang_code = detected_language.split("-")[0].lower()
                        
                        # Send transcript to client with detected language
                        await websocket.send_json({
                            "type": "transcript",
                            "transcript": transcript,
                            "is_final": result.is_final_alternative,
                            "confidence": confidence,
                            "language_code": lang_code,  # Add detected language
                            "language_full": detected_language,  # Full language code
                        })
                except queue.Empty:
                    # Check if recognition thread is still alive
                    if not recognition_thread.is_alive():
                        break
                    continue
        
        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "message": f"Streaming error: {str(e)}"
            })
        finally:
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Connection error: {str(e)}"
            })
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass

# -----------------------------
# Dev entry
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    # Proactor loop (extra safety on Windows)
    if os.name == "nt":
        with contextlib.suppress(Exception):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())  # type: ignore[attr-defined]
    uvicorn.run("main:app", host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")), reload=True)
