# OneVoice - Assistive Communication Platform

OneVoice is a comprehensive assistive communication application designed to help users with hearing, speech, and visual impairments communicate effectively. The platform features real-time speech-to-text, text-to-speech, sign language recognition, multilingual support, and full voice-activated controls.

## 🌟 Features

### Core Features

1. **Hearing Impaired Assistance**
   - Real-time speech-to-text transcription
   - Live captions with large, readable text
   - Sound alert detection (siren, doorbell, baby crying)
   - Visual and haptic feedback for alerts
   - Real-time translation to multiple languages

2. **Speech Impaired Assistance**
   - Text-to-speech in 8 languages
   - 5 customizable voice options
   - Language-specific common phrases
   - Automatic language translation before speaking
   - Support for: English, Hindi, Spanish, French, German, Telugu, Tamil, Bengali

3. **Sign Language Recognition**
   - Real-time video recording
   - Sign language gesture recognition
   - ML-based hand gesture detection
   - Confidence scoring
   - Support for multiple sign language gestures

4. **Full Voice Activation**
   - Complete voice control for all features
   - Natural language command recognition
   - Context-aware commands per panel
   - Visual feedback for voice status
   - Help panel with command reference

5. **Multilingual Support**
   - 8 languages for text-to-speech
   - Real-time translation using Google Translate API
   - Optional Gemini AI integration for enhanced translation
   - Language-specific phrase libraries

## 📁 Project Structure

```
windsurf-project/
├── backend/                    # FastAPI backend server
│   ├── main.py                 # Main backend application
│   ├── requirements.txt        # Python dependencies
│   ├── quick_start.py          # Quick start script
│   ├── train_sign_language.py  # ML model training script
│   ├── test_dataset.py         # Dataset testing utilities
│   ├── sign_language_model.pkl # Trained ML model
│   ├── TRAINING_GUIDE.md       # ML training documentation
│   ├── static/                 # Static files (generated audio)
│   └── dataset/                # Sign language training dataset
│       ├── sign_mnist_train/   # Training data
│       └── sign_mnist_test/    # Test data
│
├── frontend/                   # React + TypeScript frontend
│   ├── App.tsx                 # Main application component
│   ├── index.tsx               # Application entry point
│   ├── index.html              # HTML template
│   ├── package.json            # Node.js dependencies
│   ├── tsconfig.json           # TypeScript configuration
│   ├── vite.config.ts          # Vite build configuration
│   ├── metadata.json           # App metadata
│   │
│   ├── components/             # React components
│   │   ├── Header.tsx          # Navigation header
│   │   ├── HearingPanel.tsx    # Hearing assistance panel
│   │   ├── SpeechPanel.tsx     # Speech assistance panel
│   │   ├── SignLanguagePanel.tsx # Sign language panel
│   │   └── VoiceHelpPanel.tsx  # Voice commands help panel
│   │
│   ├── services/               # Service modules
│   │   ├── gemini.ts           # API service layer
│   │   └── voiceCommands.ts     # Voice command service
│   │
│   ├── types.ts                # TypeScript type definitions
│   └── constants.tsx           # UI constants and icons
│
└── README.md                   # This file
```

## 📄 File Descriptions

### Backend Files

#### `backend/main.py`
Main FastAPI application providing REST API endpoints:
- **Endpoints:**
  - `POST /translate_audio` - Audio translation pipeline (ASR → Translation → TTS)
  - `POST /tts` - Text-to-speech with automatic translation
  - `POST /translate_text` - Text translation (Google Translate or Gemini)
  - `POST /sign_language_to_text` - Sign language video recognition
  - `GET /healthz` - Health check endpoint
- **Features:**
  - Windows-safe subprocess handling
  - Audio conversion using ffmpeg
  - OpenAI Whisper integration for ASR
  - Google Translate API integration
  - Optional Gemini AI for translation
  - Sign language recognition using OpenCV and ML models
  - Static file serving for generated audio

#### `backend/requirements.txt`
Python dependencies:
- `fastapi==0.115.0` - Web framework
- `uvicorn[standard]==0.30.6` - ASGI server
- `httpx==0.27.2` - HTTP client
- `pydantic==2.8.2` - Data validation
- `gTTS==2.5.3` - Google Text-to-Speech
- `google-generativeai==0.7.2` - Gemini AI SDK
- `opencv-python>=4.8.0` - Computer vision
- `numpy>=1.24.0` - Numerical computing
- `scikit-learn>=1.3.0` - Machine learning
- `pandas>=2.0.0` - Data manipulation
- `exceptiongroup>=1.0.0` - Exception handling

#### `backend/train_sign_language.py`
Machine learning model training script for sign language recognition.

#### `backend/test_dataset.py`
Utilities for testing and validating the sign language dataset.

#### `backend/quick_start.py`
Quick start script for initializing the backend.

### Frontend Files

#### `frontend/App.tsx`
Main application component:
- Manages application state and mode switching
- Integrates voice command service
- Provides navigation between panels
- Displays voice command status indicator

#### `frontend/components/Header.tsx`
Navigation header component:
- Mode selection buttons (Hearing, Speech, Sign Language)
- Visual indicators for active mode
- Responsive design

#### `frontend/components/HearingPanel.tsx`
Hearing assistance panel:
- Real-time speech-to-text transcription
- Translation toggle and language selection
- Sound alert detection and display
- Voice commands for all controls

#### `frontend/components/SpeechPanel.tsx`
Speech assistance panel:
- Text input for speech generation
- Language and voice selection
- Common phrase library (8 languages)
- Text-to-speech with automatic translation
- Voice commands for all actions

#### `frontend/components/SignLanguagePanel.tsx`
Sign language recognition panel:
- Video recording interface
- Sign language gesture recognition
- Result display with confidence scores
- Voice commands for recording control

#### `frontend/components/VoiceHelpPanel.tsx`
Voice commands help panel:
- Displays available voice commands
- Context-aware command lists
- Toggleable help interface

#### `frontend/services/gemini.ts`
API service layer:
- `generateTextToSpeech()` - TTS API calls
- `translateText()` - Translation API calls
- `recognizeSignLanguage()` - Sign language API calls
- `playBase64Audio()` - Audio playback utilities

#### `frontend/services/voiceCommands.ts`
Voice command service:
- Centralized voice recognition management
- Command pattern matching
- Handler registration system
- Auto-recovery from errors
- Status callbacks

#### `frontend/types.ts`
TypeScript type definitions:
- `Mode` enum for application modes
- `SoundAlertInfo` interface
- Other shared types

#### `frontend/constants.tsx`
UI constants and SVG icons:
- Icon components (Ear, Eye, Speech, Hand, etc.)
- Reusable UI elements

## 🚀 Setup Instructions

### Prerequisites

- **Python 3.8+** (for backend)
- **Node.js 18+** (for frontend)
- **ffmpeg** (for audio processing)
- **API Keys:**
  - AssemblyAI API key (for real-time speech recognition)
  - OpenAI API key (for Whisper ASR)
  - Google Translate API key (for translation)
  - Google API key (optional, for Gemini)

### Backend Setup

1. Navigate to backend directory:
```bash
cd windsurf-project/backend
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file:
```env
ASSEMBLYAI_API_KEY=your_assemblyai_api_key
OPENAI_API_KEY=your_openai_api_key
GOOGLE_TRANSLATE_API_KEY=your_google_translate_key
GOOGLE_API_KEY=your_google_api_key  # Optional
```

5. (Optional) Verify AssemblyAI setup:
```bash
python setup_assemblyai.py
```

6. Start the server:
```bash
uvicorn main:app --reload
```

The backend will run on `http://127.0.0.1:8000`

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd windsurf-project/frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start development server:
```bash
npm run dev
```

The frontend will run on `http://localhost:5173` (or similar)

## 🎤 Voice Commands Reference

### Global Commands (Available Everywhere)

| Command | Description |
|--------|-------------|
| "Go to hearing mode" | Navigate to Hearing panel |
| "Go to speech mode" | Navigate to Speech panel |
| "Go to sign language mode" | Navigate to Sign Language panel |
| "Enable voice commands" | Start voice recognition |
| "Disable voice commands" | Stop voice recognition |

### Hearing Mode Commands

| Command | Description |
|--------|-------------|
| "Start listening" | Begin speech-to-text transcription |
| "Stop listening" | Stop transcription |
| "Enable translation" | Turn on real-time translation |
| "Disable translation" | Turn off translation |
| "Set language to [Hindi/English/Odia/Telugu/Gujarati]" | Change translation language |
| "Clear transcript" | Clear the transcript text |

### Speech Mode Commands

| Command | Description |
|--------|-------------|
| "Speak [text]" | Speak the specified text |
| "Speak this" | Speak current text in input |
| "Set language to [English/Hindi/Spanish/French/German/Telugu/Tamil/Bengali]" | Change TTS language |
| "Set voice to [Kore/Puck/Charon/Fenrir/Zephyr]" | Change voice |
| "Use phrase 1" through "Use phrase 6" | Use common phrase by number |
| "Clear text" | Clear text input |

### Sign Language Mode Commands

| Command | Description |
|--------|-------------|
| "Start recording" | Begin recording sign language video |
| "Stop recording" | Stop recording |
| "Recognize signs" | Process the recorded video |
| "Clear result" | Clear recognition results |

## 🔌 API Endpoints

### POST `/translate_audio`
Audio translation pipeline endpoint.

**Request:**
- `file`: Audio file (multipart/form-data)
- `target_lang`: Target language code (en, hi, es)

**Response:**
```json
{
  "sourceText": "Hello",
  "translatedText": "नमस्ते",
  "audioUrl": "/static/abc123.mp3",
  "language": "hi",
  "latencyMs": 2500,
  "stageTimings": {
    "asrMs": 1200,
    "translateMs": 800,
    "ttsMs": 500
  }
}
```

### POST `/tts`
Text-to-speech endpoint with automatic translation.

**Request:**
```json
{
  "text": "Hello",
  "voice": "en-Kore"
}
```

**Response:**
```json
{
  "audioBase64": "base64_encoded_audio",
  "sampleRate": 24000,
  "channels": 1
}
```

### POST `/translate_text`
Text translation endpoint.

**Request:**
```json
{
  "text": "Hello",
  "targetLanguage": "Hindi"
}
```

**Response:**
```json
{
  "translatedText": "नमस्ते"
}
```

### POST `/sign_language_to_text`
Sign language recognition endpoint.

**Request:**
- `file`: Video file (multipart/form-data)

**Response:**
```json
{
  "recognizedText": "hello",
  "confidence": 0.85,
  "processingTimeMs": 1200
}
```

### GET `/healthz`
Health check endpoint.

**Response:**
```json
{
  "ok": true,
  "version": "0.1.1"
}
```

## 🛠️ Development

### Running in Development Mode

**Backend:**
```bash
cd backend
uvicorn main:app --reload
```

**Frontend:**
```bash
cd frontend
npm run dev
```

### Building for Production

**Frontend:**
```bash
cd frontend
npm run build
```

The built files will be in `frontend/dist/`

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the `backend/` directory:

```env
# Required
OPENAI_API_KEY=sk-...
GOOGLE_TRANSLATE_API_KEY=...

# Optional
GOOGLE_API_KEY=...  # For Gemini AI translation
HOST=127.0.0.1
PORT=8000
```

### Frontend Configuration

The frontend API base URL can be configured via environment variable:
- `VITE_API_BASE` - Default: `http://127.0.0.1:8000`

## 📚 Additional Documentation

- `ASSEMBLYAI_MIGRATION_GUIDE.md` - **NEW!** AssemblyAI real-time speech recognition guide
- `MULTILINGUAL_SPEECH.md` - Multilingual speech support details
- `SIGN_LANGUAGE_SETUP.md` - Sign language recognition setup
- `backend/TRAINING_GUIDE.md` - ML model training guide

## 🎤 Real-Time Speech Recognition (NEW!)

OneVoice now uses **AssemblyAI** for real-time speech recognition, providing:

- ✅ **75% less code** - Simplified from 300 to 80 lines
- ✅ **Faster response** - 200-400ms latency (vs 500-800ms)
- ✅ **Better accuracy** - State-of-the-art models with word boosting
- ✅ **Easier setup** - Just add an API key, no complex configuration
- ✅ **Auto-recovery** - Built-in error handling and reconnection

### Quick Start with AssemblyAI

1. Get your API key from [AssemblyAI](https://www.assemblyai.com/)
2. Add to `.env`: `ASSEMBLYAI_API_KEY=your_key_here`
3. Run: `python backend/setup_assemblyai.py` to verify
4. Start the backend and enjoy real-time voice recognition!

See `ASSEMBLYAI_MIGRATION_GUIDE.md` for complete details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **AssemblyAI** for real-time speech recognition
- OpenAI Whisper for speech recognition
- Google Translate API for translation
- Google Text-to-Speech (gTTS) for TTS
- OpenCV for computer vision
- FastAPI for the backend framework
- React for the frontend framework

## 🎨 UI/UX Design

For detailed UI/UX design documentation, including layout structures, color schemes, interaction patterns, and accessibility features, see [UI_UX_DESIGN.md](./UI_UX_DESIGN.md).

### Quick UI Overview

- **Mode-Based Themes**: Each mode (Hearing, Speech, Sign Language) has a distinct color theme
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **Voice Status Indicator**: Always-visible indicator showing voice command system status
- **Help Panel**: Context-aware voice command reference (bottom-left)
- **Accessibility**: High contrast, large text, keyboard navigation, screen reader support
- **Real-time Feedback**: Visual and audio feedback for all user actions

## 📧 Support

For issues, questions, or contributions, please open an issue on the repository.

---

**OneVoice** - Empowering communication for everyone.

