# Multi-Language Speech Support

## Overview
The Speech Impaired Assistance panel now supports **8 languages** with language-specific common phrases and proper text-to-speech synthesis.

## Supported Languages

1. **English** (en)
2. **हिंदी Hindi** (hi)
3. **Español Spanish** (es)
4. **Français French** (fr)
5. **Deutsch German** (de)
6. **తెలుగు Telugu** (te)
7. **தமிழ் Tamil** (ta)
8. **বাংলা Bengali** (bn)

## Features

### Language Selector
- Users can select their preferred language from a dropdown menu
- The interface displays language names in both native script and English

### Voice Personalization
- 5 different voice options (Kore, Puck, Charon, Fenrir, Zephyr)
- Voice selection works across all languages

### Language-Specific Common Phrases
Each language has 6 pre-configured common phrases:
- "I need help, please."
- "Where is the restroom?"
- "Thank you very much."
- "Please call my family."
- "I'm not feeling well."
- "Can you write that down for me?"

All phrases are properly translated and displayed in the native script.

## Technical Implementation

### Frontend Changes
**File**: `frontend/components/SpeechPanel.tsx`

- Added `selectedLanguage` state to track current language
- Created `commonPhrases` object with translations for all 8 languages
- Added `languages` array with language codes and display names
- Modified `handleSpeak` to pass language code to backend: `${selectedLanguage}-${selectedVoice}`
- Updated UI to include language selector alongside voice selector
- Common phrases now dynamically update based on selected language

### Backend Changes
**File**: `backend/main.py`

- Expanded `tts_gtts_pcm_base64` function to support all 8 languages
- Added language mappings for gTTS library
- Backend extracts language code from voice parameter (format: `lang-voice`)

## Usage

1. **Select Language**: Choose your preferred language from the "Select Language" dropdown
2. **Select Voice**: Choose a voice that suits your preference
3. **Type or Select**: Either type your message or click a common phrase
4. **Speak**: Click the "Speak" button to hear your message in the selected language

## How It Works

1. User selects language (e.g., "Hindi")
2. Common phrases update to show Hindi text
3. User types or selects a phrase
4. Frontend sends request with language code (e.g., "hi-Kore")
5. Backend extracts language code and uses gTTS to generate speech
6. Audio is converted to PCM format and sent back as base64
7. Frontend plays the audio

## Adding More Languages

To add a new language:

1. **Frontend** (`SpeechPanel.tsx`):
   - Add language to `languages` array with code and name
   - Add translated phrases to `commonPhrases` object

2. **Backend** (`main.py`):
   - Add language code mapping in `tts_gtts_pcm_base64` function
   - Ensure gTTS supports the language (check gTTS documentation)

## Notes

- gTTS (Google Text-to-Speech) supports 100+ languages
- Voice personalization names are cosmetic; actual voice depends on gTTS
- All text is properly encoded in UTF-8 to support non-Latin scripts
- The system gracefully falls back to English if an unsupported language is requested
