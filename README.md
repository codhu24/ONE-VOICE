Here is a comprehensive README file for your project, detailing the problem it solves, its solution, the technology stack, and an explanation of the key files.

-----

# OneVoice - Accessibility Assistant

**OneVoice** (also referred to as "Accessibility Assistant") is an AI-powered assistive communication web application designed to bridge communication gaps for users with hearing and speech impairments. It provides a seamless, real-time interface for transcription, translation, and sign language recognition.

## 1\. The Problem Statement

Effective communication is a daily challenge for individuals with hearing or speech impairments. This project addresses three core problem areas:

1.  **For the Hearing Impaired:** Difficulty in participating in spoken conversations, especially in multiple languages.
2.  **For the Speech Impaired:** Difficulty in communicating with those who do not understand sign language or typed text.
3.  **Cross-Communication Barriers:** The gap between individuals who use spoken language and those who use sign language.

## 2\. The Solution

OneVoice provides an integrated, multi-modal solution in a single web application. The interface is divided into three primary modes:

### Hearing Mode

This panel provides real-time assistance for individuals with hearing impairments.

  * **Live Transcription:** Uses a microphone to capture spoken language and transcribes it into text on the screen in real-time. This is powered by a WebSocket connection to a backend service using AssemblyAI.
  * **Live Translation:** Transcribed text can be instantly translated into multiple languages, including Hindi, Spanish, French, German, and more.
  * **Sound Alerts:** The application can detect and display visual alerts for critical sounds like sirens, doorbells, or a baby crying.

### Speech Mode

This panel assists users who are unable to speak, allowing them to communicate using synthesized speech.

  * **Text-to-Speech (TTS):** Users can type text or use voice dictation, which is then converted into high-quality, spoken audio.
  * **Multi-Lingual & Multi-Voice:** Supports a wide range of languages (English, Hindi, Spanish, etc.) and various voice profiles (male/female) for personalization.
  * **Common Phrases:** Provides a quick-access list of common phrases like "I need help, please" or "Where is the restroom?".

### Sign Language Mode

This panel uses computer vision to interpret American Sign Language (ASL) gestures from a live video feed or an uploaded file.

  * **Real-Time Recognition:** Uses the user's webcam to capture video, processing frames to identify ASL gestures and append them to a running transcript.
  * **Ensemble AI Backend:** To ensure high accuracy, the backend uses an **Ensemble Predictor** that combines the results from multiple AI models:
    1.  **MediaPipe + Random Forest:** A lightweight model that uses hand landmarks for robust recognition.
    2.  **Vision Transformer (ViT):** A powerful deep learning model for image recognition.
    3.  **CNN (Convolutional Neural Network):** A custom-trained model for sign language classification.
  * **Enhanced Preprocessing:** All images are first processed with an `EnhancedPreprocessor` to auto-crop the hand, enhance contrast, and sharpen the image, significantly boosting model accuracy.

### Global Features

  * **Voice Commands:** The entire application can be navigated hands-free using voice commands like "Go to speech mode" or "Start listening".
  * **Multi-Language UI:** The user interface itself can be switched between multiple languages.

## 3\. Technology Stack

### Frontend

  * **Framework:** React 19
  * **Language:** TypeScript
  * **Bundler:** Vite
  * **Styling:** Tailwind CSS (inferred from class names in `.tsx` files)
  * **AI SDK:** `@google/genai` (for potential future use)

### Backend

  * **Framework:** Python 3.10+ with FastAPI
  * **Server:** Uvicorn
  * **Real-Time STT:** AssemblyAI
  * **Translation:** Google Translate API & Google Gemini API
  * **TTS:** gTTS (Google Text-to-Speech)

### AI / Machine Learning

  * **Computer Vision:** OpenCV
  * **Landmark Detection:** MediaPipe
  * **Deep Learning:** TensorFlow (Keras), PyTorch & Transformers (for ViT)
  * **Traditional ML:** Scikit-learn (Random Forest, PCA)
  * **Data Handling:** NumPy, Pandas, Joblib

## 4\. File Structure Explanation

This is a breakdown of the most important files and directories in the project.

### `windsurf-project/frontend/`

Contains the React.js client application.

  * `package.json`: Defines all Node.js dependencies (React, Vite) and project scripts (`dev`, `build`).
  * `vite.config.ts`: Configuration file for the Vite build tool.
  * `tsconfig.json`: TypeScript configuration for the frontend.
  * `index.html`: The main HTML entry point for the React app.
  * `src/index.tsx`: The root of the React application, mounts the main component.
  * `src/App.tsx`: The main app component. It manages the state for the current mode (Hearing, Speech, Sign) and handles voice command registration for navigation.
  * `src/components/`: Contains all reusable React components.
      * `HearingPanel.tsx`: The UI for Mode 1 (Transcription, Translation, Sound Alerts).
      * `SpeechPanel.tsx`: The UI for Mode 2 (Text-to-Speech, Dictation, Common Phrases).
      * `SignLanguagePanel.tsx`: The UI for Mode 3 (Real-time camera feed, recognition, and transcript).
      * `VoiceHelpPanel.tsx`: A floating panel that shows the user which voice commands are available for the current mode.
      * `Header.tsx`: The main navigation header to switch between the three modes.
  * `src/services/`: Contains logic for communicating with the backend and external services.
      * `gemini.ts`: A misleading name (likely a holdover), this file is the primary API client. It handles fetching TTS audio, requesting translations, and uploading videos/images for sign language recognition from the FastAPI backend.
      * `voiceCommands.ts`: A singleton service that manages all voice command registration and processing, using either the backend WebSocket or the browser's SpeechRecognition API as a fallback.
      * `websocketVoice.ts`: Manages the real-time audio streaming via WebSocket to the backend for transcription.
  * `src/contexts/LanguageContext.tsx`: A React Context that provides all UI text (translations for English, Hindi, Spanish, etc.) and allows changing the app's language.

### `windsurf-project/backend/`

Contains the Python FastAPI server that powers all AI/ML features.

  * `main.py`: The core of the backend. This FastAPI application defines all API endpoints:
      * `/healthz`: A simple health check.
      * `/tts`: Generates text-to-speech audio using `gTTS`.
      * `/translate_text`: Translates text using Gemini or Google Translate API.
      * `/ws/voice`: The WebSocket endpoint for real-time transcription via AssemblyAI.
      * `/api/recognize-asl`: Endpoint for the simple CNN model.
      * `/api/recognize-asl-vit`: Endpoint for the Vision Transformer model.
      * `/api/recognize-asl-mediapipe`: Endpoint for the MediaPipe model.
      * `/api/recognize-asl-ensemble`: The primary endpoint that runs all models and returns the most accurate result.
  * `requirements.txt`: A list of all Python dependencies (FastAPI, Uvicorn, TensorFlow, PyTorch, MediaPipe, AssemblyAI, etc.).
  * `ensemble_predictor.py`: A key file that imports all other models (`predict_asl`, `predict_vit`, `improved_mediapipe_predictor`) and combines their predictions using a weighted average for the best result.
  * `predict_*.py`: A series of files, each responsible for loading and running a specific ML model (e.g., `predict_asl.py` for the CNN, `predict_mediapipe.py` for MediaPipe, `improved_mediapipe_predictor.py` for the enhanced version).
  * `train_*.py`: Python scripts used to train the machine learning models (e.g., `train_simple_cnn.py`, `train_sign_language.py`).
  * `.env.example`: An example file showing which environment variables are needed.
  * `.env`: **(SECURITY RISK)** This file contains hard-coded API keys and should be deleted and invalidated immediately.

### Root Directory

  * `.gitignore`: A crucial file that tells Git to ignore files like `node_modules`, `__pycache__`, `.env`, and large model files (`*.pkl`, `*.h5`).
  * `.gitattributes`: Configures Git LFS (Large File Storage) for handling large model files.
 

## 5\. How to Run

### Backend (Python Server)

1.  **Navigate to the backend:**
    ```sh
    cd windsurf-project/backend
    ```
2.  **Create a virtual environment:**
    ```sh
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
4.  **Set up environment variables:**
      * Copy `.env.example` to a new file named `.env`.
      * Edit `.env` and add your API keys for:
          * `OPENAI_API_KEY` (For Whisper)
          * `GOOGLE_TRANSLATE_API_KEY` (For Google Translate)
          * `GOOGLE_API_KEY` (For Gemini)
          * `ASSEMBLYAI_API_KEY` (For real-time transcription)
5.  **Run the server:**
    ```sh
    python main.py
    ```
    The backend will be running at `http://127.0.0.1:8000`.

### Frontend (React App)

1.  **Navigate to the frontend (in a new terminal):**
    ```sh
    cd windsurf-project/frontend
    ```
2.  **Install dependencies:**
    ```sh
    npm install
    ```
3.  **Run the app:**
    ```sh
    npm run dev
    ```
    The frontend will be running at `http://localhost:3000` and will automatically connect to your backend at port 8000.

-----

