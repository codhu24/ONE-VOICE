# OneVoice - UI/UX Design Documentation

## Overview

OneVoice is designed with accessibility and user experience at its core. The application provides a clean, intuitive interface that works seamlessly with both mouse/touch interactions and voice commands, making it accessible to users with various abilities and preferences.

## Project Structure

### Frontend Architecture

```
frontend/
├── App.tsx                      # Main application component
├── AppWithLanguage.tsx          # Language context wrapper
├── index.tsx                    # Application entry point
├── index.html                   # HTML template
├── package.json                 # Node.js dependencies
├── tsconfig.json                # TypeScript configuration
├── vite.config.ts               # Vite build configuration
├── metadata.json                # App metadata
├── .env.example                 # Environment variables template
├── .env.local                   # Local environment variables
│
├── components/                  # React components
│   ├── Header.tsx               # Navigation header
│   ├── HearingPanel.tsx         # Hearing assistance panel
│   ├── SpeechPanel.tsx          # Speech assistance panel
│   ├── SignLanguagePanel.tsx    # Sign language recognition panel
│   └── VoiceHelpPanel.tsx       # Voice commands help panel
│
├── contexts/                    # React contexts
│   └── LanguageContext.tsx      # Multilingual support context
│
├── services/                    # Service modules
│   ├── gemini.ts                # API service layer
│   ├── voiceCommands.ts         # Voice command service
│   └── websocketVoice.ts        # WebSocket voice service
│
├── types.ts                     # TypeScript type definitions
└── constants.tsx                # UI constants and icons
```

### Backend Architecture

```
backend/
├── main.py                      # FastAPI backend application
├── requirements.txt             # Python dependencies
├── quick_start.py               # Quick start script
├── train_sign_language.py       # ML model training script
├── test_dataset.py              # Dataset testing utilities
├── sign_language_model.pkl      # Trained ML model
├── TRAINING_GUIDE.md            # ML training documentation
├── .env                         # Environment variables
├── .env.example                 # Environment variables template
├── static/                      # Static files (generated audio)
└── dataset/                     # Sign language training dataset
    ├── sign_mnist_train/        # Training data
    └── sign_mnist_test/         # Test data
```

## Design Philosophy

### Core Principles

1. **Accessibility First**: Every feature is accessible through both traditional UI controls and voice commands
2. **Visual Clarity**: Large text, high contrast, and clear visual feedback
3. **Intuitive Navigation**: Simple, consistent interface patterns across all modes
4. **Real-time Feedback**: Immediate visual and audio feedback for all user actions
5. **Multilingual Support**: Native language support with proper script rendering via LanguageContext
6. **Real-time Communication**: WebSocket integration for enhanced voice interaction
7. **Context-Aware**: Language context propagates throughout the application

## Color Scheme

### Mode-Based Themes

Each mode has a distinct color theme to help users quickly identify which section they're in:

- **Hearing Mode**: Blue to Purple gradient (`from-blue-500 to-purple-600`)
  - Primary: Blue (#3B82F6)
  - Accent: Purple (#9333EA)
  - Background: Dark gray with blue tint

- **Speech Mode**: Pink to Orange gradient (`from-pink-500 to-orange-500`)
  - Primary: Pink (#EC4899)
  - Accent: Orange (#F97316)
  - Background: Dark gray with pink tint

- **Sign Language Mode**: Purple to Indigo gradient (`from-purple-500 to-indigo-600`)
  - Primary: Purple (#A855F7)
  - Accent: Indigo (#4F46E5)
  - Background: Dark gray with purple tint

### General Color Palette

- **Background**: Dark gray (#1F2937, #111827) with gradient overlays
- **Text**: White (#FFFFFF) for primary text, Gray-300 (#D1D5DB) for secondary
- **Interactive Elements**: Mode-specific colors with hover states
- **Error States**: Red (#EF4444) for errors
- **Success States**: Green (#10B981) for success messages
- **Warning States**: Yellow (#F59E0B) for warnings

## Layout & Structure

### Main Layout

```
┌─────────────────────────────────────┐
│           Header (Sticky)           │
│  - OneVoice Logo/Title              │
│  - Mode Navigation (3 buttons)      │
│  - Active Mode Indicator            │
├─────────────────────────────────────┤
│                                     │
│      Voice Status Indicator         │
│      (Fixed Top Right)              │
│                                     │
├─────────────────────────────────────┤
│                                     │
│         Main Content Area           │
│    (Mode-specific panels)           │
│                                     │
├─────────────────────────────────────┤
│                                     │
│      Voice Help Panel               │
│      (Fixed Bottom Left)            │
│                                     │
└─────────────────────────────────────┘
```

### Header Component

**Purpose**: Primary navigation and mode selection

**Features**:
- Sticky positioning (remains visible while scrolling)
- Three mode buttons in a responsive grid
- Active mode highlighted with gradient background and ring
- Smooth hover animations (lift effect)
- Clear iconography for each mode
- Active mode text indicator below buttons

**Interaction**:
- Click/Tap: Switch between modes
- Voice: "Go to [mode] mode"
- Visual feedback: Scale animation, ring highlight

### Voice Status Indicator

**Location**: Fixed top-right corner

**Visual States**:
- **Active**: Green background with pulsing animation
  - White dot with ping animation
  - "Voice Active" text
- **Inactive**: Gray background
  - Gray dot (static)
  - "Voice Inactive" text

**Purpose**: Always-visible indicator of voice command system status

## Mode-Specific UI Components

### 1. Hearing Impaired Assistance Panel

#### Layout Structure
```
┌─────────────────────────────────────┐
│  Hearing Impaired Assistance        │
│  (Blue gradient theme)              │
├─────────────────────────────────────┤
│                                     │
│  Real-Time Subtitles Section        │
│  - Title                            │
│  - Description text                 │
│  - Start/Stop Listening Button      │
│  - Translation Toggle               │
│  - Language Selector (conditional)  │
│                                     │
├─────────────────────────────────────┤
│                                     │
│  Caption Display Area (Sticky)      │
│  - Large transcript text            │
│  - Translation text (conditional)   │
│  - Loading states                   │
│                                     │
└─────────────────────────────────────┘
```

#### Key Features

**Real-Time Subtitles Section**:
- Centered card with rounded corners
- Semi-transparent background with backdrop blur
- Large, prominent "Start Listening" button
  - Blue when inactive
  - Red with pulse animation when active
- Translation toggle switch
  - Smooth slide animation
  - Clear on/off states
- Language dropdown (appears when translation enabled)
  - Fade-in animation
  - Styled to match theme

**Caption Display Area**:
- Fixed at bottom of viewport
- Dark background with backdrop blur
- Large text (3xl) for readability
- White text on dark background (high contrast)
- Separate section for translations
  - Blue-tinted text
  - Border separator
  - Loading indicator (animated dot)

**Sound Alert Popup**:
- Full-screen overlay with semi-transparent background
- Centered alert card
- Icon representation of alert type
- Large, bold text
- Flash effect on appearance
- Haptic feedback (vibration)
- Auto-dismiss after 4 seconds
- Manual dismiss button

#### Visual Feedback

- **Listening State**: Button pulses, status text changes
- **Translating**: Animated dot indicator
- **Alerts**: Flash overlay, vibration, colored icons
- **Errors**: Red error messages

### 2. Speech Impaired Assistance Panel

#### Layout Structure
```
┌─────────────────────────────────────┐
│  Speech Impaired Assistance         │
│  (Pink gradient theme)              │
├─────────────────────────────────────┤
│                                     │
│  Configuration Section              │
│  - Language Selector                │
│  - Voice Selector                   │
│                                     │
├─────────────────────────────────────┤
│                                     │
│  Text Input Area                    │
│  - Large textarea                   │
│  - Voice Dictation Button           │
│  - Listening Indicator              │
│                                     │
├─────────────────────────────────────┤
│                                     │
│  Speak Button                       │
│  - Large, prominent                 │
│  - Gradient background              │
│  - Animated waves when speaking     │
│                                     │
├─────────────────────────────────────┤
│                                     │
│  Common Phrases Section             │
│  - Grid of phrase buttons           │
│  - Language-specific phrases        │
│                                     │
└─────────────────────────────────────┘
```

#### Key Features

**Configuration Section**:
- Two-column grid layout (responsive)
- Dropdown selectors for language and voice
- Consistent styling with theme colors
- Clear labels and options

**Text Input Area**:
- Large textarea (4 rows)
- Dark background with border
- Placeholder text with instructions
- Voice dictation button (mic icon)
  - Positioned in bottom-right corner
  - Pink when inactive
  - Red with pulse when active
- Listening indicator
  - Top-right corner
  - Red pulsing dot
  - "Listening..." text
- Supports both typing and voice input

**Speak Button**:
- Full-width, large button
- Pink to orange gradient
- Speaker icon
- Text changes to "Speaking..." when active
- Animated wave bars during speech
- Disabled state when no text or already speaking
- Hover scale effect

**Common Phrases Section**:
- Grid layout (2 columns on desktop, 1 on mobile)
- Language-specific phrases
- Click to select and speak
- Hover effects
- Disabled during speech

#### Visual Feedback

- **Speaking State**: Wave animation, button text change
- **Listening State**: Mic button pulses, indicator appears
- **Text Input**: Real-time updates from dictation
- **Errors**: Red error messages below button

### 3. Sign Language Recognition Panel

#### Layout Structure
```
┌─────────────────────────────────────┐
│  Sign Language Recognition          │
│  (Purple gradient theme)            │
├─────────────────────────────────────┤
│                                     │
│  Video Preview Area                 │
│  - Camera feed display              │
│  - Centered, responsive             │
│                                     │
├─────────────────────────────────────┤
│                                     │
│  Control Buttons                    │
│  - Start Recording                  │
│  - Stop Recording                   │
│  - Recognize Signs                  │
│                                     │
├─────────────────────────────────────┤
│                                     │
│  File Upload Section                │
│  - Upload video file option         │
│                                     │
├─────────────────────────────────────┤
│                                     │
│  Results Display                    │
│  - Recognized text                  │
│  - Confidence score                 │
│  - Processing time                  │
│                                     │
├─────────────────────────────────────┤
│                                     │
│  Instructions Section               │
│  - Usage instructions               │
│                                     │
└─────────────────────────────────────┘
```

#### Key Features

**Video Preview**:
- Live camera feed
- Centered with max width constraint
- Rounded corners
- Dark background
- Responsive sizing

**Control Buttons**:
- Color-coded actions
  - Blue: Start recording
  - Red: Stop recording
  - Green: Process/Recognize
- Emoji icons for visual clarity
- Disabled states during processing
- Hover effects

**Results Display**:
- Green-tinted success card
- Large, bold recognized text
- Confidence percentage
- Processing time metrics
- Clear visual hierarchy

**Instructions**:
- Blue-tinted information card
- Bullet-point list
- Clear, concise steps
- Helpful tips

#### Visual Feedback

- **Recording State**: Button color changes, video shows live feed
- **Processing State**: Button shows loading text, spinner
- **Results**: Green success card with results
- **Errors**: Red error messages

## Voice Help Panel

### Location
Fixed bottom-left corner

### Features
- Toggle button ("? Voice Commands")
- Expandable panel with command reference
- Context-aware command lists
- Scrollable content area
- Organized by category (Global, Mode-specific)

### Visual Design
- Blue button with hover effect
- Dark panel with backdrop blur
- Color-coded sections
- Monospace font for commands
- Clear descriptions

## Typography

### Font Family
- Primary: Lexend (Google Fonts)
- Fallback: System sans-serif
- Monospace: For code/commands

### Font Sizes
- **Headers**: 3xl (30px) - Bold
- **Subheaders**: 2xl (24px) - Semibold
- **Body Text**: lg (18px) - Regular
- **Captions**: xl (20px) - Medium
- **Large Display**: 3xl (30px) - Semibold (for transcripts)

### Font Weights
- Bold: Headers, important text
- Semibold: Subheaders, labels
- Medium: Secondary text
- Regular: Body text

## Spacing & Sizing

### Padding
- **Section Padding**: 6 (24px)
- **Card Padding**: 8 (32px)
- **Button Padding**: 4-8 (16-32px)
- **Input Padding**: 4 (16px)

### Margins
- **Section Margins**: 6-8 (24-32px)
- **Element Gaps**: 4-6 (16-24px)
- **Grid Gaps**: 3-4 (12-16px)

### Border Radius
- **Cards**: 2xl (16px)
- **Buttons**: lg (8px) to xl (12px)
- **Inputs**: lg (8px)
- **Badges**: full (rounded)

## Animations & Transitions

### Transition Effects
- **Color Changes**: 300ms ease
- **Transform**: 300ms ease
- **Opacity**: 500ms ease
- **Scale**: 105% on hover

### Animation Types
- **Pulse**: Status indicators, active states
- **Ping**: Voice status dot
- **Wave**: Speech animation bars
- **Fade-in**: New content appearance
- **Slide**: Toggle switches
- **Lift**: Button hover effect

### Animation Timing
- **Fast**: 100-200ms (hover states)
- **Medium**: 300-500ms (transitions)
- **Slow**: 1000ms+ (continuous animations)

## Responsive Design

### Breakpoints
- **Mobile**: < 768px (single column)
- **Tablet**: 768px - 1024px (2 columns)
- **Desktop**: > 1024px (3 columns)

### Responsive Features
- Grid layouts adapt to screen size
- Text sizes scale appropriately
- Buttons remain accessible on touch devices
- Modal/overlays adjust to viewport
- Navigation stacks on mobile

## Accessibility Features

### Keyboard Navigation
- All interactive elements are keyboard accessible
- Tab order follows logical flow
- Focus indicators visible
- Enter/Space for button activation

### Screen Reader Support
- Semantic HTML elements
- ARIA labels where needed
- Role attributes for custom components
- Alt text for icons (via titles)

### Visual Accessibility
- High contrast ratios (WCAG AA compliant)
- Large clickable areas (minimum 44x44px)
- Clear focus indicators
- Color is not the only indicator

### Voice Accessibility
- All features accessible via voice commands
- Voice feedback for actions
- Clear command patterns
- Help panel for command reference

## Interaction Patterns

### Button Interactions
1. **Hover**: Scale up slightly, color brighten
2. **Active**: Scale down slightly, color darken
3. **Disabled**: Reduced opacity, no interaction
4. **Loading**: Spinner or progress indicator

### Input Interactions
1. **Focus**: Border color changes, ring appears
2. **Active**: Cursor visible, text selectable
3. **Error**: Red border, error message
4. **Success**: Green border (where applicable)

### Modal/Overlay Interactions
1. **Open**: Fade in with backdrop
2. **Close**: Fade out, click outside or button
3. **Content**: Slide in from center
4. **Backdrop**: Semi-transparent, clickable

## User Flows

### Hearing Mode Flow
1. User lands on Hearing mode
2. Clicks "Start Listening" or says "Start listening"
3. Browser requests microphone permission
4. Transcripts appear in real-time
5. User can enable translation
6. Select target language
7. Translations appear below transcript
8. Sound alerts pop up when detected
9. User can stop listening anytime

### Speech Mode Flow
1. User lands on Speech mode
2. Selects language and voice (optional)
3. Enters text via typing or voice dictation
4. Clicks "Speak" or says "Speak this"
5. Text is translated to selected language (if needed)
6. Audio plays through speakers
7. User can use common phrases for quick access
8. Can change language/voice anytime

### Sign Language Mode Flow
1. User lands on Sign Language mode
2. Clicks "Start Recording" or says "Start recording"
3. Browser requests camera permission
4. Live video feed appears
5. User performs sign language gestures
6. Clicks "Stop Recording" or says "Stop recording"
7. Clicks "Recognize Signs" or says "Recognize signs"
8. Processing indicator appears
9. Results displayed with confidence score
10. User can record again or upload file

## Voice Command Integration

### Architecture

**Voice Command Service** (`services/voiceCommands.ts`):
- Centralized voice recognition management
- Command pattern matching
- Handler registration system
- Auto-recovery from errors
- Status callbacks

**WebSocket Voice Service** (`services/websocketVoice.ts`):
- Real-time voice communication
- WebSocket connection management
- Streaming voice data
- Enhanced voice processing

### Visual Feedback for Voice Commands
- Voice status indicator (always visible)
- Help panel with command reference
- Console logging for debugging
- No interrupting UI feedback
- Real-time connection status

### Voice Command Patterns
- Natural language recognition
- Multiple phrasings supported
- Context-aware commands
- Error handling and fallback
- WebSocket-based real-time processing

## Error Handling & Feedback

### Error States
- **Network Errors**: Red error messages
- **Permission Denied**: Clear instructions
- **API Errors**: User-friendly messages
- **Validation Errors**: Inline feedback

### Success States
- **Green indicators**: Successful actions
- **Confirmation messages**: Important actions
- **Progress indicators**: Long operations

### Loading States
- **Spinners**: Short operations
- **Progress bars**: Long operations
- **Skeleton screens**: Content loading
- **Disabled states**: Prevent double actions

## Performance Considerations

### Optimization Strategies
- Lazy loading of components
- Optimized images and assets
- Efficient re-renders
- Debounced input handling
- Cached API responses

### Loading Performance
- Fast initial load
- Progressive enhancement
- Graceful degradation
- Offline capability (where possible)

## Browser Compatibility

### Supported Browsers
- Chrome/Edge: Full support
- Firefox: Full support
- Safari: Full support (with some limitations)
- Mobile browsers: Responsive design

### Feature Detection
- Speech Recognition API
- WebRTC (camera/microphone)
- WebSocket support
- Modern CSS features

## Component Details

### Core Components

#### `App.tsx`
- Main application component
- Manages application state and mode switching
- Integrates voice command service
- Provides navigation between panels
- Displays voice command status indicator

#### `AppWithLanguage.tsx`
- Wraps the main app with LanguageContext
- Provides multilingual support throughout the application
- Enables language switching and translation features

#### `Header.tsx`
- Navigation header component
- Mode selection buttons (Hearing, Speech, Sign Language)
- Visual indicators for active mode
- Responsive design with gradient themes

#### `HearingPanel.tsx`
- Real-time speech-to-text transcription
- Translation toggle and language selection
- Sound alert detection and display
- Voice commands for all controls
- Sticky caption display area

#### `SpeechPanel.tsx`
- Text input for speech generation
- Language and voice selection (8 languages, 5 voices)
- Common phrase library with language-specific phrases
- Text-to-speech with automatic translation
- Voice dictation support
- Voice commands for all actions

#### `SignLanguagePanel.tsx`
- Video recording interface
- Sign language gesture recognition
- Result display with confidence scores
- File upload option
- Voice commands for recording control

#### `VoiceHelpPanel.tsx`
- Displays available voice commands
- Context-aware command lists
- Toggleable help interface
- Fixed bottom-left positioning

### Context Providers

#### `LanguageContext.tsx`
- Manages multilingual support
- Provides translation functions
- Language state management
- Supports 8+ languages
- Integrates with Google Translate API
- Optional Gemini AI integration

### Service Modules

#### `services/gemini.ts`
- API service layer for backend communication
- `generateTextToSpeech()` - TTS API calls
- `translateText()` - Translation API calls
- `recognizeSignLanguage()` - Sign language API calls
- `playBase64Audio()` - Audio playback utilities
- Error handling and retry logic

#### `services/voiceCommands.ts`
- Centralized voice recognition management
- Command pattern matching with regex
- Handler registration system
- Auto-recovery from errors
- Status callbacks for UI updates
- Context-aware command processing

#### `services/websocketVoice.ts`
- WebSocket connection management
- Real-time voice data streaming
- Connection state handling
- Automatic reconnection logic
- Enhanced voice processing capabilities

### Type Definitions

#### `types.ts`
- `Mode` enum for application modes (Hearing, Speech, SignLanguage)
- `SoundAlertInfo` interface for alert data
- Shared type definitions across components

#### `constants.tsx`
- UI constants and configuration
- SVG icon components (Ear, Eye, Speech, Hand, Mic, etc.)
- Reusable UI elements
- Color schemes and theme definitions

## Backend Integration

### API Endpoints

**FastAPI Backend** (`backend/main.py`):
- `POST /translate_audio` - Audio translation pipeline
- `POST /tts` - Text-to-speech with translation
- `POST /translate_text` - Text translation
- `POST /sign_language_to_text` - Sign language recognition
- `GET /healthz` - Health check endpoint
- Static file serving for generated audio

### Machine Learning

**Sign Language Model**:
- Trained model: `sign_language_model.pkl`
- Training script: `train_sign_language.py`
- Dataset testing: `test_dataset.py`
- Training guide: `TRAINING_GUIDE.md`
- OpenCV-based gesture recognition
- Confidence scoring system

## Configuration Files

### Frontend Configuration

- **package.json**: Node.js dependencies and scripts
- **tsconfig.json**: TypeScript compiler options
- **vite.config.ts**: Vite build configuration
- **.env.local**: Local environment variables
- **metadata.json**: Application metadata

### Backend Configuration

- **requirements.txt**: Python dependencies
- **.env**: Environment variables (API keys)
- **quick_start.py**: Quick setup script

## Future Enhancements

### Planned UI Improvements
- Dark/light theme toggle
- Customizable color schemes
- Font size adjustments
- Reduced motion options
- Additional language support
- Enhanced animations
- Better mobile experience
- Improved WebSocket reliability
- Enhanced voice command accuracy

### Planned Features
- Offline mode support
- User preference persistence
- Advanced sign language gestures
- Multi-user collaboration
- Enhanced ML model accuracy
- Additional voice options
- Custom phrase libraries

---

**Last Updated**: November 2024
**Version**: 1.1.0
**Architecture**: React + TypeScript + FastAPI + ML

