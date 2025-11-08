# 🚀 AssemblyAI Quick Start - 5 Minutes Setup

## Step 1: Get API Key (2 minutes)
1. Visit: https://www.assemblyai.com/
2. Click "Sign Up" → Create account
3. Go to Dashboard → Copy your API key

## Step 2: Configure (1 minute)
Open `backend/.env` and add:
```env
ASSEMBLYAI_API_KEY=paste_your_key_here
```

## Step 3: Install (1 minute)
```bash
cd backend
pip install assemblyai
```

## Step 4: Verify (30 seconds)
```bash
python setup_assemblyai.py
```

You should see:
```
✅ AssemblyAI Setup Complete!
🎤 Ready to start voice recognition!
```

## Step 5: Run (30 seconds)
```bash
uvicorn main:app --reload
```

## ✅ Done!
Your real-time speech recognition is now powered by AssemblyAI!

---

## 🎤 Test It

1. Open frontend: `http://localhost:5173`
2. Click microphone button
3. Say: "Go to hearing mode"
4. Watch it work in real-time!

## 📚 Learn More

- **Full Guide**: `ASSEMBLYAI_MIGRATION_GUIDE.md`
- **Summary**: `ASSEMBLYAI_SETUP_SUMMARY.md`
- **Docs**: https://www.assemblyai.com/docs

## 🐛 Issues?

Run diagnostics:
```bash
python setup_assemblyai.py
```

Check logs:
- Backend: Terminal where uvicorn is running
- Frontend: Browser console (F12)

## 💡 Key Commands

```bash
# Verify setup
python setup_assemblyai.py

# Start backend
uvicorn main:app --reload

# Start frontend
cd frontend && npm run dev

# Install dependencies
pip install -r requirements.txt
```

---

**That's it!** You're ready to use AssemblyAI for real-time speech recognition! 🎉
