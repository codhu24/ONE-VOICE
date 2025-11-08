"""
Quick setup script for AssemblyAI integration
Run this to verify your AssemblyAI setup is working correctly
"""

import os
import sys
from dotenv import load_dotenv

def check_assemblyai_setup():
    """Check if AssemblyAI is properly configured"""
    print("🔍 Checking AssemblyAI Setup...\n")
    
    # Load environment variables
    load_dotenv()
    
    # Check 1: AssemblyAI package
    print("1. Checking AssemblyAI package...")
    try:
        import assemblyai as aai
        print("   ✅ AssemblyAI package installed")
        print(f"   📦 Version: {aai.__version__ if hasattr(aai, '__version__') else 'Unknown'}")
    except ImportError:
        print("   ❌ AssemblyAI package not found")
        print("   💡 Install with: pip install assemblyai>=0.17.0")
        return False
    
    # Check 2: API Key
    print("\n2. Checking API key...")
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        print("   ❌ ASSEMBLYAI_API_KEY not found in .env file")
        print("   💡 Add to .env: ASSEMBLYAI_API_KEY=your_key_here")
        return False
    
    if api_key == "your_assemblyai_api_key_here":
        print("   ⚠️  API key is still the placeholder value")
        print("   💡 Replace with your actual AssemblyAI API key")
        return False
    
    print(f"   ✅ API key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Check 3: Test API connection
    print("\n3. Testing API connection...")
    try:
        aai.settings.api_key = api_key
        
        # Try to create a transcriber (this validates the API key)
        transcriber = aai.RealtimeTranscriber(
            sample_rate=16000,
            encoding=aai.AudioEncoding.pcm_s16le,
        )
        print("   ✅ API key is valid")
        print("   ✅ Can create transcriber")
        
    except Exception as e:
        print(f"   ❌ API connection failed: {str(e)}")
        print("   💡 Check your API key at https://www.assemblyai.com/app")
        return False
    
    # Check 4: Other dependencies
    print("\n4. Checking other dependencies...")
    dependencies = {
        "fastapi": "FastAPI",
        "uvicorn": "Uvicorn",
        "websockets": "WebSockets",
        "numpy": "NumPy",
    }
    
    all_deps_ok = True
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"   ✅ {name} installed")
        except ImportError:
            print(f"   ❌ {name} not found")
            all_deps_ok = False
    
    if not all_deps_ok:
        print("\n   💡 Install all dependencies: pip install -r requirements.txt")
        return False
    
    # Success!
    print("\n" + "="*50)
    print("✅ AssemblyAI Setup Complete!")
    print("="*50)
    print("\n📝 Next steps:")
    print("   1. Start the backend: uvicorn main:app --reload")
    print("   2. Test the WebSocket at: ws://localhost:8000/ws/voice")
    print("   3. Check the migration guide: ASSEMBLYAI_MIGRATION_GUIDE.md")
    print("\n🎤 Ready to start voice recognition!")
    
    return True

if __name__ == "__main__":
    success = check_assemblyai_setup()
    sys.exit(0 if success else 1)
