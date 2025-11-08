"""
Quick start script for sign language model training and testing.
Run this to set up everything automatically.
"""
import os
import sys
import subprocess

def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def check_dependencies():
    """Check if required packages are installed."""
    print_header("Checking Dependencies")
    
    required = {
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'cv2': 'opencv-python',
        'numpy': 'numpy'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is NOT installed")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nInstall them with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print("\n✅ All dependencies are installed!")
    return True

def verify_dataset():
    """Verify the dataset exists and is valid."""
    print_header("Verifying Dataset")
    
    try:
        result = subprocess.run(
            [sys.executable, 'test_dataset.py'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print("❌ Dataset verification failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error verifying dataset: {e}")
        return False

def train_model():
    """Train the sign language model."""
    print_header("Training Model")
    print("This will take 2-5 minutes. Please wait...\n")
    
    try:
        result = subprocess.run(
            [sys.executable, 'train_sign_language.py'],
            timeout=600  # 10 minutes max
        )
        
        if result.returncode == 0:
            print("\n✅ Model training complete!")
            return True
        else:
            print("\n❌ Model training failed")
            return False
    except subprocess.TimeoutExpired:
        print("\n❌ Training timed out (took more than 10 minutes)")
        return False
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        return False

def check_model():
    """Check if the trained model exists."""
    model_path = os.path.join(os.path.dirname(__file__), 'sign_language_model.pkl')
    
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ Model file exists: {model_path}")
        print(f"   Size: {size_mb:.2f} MB")
        return True
    else:
        print(f"❌ Model file not found: {model_path}")
        return False

def main():
    """Main quick start workflow."""
    print_header("Sign Language Model - Quick Start")
    print("This script will:")
    print("  1. Check dependencies")
    print("  2. Verify dataset")
    print("  3. Train the model")
    print("  4. Verify the trained model")
    
    input("\nPress Enter to continue...")
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first.")
        print("Run: pip install -r requirements.txt")
        return False
    
    # Step 2: Verify dataset
    if not verify_dataset():
        print("\n❌ Dataset verification failed.")
        print("Please check that dataset files exist in the dataset/ folder.")
        return False
    
    # Step 3: Check if model already exists
    if check_model():
        print("\n⚠️  Model already exists!")
        response = input("Do you want to retrain? (y/n): ").lower()
        if response != 'y':
            print("\nSkipping training. Using existing model.")
            print_success()
            return True
    
    # Step 4: Train model
    if not train_model():
        print("\n❌ Training failed. Please check the error messages above.")
        return False
    
    # Step 5: Verify trained model
    print_header("Verifying Trained Model")
    if not check_model():
        print("\n❌ Model file was not created. Training may have failed.")
        return False
    
    print_success()
    return True

def print_success():
    """Print success message with next steps."""
    print_header("✅ Setup Complete!")
    print("Your sign language model is ready to use!\n")
    print("Next steps:")
    print("  1. Start the backend:")
    print("     uvicorn main:app --reload")
    print("\n  2. Start the frontend:")
    print("     cd ../frontend")
    print("     npm run dev")
    print("\n  3. Open your browser and click 'Sign Language' mode")
    print("\n  4. Record or upload a video showing an ASL letter")
    print("\n  5. Click 'Recognize Signs' to see the result!")
    print("\nFor more information:")
    print("  - Training details: TRAINING_GUIDE.md")
    print("  - Setup guide: ../SIGN_LANGUAGE_SETUP.md")
    print("  - ML summary: ../ML_TRAINING_SUMMARY.md")

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
