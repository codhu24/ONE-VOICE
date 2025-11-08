"""
Quick script to verify the Sign Language MNIST dataset.
"""
import os
import pandas as pd
import numpy as np

# Paths
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
TRAIN_CSV = os.path.join(DATASET_DIR, "sign_mnist_train.csv")
TEST_CSV = os.path.join(DATASET_DIR, "sign_mnist_test.csv")

# Label mapping
LABEL_MAP = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
    18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
}

def check_dataset():
    """Check dataset files and print statistics."""
    print("="*60)
    print("Sign Language MNIST Dataset Verification")
    print("="*60)
    
    # Check if files exist
    print("\n1. Checking files...")
    if not os.path.exists(TRAIN_CSV):
        print(f"❌ Training file not found: {TRAIN_CSV}")
        return False
    print(f"✅ Training file found: {TRAIN_CSV}")
    
    if not os.path.exists(TEST_CSV):
        print(f"❌ Test file not found: {TEST_CSV}")
        return False
    print(f"✅ Test file found: {TEST_CSV}")
    
    # Load training data
    print("\n2. Loading training data...")
    try:
        train_df = pd.read_csv(TRAIN_CSV)
        print(f"✅ Loaded {len(train_df)} training samples")
        print(f"   Shape: {train_df.shape}")
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return False
    
    # Load test data
    print("\n3. Loading test data...")
    try:
        test_df = pd.read_csv(TEST_CSV)
        print(f"✅ Loaded {len(test_df)} test samples")
        print(f"   Shape: {test_df.shape}")
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return False
    
    # Check data format
    print("\n4. Checking data format...")
    expected_cols = 785  # 1 label + 784 pixels
    if train_df.shape[1] != expected_cols:
        print(f"❌ Expected {expected_cols} columns, got {train_df.shape[1]}")
        return False
    print(f"✅ Correct format: 1 label + 784 pixels")
    
    # Check labels
    print("\n5. Analyzing labels...")
    train_labels = train_df.iloc[:, 0].values
    test_labels = test_df.iloc[:, 0].values
    
    unique_train = sorted(set(train_labels))
    unique_test = sorted(set(test_labels))
    
    print(f"   Training set labels: {unique_train}")
    print(f"   Test set labels: {unique_test}")
    print(f"   Number of classes: {len(unique_train)}")
    
    # Class distribution
    print("\n6. Class distribution (training set):")
    from collections import Counter
    label_counts = Counter(train_labels)
    for label in sorted(label_counts.keys()):
        letter = LABEL_MAP.get(label, '?')
        count = label_counts[label]
        print(f"   {letter} (label {label:2d}): {count:5d} samples")
    
    # Check pixel values
    print("\n7. Checking pixel values...")
    pixels = train_df.iloc[:, 1:].values
    print(f"   Min pixel value: {pixels.min()}")
    print(f"   Max pixel value: {pixels.max()}")
    print(f"   Mean pixel value: {pixels.mean():.2f}")
    
    if pixels.min() < 0 or pixels.max() > 255:
        print(f"❌ Pixel values out of range [0, 255]")
        return False
    print(f"✅ Pixel values in valid range")
    
    # Visualize a sample (if matplotlib available)
    print("\n8. Sample visualization...")
    try:
        import matplotlib.pyplot as plt
        
        # Show first 10 samples
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        axes = axes.ravel()
        
        for i in range(10):
            label = train_labels[i]
            letter = LABEL_MAP.get(label, '?')
            img = pixels[i].reshape(28, 28)
            
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"{letter} (label {label})")
            axes[i].axis('off')
        
        plt.tight_layout()
        sample_path = os.path.join(DATASET_DIR, 'dataset_samples.png')
        plt.savefig(sample_path)
        print(f"✅ Sample visualization saved to: {sample_path}")
        plt.close()
    except ImportError:
        print("   ℹ️  matplotlib not installed, skipping visualization")
    except Exception as e:
        print(f"   ⚠️  Could not create visualization: {e}")
    
    print("\n" + "="*60)
    print("✅ Dataset verification complete!")
    print("="*60)
    print(f"\nSummary:")
    print(f"  - Training samples: {len(train_df)}")
    print(f"  - Test samples: {len(test_df)}")
    print(f"  - Classes: {len(unique_train)} (ASL letters A-Y, excluding J and Z)")
    print(f"  - Image size: 28x28 pixels")
    print(f"\nReady to train! Run: python train_sign_language.py")
    
    return True

if __name__ == "__main__":
    success = check_dataset()
    exit(0 if success else 1)
