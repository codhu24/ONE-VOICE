"""
Simple ASL CNN Training - Works with Your Data!
Train a CNN model from Sign MNIST CSV files
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import json

# Try imports
try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__} loaded")
except:
    print("✗ TensorFlow not available")
    exit(1)

try:
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except:
    PLOT_AVAILABLE = False
    print("Warning: matplotlib not available, skipping plots")

print("=" * 70)
print("Simple ASL CNN Training")
print("=" * 70)

# Configuration
IMG_SIZE = 64  # Reduced from 200 for faster training
BATCH_SIZE = 64
EPOCHS = 20
TRAIN_CSV = 'dataset/sign_mnist_train.csv'
TEST_CSV = 'dataset/sign_mnist_test.csv'
MODEL_SAVE_PATH = 'models/asl_model.h5'

LABEL_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

print(f"\nSettings:")
print(f"- Image size: {IMG_SIZE}x{IMG_SIZE} (faster training)")
print(f"- Epochs: {EPOCHS}")
print(f"- Batch size: {BATCH_SIZE}")
print(f"- Classes: {len(LABEL_NAMES)}")

# Create models directory
os.makedirs('models', exist_ok=True)

# Load data
print("\nLoading data...")
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
print(f"✓ Train: {len(train_df)} samples")
print(f"✓ Test: {len(test_df)} samples")

# Preprocess
print("\nPreprocessing...")
def prep_data(df):
    labels = df.iloc[:, 0].values
    pixels = df.iloc[:, 1:].values / 255.0
    images = pixels.reshape(-1, 28, 28, 1)
    return images, labels

X_train_28, y_train = prep_data(train_df)
X_test_28, y_test = prep_data(test_df)

# Check unique labels
unique_labels = np.unique(np.concatenate([y_train, y_test]))
print(f"✓ Unique labels in data: {unique_labels}")
print(f"✓ Number of classes: {len(unique_labels)}")

# Sign MNIST uses 0-24 for 25 classes (A-Z excluding J)
# We need to use 25 classes, not 24
NUM_CLASSES = len(unique_labels)

# Resize using TensorFlow
print(f"Resizing images to {IMG_SIZE}x{IMG_SIZE}...")
X_train_full = tf.image.resize(X_train_28, [IMG_SIZE, IMG_SIZE]).numpy()
X_test = tf.image.resize(X_test_28, [IMG_SIZE, IMG_SIZE]).numpy()

# Convert grayscale to RGB
X_train_full = np.repeat(X_train_full, 3, axis=-1)
X_test = np.repeat(X_test, 3, axis=-1)

# Split train/val
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train, test_size=0.2, random_state=42
)

print(f"✓ Training: {X_train.shape}")
print(f"✓ Validation: {X_val.shape}")
print(f"✓ Test: {X_test.shape}")

# Build model
print("\nBuilding CNN model...")
model = tf.keras.Sequential([
    # Conv Block 1
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Dropout(0.25),
    
    # Conv Block 2
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Dropout(0.25),
    
    # Conv Block 3
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Dropout(0.25),
    
    # Dense layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(24, activation='softmax')  # 24 classes
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model built")
print(f"\nTotal parameters: {model.count_params():,}")

# Train
print("\n" + "=" * 70)
print("Starting Training (this will take 15-45 minutes)")
print("=" * 70)

history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5)
    ],
    verbose=1
)

# Evaluate
print("\nEvaluating...")
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n✓ Test Accuracy: {test_acc*100:.2f}%")

# Save
print("\nSaving model...")
model.save(MODEL_SAVE_PATH)
print(f"✓ Model saved to {MODEL_SAVE_PATH}")

# Save class indices
class_indices = {name: idx for idx, name in enumerate(LABEL_NAMES)}
with open('models/class_indices.json', 'w') as f:
    json.dump(class_indices, f, indent=2)
print(f"✓ Class indices saved")

# Plot
if PLOT_AVAILABLE:
    print("\nGenerating plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Val')
    ax1.set_title('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_title('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    print(f"✓ Plot saved to models/training_history.png")
    plt.close()

print("\n" + "=" * 70)
print("🎉 TRAINING COMPLETE!")
print("=" * 70)
print(f"✓ Model: {MODEL_SAVE_PATH}")
print(f"✓ Accuracy: {test_acc*100:.2f}%")
print(f"✓ Size: ~15-20MB")
print("\nNext steps:")
print("1. Restart server: python main.py")
print("2. Should see: '✓ CNN model loaded'")
print("3. Test in browser - accuracy should be 85-90%!")
print("=" * 70)
