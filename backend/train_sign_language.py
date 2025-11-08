"""
Train a sign language recognition model using the Sign Language MNIST dataset.
Dataset: 24 classes (A-Y, excluding J and Z which require motion)
"""
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import cv2

# Paths
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
TRAIN_CSV = os.path.join(DATASET_DIR, "sign_mnist_train.csv")
TEST_CSV = os.path.join(DATASET_DIR, "sign_mnist_test.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "sign_language_model.pkl")

# Label mapping (0-25, excluding J=9 and Z=25)
LABEL_MAP = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
    18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
}

def load_data(csv_path):
    """Load and preprocess data from CSV."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # First column is label, rest are pixels
    labels = df.iloc[:, 0].values
    pixels = df.iloc[:, 1:].values
    
    # Normalize pixel values to [0, 1]
    pixels = pixels.astype('float32') / 255.0
    
    print(f"Loaded {len(labels)} samples")
    return pixels, labels

def extract_features(pixels):
    """Extract features from raw pixels."""
    # For now, use raw pixels as features
    # You can add more sophisticated feature extraction here
    return pixels

def train_model():
    """Train the sign language recognition model."""
    print("="*60)
    print("Sign Language Recognition Model Training")
    print("="*60)
    
    # Load training data
    X_train, y_train = load_data(TRAIN_CSV)
    print(f"Training set: {X_train.shape}")
    
    # Load test data
    X_test, y_test = load_data(TEST_CSV)
    print(f"Test set: {X_test.shape}")
    
    # Extract features
    print("\nExtracting features...")
    X_train_features = extract_features(X_train)
    X_test_features = extract_features(X_test)
    
    # Train Random Forest classifier
    print("\nTraining Random Forest classifier...")
    print("This may take a few minutes...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,  # Use all CPU cores
        verbose=1
    )
    
    model.fit(X_train_features, y_train)
    
    # Evaluate on training set
    print("\n" + "="*60)
    print("Training Set Performance")
    print("="*60)
    y_train_pred = model.predict(X_train_features)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_accuracy*100:.2f}%")
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Test Set Performance")
    print("="*60)
    y_test_pred = model.predict(X_test_features)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=[LABEL_MAP.get(i, f'Class_{i}') for i in sorted(set(y_test))]))
    
    # Save model
    print(f"\nSaving model to {MODEL_PATH}...")
    model_data = {
        'model': model,
        'label_map': LABEL_MAP,
        'accuracy': test_accuracy
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    
    return model, test_accuracy

def visualize_predictions(num_samples=10):
    """Visualize some predictions from the test set."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping visualization.")
        return
    
    # Load model
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['model']
    label_map = model_data['label_map']
    
    # Load test data
    X_test, y_test = load_data(TEST_CSV)
    
    # Make predictions
    y_pred = model.predict(X_test[:num_samples])
    
    # Visualize
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        # Reshape to 28x28
        img = X_test[i].reshape(28, 28)
        
        # Plot
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"True: {label_map.get(y_test[i], '?')}\nPred: {label_map.get(y_pred[i], '?')}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(DATASET_DIR, 'predictions.png'))
    print(f"Visualization saved to {os.path.join(DATASET_DIR, 'predictions.png')}")
    plt.show()

if __name__ == "__main__":
    # Train the model
    model, accuracy = train_model()
    
    # Optionally visualize predictions
    try:
        visualize_predictions()
    except Exception as e:
        print(f"Could not visualize: {e}")
