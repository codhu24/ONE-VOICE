# Sign Language Model Training Guide

## Dataset Overview

Your `dataset/` folder contains the **Sign Language MNIST** dataset:

- **Training set**: 27,455 images (`sign_mnist_train.csv`)
- **Test set**: 7,172 images (`sign_mnist_test.csv`)
- **Classes**: 24 ASL letters (A-Y, excluding J and Z which require motion)
- **Image size**: 28x28 grayscale pixels
- **Format**: CSV with label in first column, 784 pixel values in remaining columns

### Label Mapping
```
0=A, 1=B, 2=C, 3=D, 4=E, 5=F, 6=G, 7=H, 8=I,
10=K, 11=L, 12=M, 13=N, 14=O, 15=P, 16=Q, 17=R,
18=S, 19=T, 20=U, 21=V, 22=W, 23=X, 24=Y
```
*(Note: J=9 and Z=25 are excluded as they require motion)*

## Training the Model

### Step 1: Install Dependencies

```bash
cd backend
pip install scikit-learn pandas opencv-python numpy
```

### Step 2: Train the Model

```bash
python train_sign_language.py
```

This will:
1. Load the training and test datasets
2. Train a Random Forest classifier (100 trees)
3. Evaluate on test set
4. Save the trained model to `sign_language_model.pkl`

**Expected Output:**
```
Training Accuracy: ~99%
Test Accuracy: ~85-90%
```

**Training Time:** ~2-5 minutes on modern hardware

### Step 3: Verify Model

The trained model will be automatically loaded by the backend when it starts:

```bash
uvicorn main:app --reload
```

Look for this message in the logs:
```
Loaded ML model with 87.5% accuracy
```

## How It Works

### Training Pipeline

```
CSV Data → Load & Normalize → Extract Features → Train Random Forest → 
Evaluate → Save Model
```

### Inference Pipeline

```
Video Frame → Detect Hand → Extract 28x28 Region → Normalize → 
ML Model Prediction → Letter Recognition
```

### Model Architecture

- **Algorithm**: Random Forest Classifier
- **Trees**: 100
- **Max Depth**: 20
- **Features**: 784 (28x28 pixels)
- **Output**: 24 classes (ASL letters A-Y)

## Using the Trained Model

### From the Web Interface

1. Start backend: `uvicorn main:app --reload`
2. Start frontend: `npm run dev`
3. Click "Sign Language" mode
4. Record or upload a video showing an ASL letter
5. Click "Recognize Signs"
6. Result will show: `"Letter: A"` with confidence score

### From the API

```bash
curl -X POST http://127.0.0.1:8000/sign_language_to_text \
  -F "file=@sign_video.webm"
```

**Response:**
```json
{
  "recognizedText": "Letter: A",
  "confidence": 0.92,
  "processingTimeMs": 1234
}
```

## Model Performance

### Accuracy by Letter

The model performs best on:
- **Simple shapes**: A, B, C, O, S
- **Distinct gestures**: L, V, W, Y

May struggle with:
- **Similar shapes**: M vs N, U vs V
- **Complex gestures**: G, H, P, Q

### Improving Accuracy

1. **Better lighting**: Ensure even, bright lighting
2. **Plain background**: Use a solid color background
3. **Clear gestures**: Hold the sign steady for 1-2 seconds
4. **Camera angle**: Face the camera directly
5. **Hand position**: Center hand in frame

## Advanced: Improving the Model

### 1. Use a Better Algorithm

Replace Random Forest with a neural network:

```python
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    max_iter=50,
    random_state=42
)
```

### 2. Add Data Augmentation

```python
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import rotate, shift

# Augment training data with rotations and shifts
# This can improve generalization
```

### 3. Use Deep Learning

For production, consider using TensorFlow/PyTorch:

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(24, activation='softmax')
])
```

### 4. Fine-tune on Your Own Data

Collect your own sign language videos and add them to the training set:

1. Record videos of yourself making ASL signs
2. Extract frames and label them
3. Add to `sign_mnist_train.csv`
4. Retrain the model

## Troubleshooting

### Model Not Loading

**Error**: `Could not load ML model`

**Solutions**:
- Ensure `sign_language_model.pkl` exists in `backend/`
- Check file permissions
- Verify pickle is installed: `pip install pickle-mixin`

### Low Accuracy

**Symptoms**: Model predicts wrong letters frequently

**Solutions**:
1. **Retrain with more trees**:
   ```python
   model = RandomForestClassifier(n_estimators=200)
   ```

2. **Adjust preprocessing**:
   - Check skin detection HSV ranges
   - Verify hand extraction is working
   - Ensure 28x28 resize maintains aspect ratio

3. **Collect more data**:
   - Add your own training examples
   - Use data augmentation

### Slow Inference

**Symptoms**: Video processing takes too long

**Solutions**:
1. **Process fewer frames**:
   ```python
   if frame_count % 10 != 0:  # Process every 10th frame instead of 5th
       continue
   ```

2. **Use a simpler model**:
   ```python
   model = RandomForestClassifier(n_estimators=50, max_depth=10)
   ```

3. **Resize video before processing**:
   ```python
   frame = cv2.resize(frame, (320, 240))
   ```

## Dataset Citation

If you use this dataset in research, please cite:

```
@misc{signlanguagemnist,
  title={Sign Language MNIST},
  author={tecperson},
  year={2017},
  publisher={Kaggle},
  howpublished={\url{https://www.kaggle.com/datasets/datamunge/sign-language-mnist}}
}
```

## Next Steps

1. ✅ Train the basic model
2. ✅ Test with your own videos
3. 🔄 Collect your own training data
4. 🔄 Experiment with different algorithms
5. 🔄 Deploy to production

## Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Random Forest Explained](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [ASL Alphabet Guide](https://www.startasl.com/american-sign-language-alphabet/)
- [OpenCV Hand Detection](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)

## Support

For issues or questions:
1. Check `SIGN_LANGUAGE_SETUP.md` for general setup
2. Review this guide for training-specific issues
3. Check the model's classification report for per-class accuracy
