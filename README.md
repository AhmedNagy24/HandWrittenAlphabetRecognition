# Handwritten Alphabet Classification Using SVM and Neural Networks

This project trains models to recognize handwritten English alphabets (A–Z) using the **A–Z Handwritten Data** dataset. It compares traditional ML (SVM) with deep learning models (using TensorFlow/Keras).

## 📁 Dataset
- Dataset file: `A_Z Handwritten Data.csv`
- Each record: 784 pixel values (28x28 image) + 1 label (0–25 for A–Z)

## 📊 Steps Overview

### 1. Data Exploration & Preprocessing
- Load CSV data
- Normalize pixel values to `[0, 1]`
- Reshape flat images to 28×28 format
- Shuffle and split into:
  - 80% training
  - 20% testing

### 2. Experiment 1: SVM Classification
- **Linear SVM (LinearSVC)**
- **Non-linear SVM (RBF kernel)**

Metrics:
- Confusion Matrix
- Weighted F1 Score

### 3. Experiment 2: Neural Networks
- **Model 1 (Simple)**:  
  - 1 Hidden Layer (128 neurons, ReLU)
- **Model 2 (Complex)**:  
  - 2 Hidden Layers (256 and 128 neurons, ReLU)

Both use:
- `Flatten` input
- `Dense` output (26 neurons, softmax)
- Loss: `sparse_categorical_crossentropy`
- Optimizer: `adam`

Trained with:
- 90% of training data
- 10% for validation

Metrics:
- Accuracy & Loss (Training vs Validation)

### 4. Visualization
- Plot accuracy and loss curves for each model using `matplotlib`.

### 5. Model Selection & Saving
- Compare validation accuracies
- Save the best model as `best_model.keras`

### 6. Evaluation on Test Set
- Load best model
- Evaluate on testing set
- Report:
  - Test Accuracy
  - Confusion Matrix
  - Weighted F1 Score

### 7. Prediction on External Images
- 13 grayscale `.png` images (letters A to Z)
- Resize to 28x28
- Normalize and predict using the best model
- Display image and predicted label

## 📦 Dependencies

- `numpy`\n- `pandas`\n- `scikit-learn`\n- `tensorflow`\n- `matplotlib`\n- `PIL` (Python Imaging Library)
