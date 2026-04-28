# 💎 Jewelry Detection Model

This repository contains a Deep Learning model built with TensorFlow and Keras to detect and classify jewelry images, specifically focusing on **Necklaces** and **Rings**.

## 🚀 Key Features
- **CNN Architecture**: Custom 5-block Convolutional Neural Network.
- **Image Preprocessing**: Built-in data augmentation (rotation, zoom, flips).
- **Optimized Training**: Includes EarlyStopping and Learning Rate Reduction.
- **High Accuracy**: Designed for robust classification in varied lighting and backgrounds.

## 🛠️ Technology Stack
- **Framework**: TensorFlow 2.x / Keras
- **Libraries**: NumPy, Pandas, Matplotlib, PIL
- **Input Resolution**: 224x224 RGB images
- **Dataset**: `Jewellery_Data/` directory with categorized subfolders.

## 📁 Project Structure
- `Jewelry_Detection_Model.ipynb`: Complete training and evaluation pipeline.
- `Jewellery_Data/`: Training and validation image dataset.
- `best_jewelry_model.h5`: Pre-trained model weights (after execution).

## 🧩 Model Architecture
1. **Feature Extraction**: 5 sets of Conv2D + BatchNormalization + MaxPooling.
2. **Regularization**: Extensive use of Dropout (0.3 - 0.5) to prevent overfitting.
3. **Classification**: Dense layers with Softmax activation for multi-class output.

## 🚦 How to Use
1. **Training**: Open the `.ipynb` file in Google Colab or Jupyter.
2. **Data**: Ensure the `Jewellery_Data` folder is present in the workspace.
3. **Inference**:
   ```python
   from tensorflow.keras.models import load_model
   model = load_model('best_jewelry_model.h5')
   # Preprocess and predict on new images
   ```

## 📊 Evaluation
The model provides comprehensive evaluation metrics including:
- **Accuracy/Loss Plots** for training visualization.
- **Confusion Matrix** to identify misclassifications.
- **Classification Report** (Precision, Recall, F1-Score)..
