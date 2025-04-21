
# 🌄 Intel Image Classification

This project performs image classification on the Intel Image Classification dataset using a Convolutional Neural Network (CNN) implemented in TensorFlow/Keras. The model is trained to classify images into various categories like buildings, forests, mountains, seas, streets, and glaciers.

## 📁 Repository Structure

```
intel_image_classification/
│
├── Dataset/                                 # Contains the image dataset (train/test/val folders)
├── intel_image_classification.ipynb         # Main Jupyter Notebook with the training and evaluation code
├── accuracy and loss function visualizaton.png  # Training/validation metrics visualization
├── README.md                                # This file
```

## 📊 Dataset

The dataset used in this project is the Intel Image Classification dataset, which contains natural scene images categorized into:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

Each category includes RGB images of 150x150 pixels.

You can download the dataset from [Intel Image Classification Dataset on Kaggle](https://www.kaggle.com/puneet6060/intel-image-classification) *(if not included due to size)*.

## 🚀 Model Architecture

- CNN with multiple Conv2D, MaxPooling2D layers
- Batch Normalization and Dropout for regularization
- Dense layers for classification
- Activation functions: ReLU and Softmax

## 📈 Training Performance

The training and validation accuracy/loss curves are available in the `accuracy and loss function visualizaton.png` image file. This helps visualize model convergence and detect overfitting.

## 🛠️ Dependencies

- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

You can install the dependencies using:

```bash
pip install -r requirements.txt
```

*(You can generate a `requirements.txt` with `pip freeze > requirements.txt`)*

## ✅ How to Run

1. Clone the repository
2. Place the dataset in the `Dataset/` folder
3. Open and run `intel_image_classification.ipynb` in Jupyter Notebook or Google Colab

