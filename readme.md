# 📄 Few-Shot Medical Image Classification with Explainability

An end-to-end deep learning system for **few-shot medical image classification**, designed for scenarios with limited labeled data. The model uses **Prototypical Networks with a ResNet-18 encoder**, trained on hospital-provided eye disease data, and integrates **Grad-CAM explainability** for visual interpretation of predictions.

---

## 🚀 Overview

Medical datasets are often small, imbalanced, and expensive to label. This project addresses that challenge using **few-shot learning**, enabling the model to generalize to new classes with very few examples.

The system learns an embedding space where images from the same class cluster together. During inference, it classifies new samples by comparing them to class prototypes derived from a small support set.

---

## ✨ Key Features

### 🧠 Few-Shot Learning with Prototypical Networks

* Implements N-way K-shot episodic training
* Learns feature embeddings instead of fixed classifiers
* Generalizes to new classes with minimal data

---

### 🏥 Medical Dataset Integration

* Dataset sourced directly from a hospital
* Focused on eye disease classification
* Designed for real-world clinical relevance

---

### 🔍 End-to-End Training

* ResNet-18 used as the encoder backbone
* Fully trained within the Prototypical Network framework
* Optimized using episodic sampling strategy

---

### 📊 Distance-Based Inference

* Computes class prototypes from support set
* Uses Euclidean distance in embedding space
* Predicts class based on nearest prototype

---

### 🔬 Explainable AI with Grad-CAM

* Generates heatmaps highlighting important regions
* Helps interpret model decisions
* Useful for medical validation and trust

---

### 🖥️ Interactive Testing Interface

* GUI-based interface for inference
* Upload/query images and visualize predictions
* Displays Grad-CAM outputs alongside results

---

## 🧠 How It Works

### Training Phase

1. Sample an episode with N classes and K support images per class
2. Pass all images through the encoder to obtain embeddings
3. Compute class prototypes as mean embeddings
4. Classify query images based on distance to prototypes
5. Update model weights using episodic loss

---

### Inference Phase

1. Provide a small labeled support set
2. Compute class prototypes
3. Pass query image through encoder
4. Compare with prototypes using Euclidean distance
5. Output predicted class

---

### Explainability

* Grad-CAM is applied to the encoder
* Produces activation heatmaps for predictions
* Highlights regions influencing the decision

---

## ⚙️ Tech Stack

* Deep Learning: PyTorch
* Model Architecture: ResNet-18
* Few-Shot Framework: Prototypical Networks
* Explainability: Grad-CAM
* Interface: Python GUI
* Data Handling: Custom episodic dataset loader

---

## 📂 Project Structure

```
project/
│
├── train.py                 # Episodic training pipeline
├── model.py                 # Prototypical Network + encoder
├── dataset.py               # Episodic dataset generation
├── inference.py             # Few-shot inference logic
├── gradcam.py               # Explainability module
├── gui.py                   # Testing interface
│
├── weights/                 # Saved model checkpoints
├── data/                    # Medical dataset
│
├── requirements.txt
└── README.md
```

---

## 📊 Model Details

* Backbone: ResNet-18
* Training Strategy: Episodic training
* Loss Function: Distance-based classification loss
* Metric: Euclidean distance in embedding space

---

## 📈 Use Cases

* Medical image classification with limited data
* Rare disease detection
* Clinical decision support systems
* Research in low-data deep learning

---

## 📌 Future Improvements

* Extend to multi-modal medical data
* Add uncertainty estimation
* Deploy as a web or mobile application
* Integrate with hospital systems for real-time use

---

## ⚠️ Notes

* Dataset is sourced from a hospital and may be restricted
* Ensure proper preprocessing for medical images
* GPU recommended for training

---

## 📜 License

For research and academic use only.

---

## 🎯 Project Vision

This project aims to bridge the gap between **limited medical data and reliable AI systems**, combining few-shot learning and explainability to build models that are both accurate and interpretable.


