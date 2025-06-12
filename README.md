# 🎸 Guitar Chord Recognition Using CNN 

This project explores a **vision-based approach to guitar chord recognition** using deep learning, specifically through analyzing **finger placements on a guitar fretboard** captured via a webcam. The system is designed to **assist musicians by providing real-time feedback**, making it a powerful **AI-assisted music learning tool**.

---

## 🧠 Overview

Rather than analyzing audio, this system uses **live video input** to recognize chords, bypassing the issues of noisy environments or poor audio quality.

---

## 🔍 Key Features

- 📷 **Visual Finger Recognition** on the fretboard via webcam
- ⚙️ **EfficientNetV2-S CNN Architecture** for fast and accurate classification
- 🎯 **Real-time prediction** with high accuracy across 7 chord classes
- 📈 Robust performance under varying lighting and angles

---

## 🧪 Methodology

The system follows a two-stage pipeline:

### 1. Fretboard Detection & Extraction
- Isolates the fretboard region
- Extracts key features like strings, frets, and finger positions

### 2. Chord Classification
- Employs **EfficientNetV2-S** CNN model
- Outperforms alternatives (ResNet-RS, DeiT-ViT) with:
  - 🏃‍♂️ **Up to 11× faster training**
  - 🧠 **6.8× better parameter efficiency**

---

## 📊 Dataset & Training

- Dataset from **Roboflow**, with **2,684 images** across 7 major chords:
  - `A`, `B`, `C`, `D`, `E`, `F`, `G`
- Trained over **25 epochs** on a local GPU (~2 hours)
- Evaluated on **3,068 additional test images**

---

## ✅ Performance

| Metric       | Value  |
|--------------|--------|
| Accuracy     | 0.99   |
| Precision    | 1.00   |
| Recall       | 0.99   |
| F1 Score     | 0.99   |
| AUC-ROC      | 0.996  |

🔄 **Confusion Matrix Observations**:
- Minor misclassifications like:
  - E predicted as A (1%)
  - F confused with E (3%)

---

## 🆚 Comparison with Existing Works

| Feature             | This Work | GuitarGuru |
|---------------------|-----------|------------|
| Accuracy            | 0.99      | 0.97       |
| Precision           | 1.00      | ~0.29      |
| Chord Classes       | 7         | 4          |
| Real-time Feedback  | ✅        | ❌         |

---

## 🚀 Significance

- Encourages **interactive, real-time music learning**
- Helps players **correct finger placements** while playing
- Improves engagement and learning for both beginners and professionals

---

## 🔭 Future Work

- Add support for **barre, sharp, and complex chords**
- Optimize system for **real-time performance** on low-end devices
- Build a **GUI for visual feedback** and ease of use

---

## 📁 Project Structure
📂 Guitar-Chord-Recognition-Using-CNN

├── 📁 Codes/ # contains the code files

├── 📁 images1/ # Contains images used by UI

├── 📄 guitar_chord_recognition_final.pth # contains trained weights

└── 📄 README.md # Project overview (this file)




