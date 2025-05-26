# SKIN-DISEASE-DETECTION-USING-DEEP-LEARNING-WITH-TREATMENT-RECOMMENDATIONS-

## Skin Disease Detection and Classification using Deep Learning

This project is a deep learning-based system that detects and classifies multiple skin diseases from medical images and provides tailored treatment recommendations. Designed as a major academic project, it combines CNN-based image classification with a user-friendly GUI for real-time diagnosis.

---

## Project Overview

- Utilizes a **Convolutional Neural Network (CNN)** with **ResNet50** backbone trained on a dataset of 23 skin diseases.
- Achieved **90% classification accuracy** using transfer learning and image augmentation techniques.
- Offers **real-time predictions** through a **Tkinter-based GUI** which accepts both uploaded images and webcam input.
- Automatically provides **medical recommendations** (medications and precautions) based on the classified condition.

---

##  Key Features

- **23-class classifier** for diseases such as Acne, Melanoma, Eczema, Fungal Infections, and more.
- Integrated **treatment mapping system** with disease-specific prescriptions and care guidelines.
- **Grad-CAM** integration (optional) for model interpretability.
- GUI app built with **Tkinter** for accessibility and offline use.

---

##  Project Structure
├── app/ # GUI application with live predictions
├── models/ # Trained Keras .h5 model
├── src/ # Training scripts and treatment logic
├── data/ # Raw dataset directory
├── outputs/ # Visualizations, confusion matrices, metrics
├── README.md # Project documentation
└── requirements.txt # Python dependencies

---

## 🛠 Tech Stack

- Python 3.10
- TensorFlow / Keras
- Tkinter (GUI)
- OpenCV, PIL, NumPy
- Scikit-learn (evaluation)

