# 🧠 Neural Network from Scratch: Cats vs Dogs

This project showcases a fully custom **feedforward neural network** built entirely with **NumPy** to classify images of cats and dogs. It demonstrates every step manually — from image preprocessing to forward and backward propagation, gradient descent, and prediction — without using any ML libraries like TensorFlow or PyTorch.

---

## 📌 Project Overview

- Preprocess image datasets using Pillow and NumPy
- Construct a two-layer neural network with:
  - **Tanh** activation in the hidden layer
  - **Sigmoid** activation in the output layer (binary classification)
- Implement **manual forward and backward propagation**
- Train using **cross-entropy loss** and **gradient descent**
- Predict on new images and visualize model learning with cost curves

---

## 🛠️ Features

- ❌ No external ML libraries — just **NumPy**
- 🔧 Configurable hidden layer size (`n_h`)
- 🖼️ Image normalization, reshaping, and flattening
- 📉 Visual cost tracking across iterations
- 🧪 Predict custom user-provided images

---

## 🚀 Getting Started

### ✅ Prerequisites

Make sure Python 3.x is installed. Then install the dependencies:

```bash
pip install -r requirements.txt
```

project-root/
├── Neural_Network_From_Scratch.ipynb   # Main Jupyter notebook
├── requirements.txt                    # Dependencies list
├── README.md                           # Project README
├── info.txt                            # Description for web/project pages
└── dataset/
    ├── Cat/
    └── Dog/

```python
from PIL import Image
import numpy as np

img = Image.open('./dataset/manual_set/4.jpg').resize((64, 64)).convert('RGB')
img_array = np.array(img).reshape(-1, 1) / 255.0

A2, _ = L_model_forward(img_array, parameters, 'tanh')
prediction = (A2 > 0.5)

print("Predicted class:", "Cat" if prediction else "Dog")
```
