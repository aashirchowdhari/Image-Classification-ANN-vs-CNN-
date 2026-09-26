<a name="top"></a>
<div align="center">

# 🖼️ Image Classification: ANN vs CNN

### Do convolutions actually matter? CIFAR-10 says yes.

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
![CIFAR-10](https://img.shields.io/badge/dataset-CIFAR--10-8F82E8?style=for-the-badge)

A head-to-head comparison of a fully-connected **Artificial Neural Network** and a **Convolutional Neural Network** on CIFAR-10 — same data, same output layer, very different results.

[Dataset](#-dataset) · [Models](#%EF%B8%8F-models-implemented) · [Stack](#%EF%B8%8F-tech-stack) · [Quickstart](#%EF%B8%8F-installation) · [Results](#-results)

</div>

---

## 🎯 Why this comparison

It's easy to say "CNNs are better for images" — this project puts a number on *how much* better, using the exact same dataset and training setup for both architectures. The ANN treats an image as a flat vector of pixels; the CNN treats it as a spatial grid. The accuracy gap is the cost of throwing away that spatial structure.

```mermaid
flowchart LR
    A["CIFAR-10<br/>60,000 images, 32×32×3"] --> B["Normalize pixel values"]
    B --> C{Architecture}
    C -->|"Flatten"| D["ANN<br/>Dense layers only"]
    C -->|"Keep spatial grid"| E["CNN<br/>Conv2D + MaxPooling"]
    D --> F["Softmax: 10 classes"]
    E --> F
    F --> G["Accuracy · Loss ·<br/>Confusion Matrix"]

    style E fill:#FF6F00,stroke:#0A0A0F,color:#fff
    style D fill:#8F82E8,stroke:#0A0A0F,color:#fff
    style G fill:#16C060,stroke:#0A0A0F,color:#fff
```

---

## 📊 Dataset

- **CIFAR-10** — 60,000 color images, 32×32 pixels each
- **10 categories**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Preprocessed with pixel normalization before training

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🧹 **Preprocessing** | Pixel normalization for stable training |
| 🔷 **ANN model** | Fully-connected baseline |
| 🔶 **CNN model** | Convolution + pooling for spatial feature extraction |
| 📈 **Evaluation** | Accuracy, loss curves, and a full classification report |
| 🖼️ **Visualizations** | Sample images with labels, confusion matrix, predicted vs. actual classes |

---

## 🛠️ Models Implemented

### 🔹 Artificial Neural Network (ANN)
- **Input**: flattened 32×32×3 images → 3,072-length vector
- Dense layers with ReLU and Sigmoid activations
- **Output**: 10 neurons, Softmax

### 🔹 Convolutional Neural Network (CNN)
- Conv2D layers with ReLU activation
- MaxPooling2D layers for downsampling
- Dense layers for final classification
- **Output**: 10 neurons, Softmax

---

## 🧱 Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow%20%2F%20Keras-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=plotly&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)

---

## 🗂️ Project Structure

```
cifar10-ann-cnn/
├── ANN_Project.ipynb        # Main notebook — ANN + CNN, side by side
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/cifar10-ann-cnn.git
cd cifar10-ann-cnn
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the notebook:

```bash
jupyter notebook ANN_Project.ipynb
```

Steps performed:

1. Load the CIFAR-10 dataset
2. Normalize image data
3. Train the ANN and CNN models
4. Evaluate using accuracy and a classification report

---

## 📊 Results

| Model | Test Accuracy |
|---|---|
| ANN | ~44% |
| CNN | ~57% |

- The ANN struggles with raw pixel data — flattening the image throws away spatial relationships between neighboring pixels.
- The CNN outperforms it by a wide margin, since convolutional filters can actually learn edges, textures, and shapes.

---

## 🔮 Future Enhancements

- [ ] Deeper CNN with dropout & batch normalization
- [ ] Try modern architectures (ResNet, VGG, etc.)
- [ ] Deploy as a web app for real-time classification

---

## 🤝 Contributing

Contributions are welcome — fork the repo, add features, and submit a PR.

<div align="center">

<a href="#top">⬆️ Back to top</a>

</div>
