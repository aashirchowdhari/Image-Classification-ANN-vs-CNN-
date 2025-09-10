



```markdown
# 🖼️  Image Classification (ANN vs CNN)

This project implements image classification on the CIFAR-10 dataset using both an Artificial Neural Network (ANN) and a Convolutional Neural Network (CNN).  
The goal is to compare performance of fully-connected vs convolutional approaches on small image datasets.

---

## 🚀 Features
- 📊 Dataset: **CIFAR-10** (60,000 32x32 color images, 10 classes)
- 🧮 Models implemented:
  - **ANN**: Flatten + Dense layers (ReLU, Sigmoid)
  - **CNN**: Conv2D + MaxPooling + Dense layers
- 📈 Training & evaluation using accuracy, loss, and classification reports
- 📊 Visualizations: sample images, confusion matrix, predicted labels
- ⚡ Comparison of ANN vs CNN performance

---

## 📂 Project Structure
```

├── ANN\_Project.ipynb        # Main Jupyter Notebook (ANN + CNN)
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies

````

---

## ⚙️ Installation
Clone the repository:
```bash
git clone https://github.com/your-username/cifar10-ann-cnn.git
cd cifar10-ann-cnn
````

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

1. Load CIFAR-10 dataset
2. Normalize image data
3. Train ANN and CNN models
4. Evaluate using accuracy & classification report

---

## 🛠️ Tech Stack

* **Python** 🐍
* **TensorFlow / Keras** — ANN and CNN models
* **NumPy, Pandas** — data handling
* **Matplotlib** — visualization
* **Scikit-learn** — evaluation metrics

---

## 📊 Results

| Model | Test Accuracy |
| ----- | ------------- |
| ANN   | \~44%         |
| CNN   | \~57%         |

* ANN struggles with raw image data due to lack of spatial feature extraction.
* CNN significantly outperforms ANN by leveraging convolutional filters.

---

## 🔮 Future Enhancements

* ✅ Add deeper CNN with dropout & batch normalization
* ✅ Try modern architectures (ResNet, VGG, etc.)
* ✅ Deploy as a web app for real-time classification

---

## 🤝 Contributing

Contributions are welcome! Fork the repo, add features, and submit a PR.

---

## 📜 License

Distributed under the **MIT License**.

```

---
```
