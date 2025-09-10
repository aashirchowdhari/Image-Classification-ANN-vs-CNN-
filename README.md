
# 🖼️ Image Classification with ANN vs CNN

This project applies **image classification on the CIFAR-10 dataset** using two different approaches:  
1. **Artificial Neural Network (ANN)**  
2. **Convolutional Neural Network (CNN)**  

The objective is to compare the performance of fully-connected vs convolutional models on small image datasets.

---

## 📊 Dataset
- **CIFAR-10** dataset  
- 60,000 color images (32x32 pixels each)  
- 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck  

---

## 🚀 Features
- Data preprocessing (normalization)  
- ANN model with fully connected layers  
- CNN model with convolution + pooling layers  
- Model evaluation with accuracy, loss, and classification report  
- Visualizations:
  - Sample images with labels  
  - Confusion matrix  
  - Predicted vs actual classes  

---

## 🛠️ Models Implemented
### 🔹 Artificial Neural Network (ANN)
- Input: Flattened 32×32×3 images  
- Dense layers with ReLU and Sigmoid activations  
- Output: 10 neurons (Softmax for classification)  

### 🔹 Convolutional Neural Network (CNN)
- Conv2D layers with ReLU activation  
- MaxPooling2D layers for downsampling  
- Dense layers for classification  
- Output: 10 neurons (Softmax for classification)  

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



