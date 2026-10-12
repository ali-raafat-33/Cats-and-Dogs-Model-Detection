# 🐱🐶 Cats vs Dogs Image Classification Using CNN

A beginner-friendly **Deep Learning project** that classifies an image as either a **cat** or a **dog** using a Convolutional Neural Network (CNN) built from scratch with **PyTorch**.

> **Learning project:** This repository is intended to help students understand the complete image classification workflow, from preparing images to training, evaluating, and testing a CNN on their own photos.

## 🎯 Learning Objectives

By working through this notebook, you will learn how to:

- Load an image dataset using `torchvision.datasets.ImageFolder`.
- Resize images and convert them into PyTorch tensors.
- Split a dataset into training and testing sets.
- Build a CNN with convolution, ReLU, max-pooling, and fully connected layers.
- Train a binary classifier using `BCEWithLogitsLoss` and the Adam optimizer.
- Evaluate classification accuracy on held-out images.
- Predict whether a user-selected image contains a cat or a dog.

## 🛠️ Technologies Used

- **Python** — programming language
- **PyTorch** — model building and training
- **Torchvision** — image transforms and dataset loading
- **NumPy** — numerical utilities and reproducibility
- **Pillow (PIL)** — opening images for prediction
- **Matplotlib** — displaying images and predictions
- **Tkinter** — choosing an image from your computer

## 📁 Dataset

The notebook expects an image dataset arranged into class folders, for example:

```text
PetImages/
├── Cat/
│   ├── cat1.jpg
│   └── ...
└── Dog/
    ├── dog1.jpg
    └── ...
```

Images are loaded with `ImageFolder`, which assigns numeric labels based on the alphabetical order of folder names (typically **Cat = 0** and **Dog = 1**).

**Dataset configuration used in the notebook:**

| Setting | Value |
|---|---|
| Image size | 64 × 64 |
| Input channels | 3 (RGB) |
| Batch size | 32 |
| Training split | 90% |
| Testing split | 10% |
| Random seed | 42 |

> **Important:** The dataset is not included in the notebook. Update the dataset path in the `ImageFolder` cell to match its location on your computer. If your dataset contains corrupted images, remove or repair them before training.

## 🧠 CNN Architecture

The model is named `SimpleCNN` and contains three convolutional blocks followed by a fully connected classifier.

```text
Input Image (3 × 64 × 64)
        ↓
Conv2d (3 → 16, 3×3) + ReLU + MaxPool2d
        ↓
     16 × 32 × 32
        ↓
Conv2d (16 → 32, 3×3) + ReLU + MaxPool2d
        ↓
     32 × 16 × 16
        ↓
Conv2d (32 → 64, 3×3) + ReLU + MaxPool2d
        ↓
      64 × 8 × 8
        ↓
      Flatten
        ↓
 Linear (4096 → 128) + ReLU
        ↓
    Linear (128 → 1)
        ↓
     Output Logit
        ↓
 Sigmoid (during prediction)
        ↓
 Cat (0) or Dog (1)
```

**Why a single output neuron?** This is a binary classification problem. The network produces one *logit*, which is converted into a probability using the sigmoid function when making predictions.

## ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Loss function | `nn.BCEWithLogitsLoss()` |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Epochs | 4 |
| Device | CUDA if available; otherwise CPU |
| Decision threshold | 0.5 |

`BCEWithLogitsLoss` combines the sigmoid operation and binary cross-entropy loss in a numerically stable way. **Do not apply sigmoid before passing outputs to this loss function.**

During training, the notebook calculates and prints **training accuracy** after each epoch. It then evaluates **test accuracy** using `model.eval()` and `torch.no_grad()`.

> Results depend on the dataset and training environment. This README does not claim a specific accuracy because no verified final metric is provided here.

## 🚀 Getting Started

### 1. Install dependencies

Make sure Python is installed, then run:

```bash
pip install torch torchvision numpy pillow matplotlib jupyter
```

Tkinter is included with many Python installations but may need to be installed separately on some systems.

### 2. Prepare the dataset

Download or prepare cat and dog images in the folder structure shown above. Update this notebook cell with your actual path:

```python
dataset = datasets.ImageFolder(
    root="path/to/PetImages",
    transform=transform
)
```

### 3. Open the notebook

```bash
jupyter notebook
```

Open `cats_and_dogs(1).ipynb` and run the cells in order.

### 4. Train and evaluate

The notebook will:

1. Set the random seed.
2. Resize images to **64 × 64** and convert them to tensors.
3. Split the dataset into **90% training** and **10% testing**.
4. Create data loaders and initialize the CNN.
5. Train the model for **4 epochs**.
6. Calculate accuracy on the test set.

### 5. Test your own image

The final notebook cell opens a file picker. Choose a `.jpg`, `.jpeg`, `.png`, or `.webp` image containing a cat or dog.

The notebook will display:

- The selected image
- Predicted class: **Cat** or **Dog**
- Model confidence, formatted as a percentage

**Note:** The file picker uses Tkinter and requires a local desktop environment with a graphical interface. It may not work in cloud-hosted notebooks.

## 🔍 How Prediction Works

```python
with torch.no_grad():
    output = model(image_tensor)
    dog_probability = torch.sigmoid(output).item()

if dog_probability >= 0.5:
    prediction = "Dog"
    confidence = dog_probability
else:
    prediction = "Cat"
    confidence = 1 - dog_probability
```

- A probability **≥ 0.5** is classified as **Dog**.
- A probability **< 0.5** is classified as **Cat**.
- The displayed confidence is the model's probability for the predicted class; it is **not a guarantee** that the prediction is correct.

## 📚 Key Concepts Practiced

`ImageFolder` · `DataLoader` · `random_split` · `Conv2d` · `ReLU` · `MaxPool2d` · `Flatten` · `Linear` · `BCEWithLogitsLoss` · `Adam` · `Sigmoid` · `model.train()` · `model.eval()` · `torch.no_grad()`

## 💡 Possible Improvements

- Add a separate validation set.
- Use data augmentation to improve generalization.
- Plot training loss and accuracy across epochs.
- Calculate a confusion matrix, precision, recall, and F1-score.
- Save and reload trained model weights with `torch.save()` and `torch.load()`.
- Compare the simple CNN with a pretrained model using transfer learning.

## 📌 Project Purpose

This project was developed **for educational purposes** to demonstrate how CNN-based binary image classification works in PyTorch. It is a hands-on starting point for students learning **Deep Learning** and **Computer Vision**, not a production-ready classifier.
