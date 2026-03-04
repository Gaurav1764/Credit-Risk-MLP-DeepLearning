# Neural Network–Based Credit Risk Prediction using Multi-Layer Perceptron (MLP)

## Deep Learning Architectures and Techniques – Laboratory Assignment

This project implements a **feedforward neural network (Multi-Layer Perceptron)** to perform **binary classification for credit risk prediction** using structured financial data.

The objective of the experiment is to understand the working principles of artificial neurons and feedforward neural networks and apply them to a real-world classification problem.

The model predicts whether a loan applicant represents a **good credit risk** or **bad credit risk** based on financial attributes.

---

# Project Overview

Credit risk prediction is an important task in the financial industry. Banks and financial institutions need to evaluate whether a customer is likely to repay a loan or default.

Traditional statistical methods are often limited when dealing with complex relationships in financial data. Neural networks provide a powerful alternative because they can learn nonlinear patterns and relationships between input features.

In this project:

* A **Multi-Layer Perceptron (MLP)** model is implemented using TensorFlow/Keras.
* The model is trained on the **German Credit Dataset**.
* Data preprocessing techniques are applied before training.
* Model performance is evaluated using classification metrics and visualization.

---

# Dataset

The **German Credit Dataset** is used in this experiment.

This dataset contains financial information about loan applicants and their credit risk classification.

Dataset characteristics:

| Feature         | Description              |
| --------------- | ------------------------ |
| Dataset Type    | Tabular Financial Data   |
| Problem Type    | Binary Classification    |
| Target Variable | Credit Risk              |
| Classes         | Good Credit / Bad Credit |

Examples of features include:

* Credit history
* Loan duration
* Credit amount
* Employment status
* Housing type
* Age of applicant

Target variable encoding:

* **1 → Good Credit Risk**
* **0 → Bad Credit Risk**

---

# Project Workflow

The project follows a standard machine learning pipeline:

1. Dataset loading
2. Data preprocessing
3. Handling missing values
4. Encoding categorical variables
5. Feature normalization
6. Train–test split
7. Neural network model design
8. Model training
9. Performance evaluation
10. Visualization of results

---

# Technologies Used

The project is implemented using the following tools and libraries:

* Python
* TensorFlow / Keras
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook

---

# Data Preprocessing

Several preprocessing steps were applied before training the neural network:

### Missing Value Handling

Missing values were handled using forward-fill techniques to ensure dataset consistency.

### Categorical Encoding

Categorical features were converted into numeric form using **Label Encoding**.

### Feature Scaling

Feature scaling was applied using **StandardScaler** to normalize numerical values.

Normalization helps neural networks converge faster during training.

### Train-Test Split

The dataset was split into:

* **80% training data**
* **20% testing data**

This allows evaluation of the model on unseen data.

---

# MLP Model Architecture

The neural network used in this experiment is a **Multi-Layer Perceptron (MLP)**.

Architecture:

Input Layer
→ Dense Layer (32 neurons, ReLU activation)
→ Dense Layer (16 neurons, ReLU activation)
→ Output Layer (1 neuron, Sigmoid activation)

Activation Functions:

* **ReLU** for hidden layers
* **Sigmoid** for output layer

Loss Function:

* Binary Cross-Entropy

Optimizer:

* Adam Optimizer

---

# Training Process

The model was trained using the training dataset for multiple epochs.

During training:

* Training accuracy
* Validation accuracy
* Training loss
* Validation loss

were recorded to analyze model performance.

---

# Model Evaluation

The trained model was evaluated using several classification metrics.

Evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

These metrics provide a comprehensive understanding of the model’s classification performance.

---

# Results

Typical results obtained from the model:

| Metric    | Performance |
| --------- | ----------- |
| Accuracy  | ~80–85%     |
| Precision | ~80%        |
| Recall    | ~78%        |
| F1 Score  | ~79%        |

The model successfully learns patterns in financial data and performs accurate credit risk classification.

---

# Project Structure

```
Credit-Risk-MLP-DeepLearning
│
├── Assignment_1.ipynb
├── Credit_Risk_MLP.py
│
├── dataset
│   └── german_credit.csv
│
├── plots
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   └── confusion_matrix.png
│
├── report
│   └── Credit_Risk_MLP_Report.pdf
│
└── README.md
```

---

# How to Run the Project

## Using Jupyter Notebook

1. Open Jupyter Notebook or Google Colab
2. Open the notebook

```
Credit_Risk_MLP.ipynb
```

3. Run all cells

The notebook will:

* Load the dataset
* Preprocess the data
* Train the MLP model
* Generate training plots
* Evaluate model performance

---

# Learning Outcomes

This experiment demonstrates the following deep learning concepts:

* Artificial neurons
* Feedforward neural networks
* Activation functions
* Loss functions
* Optimization techniques
* Binary classification using neural networks

---

# Author

**Gaurav Kumar**

MCA – Deep Learning Architectures and Techniques
School of Engineering and Technology (SOET)

---

# License

This project is developed for academic purposes as part of the Deep Learning laboratory coursework.

---
