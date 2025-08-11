Here's a concise **README** explaining the **architecture choices** and **convergence analysis** for your project:

---

# 📘 README

## 🧠 Project: Fully Connected Neural Network for Regression (NumPy-only)

This project implements a fully connected feedforward neural network from scratch using **NumPy**. The model is trained using **stochastic gradient descent (SGD)** on a **noisy cubic function**, demonstrating its ability to learn non-linear patterns.

---

## 🏗️ Architecture Choices

### Model Structure

* **Input layer**: 1 neuron (univariate input)
* **Hidden layer**: 64 neurons

  * Chosen as a tradeoff between capacity and overfitting
  * Non-linear transformation via **ReLU** activation
* **Output layer**: 1 neuron (for regression output)

  * No activation on the output layer (identity function)

### Activation Functions

* **ReLU**: Used in the hidden layer for better performance on non-linear functions. It helps mitigate vanishing gradients during training.
* **Optional Sigmoid**: Code supports switching to sigmoid if needed.

### Loss Function

* **Mean Squared Error (MSE)**: Commonly used for regression tasks. Measures the average squared difference between predictions and actual values.

### Optimizer

* **Stochastic Gradient Descent (SGD)**:

  * Adjustable **learning rate** and **mini-batch size**
  * Enables efficient training with noisy gradient updates
  * Random shuffling and batching improve generalization

---

## 📈 Convergence Analysis

### Training Behavior

* The model is trained for **1000 epochs**.
* **Loss curve** shows a steady decrease in MSE over time, indicating successful convergence.
* Training is monitored every 20 epochs for visibility.

### Final Results

* The model effectively fits the **noisy cubic function**, capturing the non-linear pattern.
* The prediction curve closely aligns with ground truth data, despite added noise.

### Challenges Handled

* Noise in the data is smoothed out by the model's generalization ability.
* ReLU activation helped avoid saturation issues that could have occurred with sigmoid.

---

## 📊 Visualizations

* **Scatter plot** of synthetic data
* **Loss curve** showing training dynamics
* **Prediction vs Ground Truth** for visual inspection of regression quality

---

## ✅ How to Run

1. Ensure Python 3.x with `numpy` and `matplotlib` is installed.
2. Run the provided `train.ipynb` notebook to:

   * Generate data
   * Train the model
   * Visualize training and results

---
