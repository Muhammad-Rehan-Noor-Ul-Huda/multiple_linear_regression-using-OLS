# 🚀 California Housing Price Prediction from Scratch

A custom implementation of **Multiple Linear Regression** built completely from scratch using Python and NumPy. This project bypasses high-level machine learning libraries like `scikit-learn` to showcase a deep, foundational understanding of the underlying mathematics and optimization algorithms behind linear models.

---

## 📊 Performance Summary
* **Dataset:** California Housing Data
* **Evaluation Metric:** Coefficient of Determination ($R^2$)
* **Model Baseline Performance:** **0.6454**

---

## 🛠️ Project Workflow & Technical Dedication

### 1. Mathematical Derivation
Instead of calling a pre-built solver, the optimal parameters were derived analytically using the **Ordinary Least Squares (OLS) Normal Equation**. The objective was to minimize the Residual Sum of Squares (RSS), leading to the closed-form solution:

$$\mathbf{b} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

### 2. Rigorous Data Cleaning & Preprocessing
* Handled missing attributes and isolated independent feature matrices from target matrices.
* Implemented custom bias-handling by programmatically injecting an intercept column ($X_0 = 1$) into the feature matrix to support proper matrix multiplication layout.

### 3. Model Architecture & Implementation
* Engineered a clean, custom object-oriented `LinearRegression` class with vectorized matrix multiplication methods.
* Calculated exact matrix inversions and transpositions utilizing highly optimized NumPy arrays.

---

## 📈 Visualizing Results

The model's predictions were evaluated by mapping actual median house values ($y$) directly against predicted outputs ($y_{predict}$). 

```python
# Implementation snippet for model verification
import matplotlib.pyplot as plt

plt.scatter(y, y_predict, alpha=0.5, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel('Actual Values (y)')
plt.ylabel('Predicted Values (y_predict)')
plt.title('Actual vs. Predicted Values')
plt.show()
