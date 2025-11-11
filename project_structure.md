## 🧩 Project: “NumPy Deep Learning — Build and Train a Neural Network from Scratch”

### 🎯 **Goal**

Implement a **complete supervised learning pipeline** from raw data to model evaluation — including **forward propagation, backpropagation, optimization, and performance measurement** — all using only `NumPy`.

You’ll build a configurable **Deep Neural Network (DNN)** that can handle both **regression** and **classification** tasks.

---

## 1️⃣ **Project Overview**

You will:

1. Build a flexible neural-network class that supports any number of layers and neurons.
2. Implement multiple activation functions (ReLU, sigmoid, softmax).
3. Implement multiple loss functions (MSE, binary cross-entropy, multiclass cross-entropy).
4. Train the model using gradient descent (GD, SGD, Momentum, Adam).
5. Visualize the learning process and analyze generalization performance.

---

## 2️⃣ **Dataset Options**

Choose one of the following:

- 🟦 **Regression:** Predict a continuous function such as `y = sin(x)` or a noisy polynomial.
- 🟥 **Classification:**

  - Binary: 2D “moon” dataset (`sklearn.datasets.make_moons`)
  - Multiclass: 3-class spiral or circle dataset

> You can use scikit-learn only to **generate data** (no ML functions).

---

## 3️⃣ **Implementation Plan**

### **Step 1: Data Preparation**

- Generate and normalize your dataset.
- Split into **train/test** (e.g., 80/20).
- Visualize input data (2D scatter or line plot).

---

### **Step 2: Model Architecture (Ch. 2–4)**

Implement a `NeuralNetwork` class:

```python
class NeuralNetwork:
    def __init__(self, layer_dims, activations, seed=42):
        self.params = self._initialize_params(layer_dims)
        self.activations = activations
```

- `layer_dims` = list like `[2, 16, 8, 1]`
- `_initialize_params()` uses **He** or **Xavier** initialization.

#### Activation functions:

```python
def relu(z): return np.maximum(0, z)
def relu_derivative(z): return (z > 0).astype(float)
def sigmoid(z): return 1 / (1 + np.exp(-z))
def softmax(z): exp_z = np.exp(z - np.max(z, axis=1, keepdims=True)); return exp_z / np.sum(exp_z, axis=1, keepdims=True)
```

---

### **Step 3: Forward Propagation (Ch. 3–4)**

Compute all layer outputs:

```python
def forward_pass(self, X):
    caches = []
    A = X
    for l in range(1, L):
        Z = A @ W[l].T + b[l]
        A = relu(Z) if act[l] == 'relu' else sigmoid(Z)
        caches.append((A, Z))
    # Output layer (softmax or linear)
```

Store intermediate `Z`, `A` values for backpropagation.

---

### **Step 4: Loss Functions (Ch. 5)**

Implement:

- **MSE (Regression):**
  [
  L = \frac{1}{N}\sum_i (y_i - \hat{y}_i)^2
  ]
- **Binary Cross-Entropy:**
  [
  L = -\frac{1}{N}\sum_i [y_i\log(\hat{y}_i)+(1-y_i)\log(1-\hat{y}_i)]
  ]
- **Multiclass Cross-Entropy:**
  [
  L = -\frac{1}{N}\sum_i \sum_c y_{ic}\log(\hat{y}_{ic})
  ]

---

### **Step 5: Backpropagation (Ch. 7)**

Compute gradients for all parameters using the **chain rule**.

Example for layer `l`:

```python
dZ = dA * relu_derivative(Z)
dW = (1/m) * dZ.T @ A_prev
db = (1/m) * np.sum(dZ, axis=0, keepdims=True)
dA_prev = dZ @ W
```

Store gradients, then update parameters.

---

### **Step 6: Optimization Algorithms (Ch. 6)**

Implement several optimizers:

1. **Basic Gradient Descent**
   ( W := W - \eta , dW )
2. **Momentum**
   ( v = \beta v + (1-\beta)dW )
3. **Adam**
   ( m_t, v_t ) updates with bias correction and adaptive learning rate.

Allow the user to choose optimizer type at initialization.

---

### **Step 7: Training Loop**

```python
for epoch in range(num_epochs):
    y_hat, caches = model.forward_pass(X_train)
    loss = compute_loss(y_hat, y_train)
    grads = model.backward_pass(y_hat, y_train, caches)
    model.update_params(grads, optimizer)
```

Print or plot training loss every few epochs.

---

### **Step 8: Performance Measurement (Ch. 8)**

- Compute **train and test losses**.
- For classification, compute **accuracy**:

  ```python
  preds = np.argmax(y_hat, axis=1)
  acc = np.mean(preds == y_true)
  ```

- Plot:

  - Loss vs. epochs.
  - Train vs. test error.
  - Decision boundaries for 2D data.

#### Optional:

Perform **hyperparameter search** (learning rate, hidden units) and plot results.

---

## 4️⃣ **Deliverables**

| Component           | Description                                          |
| ------------------- | ---------------------------------------------------- |
| `neural_network.py` | Implementation of your class and training functions. |
| `activations.py`    | ReLU, Sigmoid, Softmax implementations.              |
| `losses.py`         | MSE, BCE, CCE losses.                                |
| `optimizers.py`     | GD, Momentum, Adam implementations.                  |
| `train.ipynb`       | Notebook demonstrating training, plots, analysis.    |

---

## 5️⃣ **Extensions**

Once you complete the base version:

- 🧮 Add **batch normalization**.
- 🎛 Add **learning-rate scheduling**.
- 🧠 Implement **dropout** for regularization.
- 📉 Add gradient-checking to verify backprop correctness.

---

## 6️⃣ **Learning Outcomes**

By completing this, you’ll have:

- Built the **entire supervised learning pipeline from scratch**, end-to-end.
- Understood every mathematical component of Chapters 1–8.
- Created a reusable NumPy framework that mirrors modern DL libraries.

---

### 🕐 **Estimated time**

| Skill Level                   | Approx. Duration |
| ----------------------------- | ---------------- |
| Intermediate (NumPy + Python) | 15–20 hours      |
| Advanced                      | 10–12 hours      |
