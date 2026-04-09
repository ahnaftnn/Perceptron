# Perceptron Implementation

This repository provides a scratch implementation of a single-layer Perceptron using Python and NumPy. The entire implementation and its demonstration are contained within a single Jupyter Notebook (`Perceptron.ipynb`).

The Perceptron is a fundamental algorithm for supervised learning of binary classifiers. This project demonstrates its ability to learn linearly separable patterns (like the AND and OR logic gates) and its limitations when faced with non-linearly separable data (the XOR logic gate).

## How It Works

The core logic is encapsulated in the `perceptron` class.

### `perceptron` Class

-   **`__init__(self, eta, epochs)`**: Initializes the model with a specific learning rate (`eta`) and a fixed number of training iterations (`epochs`). The weights are initialized with small random values.
-   **`activationFunction(self, inputs, weights)`**: A simple step function that returns `1` if the dot product of inputs and weights is greater than zero, and `0` otherwise.
-   **`fit(self, X, y)`**: Trains the perceptron. In each epoch, it calculates the predicted output (`y_hat`), determines the error (`y - y_hat`), and updates the weights based on the error and the learning rate. A bias term is automatically added to the input features.
-   **`predict(self, X)`**: Uses the trained weights to make predictions on new, unseen data.

```python
import numpy as np

class perceptron:
  def __init__(self, eta, epochs):
    self.weights = np.random.randn(3) * 1e-4
    self.eta = eta
    self.epochs = epochs

  def activationFunction(self, inputs, weights):
    z = np.dot(inputs, weights)
    return np.where(z > 0 , 1, 0)

  def fit(self, X, y):
    self.X = X
    self.y = y

    # Add bias term to the input data
    X_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))]

    for epoch in range(self.epochs):
      y_hat = self.activationFunction(X_with_bias, self.weights)
      error = self.y - y_hat
      self.weights = self.weights + self.eta * np.dot(X_with_bias.T, error)

  def predict(self, X):
    X_with_bias = np.c_[X, -np.ones((len(self.X), 1))]
    return self.activationFunction(X_with_bias, self.weights)
```

## Demonstration with Logic Gates

The notebook demonstrates the Perceptron's behavior by training it to replicate the `AND`, `OR`, and `XOR` logic gates.

### Usage Example: Training on the `AND` Gate

1.  **Prepare the Data**: The data for the `AND` gate is created using a Pandas DataFrame.

    ```python
    import pandas as pd

    AND = {
        'x1' : [0,0,1,1],
        'x2' : [0,1,0,1],
        'y'  : [0,0,0,1]
    }
    AND = pd.DataFrame(AND)
    
    X = AND.drop("y", axis=1)
    y = AND['y']
    ```

2.  **Initialize and Train the Model**: An instance of the `perceptron` class is created and trained on the `AND` data.

    ```python
    # Initialize the model with a learning rate of 0.5 and 10 epochs
    model = perceptron(eta = 0.5, epochs = 10)
    
    # Train the model
    model.fit(X, y)
    ```
    The training process shows the weights converging to a state where the model correctly predicts the `AND` logic.

3.  **Make Predictions**:

    ```python
    model.predict(X)
    # Expected Output: array([0, 0, 0, 1])
    ```

### Results

#### AND Gate (Linearly Separable)

The Perceptron successfully learns the `AND` gate. The decision boundary found by the model correctly separates the data points into their respective classes.

```python
# Plotting the data and the decision boundary
AND.plot(kind="scatter", x="x1", y="x2", c="y", s=50, cmap="winter")
x = np.linspace(0, 1.4)
y = 1.5 - 1*np.linspace(0, 1.4) # Example decision boundary after training
plt.plot(x, y, "r--")
```

#### XOR Gate (Non-Linearly Separable)

The notebook also demonstrates that the single-layer Perceptron fails to converge on the `XOR` problem. This is because the `XOR` data is not linearly separable, and a single straight line cannot divide the four data points into their correct classes. The training logs for the `XOR` gate show the weights oscillating without ever reaching a state where the error is zero.

## Requirements

To run the `Perceptron.ipynb` notebook, you need the following libraries:

*   `numpy`
*   `pandas`
*   `matplotlib`
*   `joblib`
*   `jupyter notebook` or `jupyter lab`

## Setup and Installation

1.  Clone the repository:
    ```sh
    git clone https://github.com/ahnaftnn/Perceptron.git
    ```

2.  Navigate to the project directory:
    ```sh
    cd Perceptron
    ```

3.  Install the required dependencies:
    ```sh
    pip install numpy pandas matplotlib joblib jupyter
    ```

4.  Run the Jupyter Notebook:
    ```sh
    jupyter-notebook Perceptron.ipynb
