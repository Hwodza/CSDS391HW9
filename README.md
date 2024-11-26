# CSDS391HW9

This project implements a simple neural network to classify Iris flower species based on petal length and petal width. The neural network is trained using gradient descent and visualized using decision boundaries and learning curves.

## Project Structure

- `main.py`: Contains the implementation of the neural network, gradient descent, and plotting functions.
- `irisdata.csv`: Dataset containing Iris flower measurements.
- `.gitignore`: Specifies files and directories to be ignored by git.
- `README.md`: This file.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib

You can install the required packages using pip:

```sh
pip install pandas numpy matplotlib
```

## Functions
**plot_iris_data(df)**
Plots the Iris data points with petal length and petal width.

**sigmoid(x)**
Applies the sigmoid function to the input x.

**neural_network_output(petal_length, petal_width, weights, bias)**
Computes the output of the neural network for given petal length, petal width, weights, and bias.

**plot_decision_boundary(df, weights, bias, title)**
Plots the decision boundary of the neural network.

**plot_3d_output(df, weights, bias)**
Creates a 3D plot of the neural network output.

**calculate_mse(data_vectors, weights, bias, pattern_classes)**
Calculates the mean squared error of the neural network predictions.

**calculate_gradients(data_vectors, weights, bias, pattern_classes)**
Calculates the gradients of the weights and bias for the neural network.

**gradient_descent(weights, bias, data_vectors, pattern_classes, learning_rate=0.5, iter=10000, epsilon=0.000000001)**
Performs gradient descent to optimize the weights and bias of the neural network.

**plot_learning_curve(mse_history, title='Learning Curve')**
Plots the learning curve of the mean squared error over iterations.