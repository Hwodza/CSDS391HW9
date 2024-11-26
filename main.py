import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


np.random.seed(0)

# Exercise 1a
def plot_iris_data(df):
    # Extract petal length, petal width, and species
    iris_species = df['species']
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    for species in iris_species.unique():
        subset = df[df['species'] == species]
        plt.scatter(subset['petal_length'], subset['petal_width'], label=species)

    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.title('Petal Length vs Petal Width by Iris Species')
    plt.legend()
    plt.show()


# Exercise 1b
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Exercise 1b
def neural_network_output(petal_length, petal_width, weights, bias):
    # Combine petal length and width into a single input array
    inputs = np.array([petal_length, petal_width])
    
    # Compute the weighted sum
    weighted_sum = np.dot(weights, inputs) + bias
    
    # Apply the sigmoid non-linearity
    output = sigmoid(weighted_sum)
    
    return output


# Exercise 1c
def plot_decision_boundary(df, weights, bias, title):
    # Create a mesh grid
    x_min, x_max = df['petal_length'].min() - 1, df['petal_length'].max() + 1
    y_min, y_max = df['petal_width'].min() - 1, df['petal_width'].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Compute the neural network output for each point in the mesh grid
    Z = np.array([neural_network_output(x, y, weights, bias) for x, y in zip(np.ravel(xx), np.ravel(yy))])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    contour = plt.contourf(xx, yy, Z, alpha=0.8, levels=[0, 0.5, 1], cmap='viridis')
    
    # # Add a color bar
    cbar = plt.colorbar(contour)
    cbar.set_label('Neural Network Output')
    
    # Plot the original data
    for species in df['species'].unique():
        subset = df[df['species'] == species]
        plt.scatter(subset['petal_length'], subset['petal_width'], label=species)
    
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.title(title)
    plt.legend()
    plt.show()


# Exercise 1d
def plot_3d_output(df, weights, bias):
    # Create a mesh grid
    x_min, x_max = df['petal_length'].min() - 1, df['petal_length'].max() + 1
    y_min, y_max = df['petal_width'].min() - 1, df['petal_width'].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Compute the neural network output for each point in the mesh grid
    Z = np.array([neural_network_output(x, y, weights, bias) for x, y in zip(np.ravel(xx), np.ravel(yy))])
    Z = Z.reshape(xx.shape)
    print(Z)
    
    # Create a 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, Z, cmap='viridis', alpha=0.8)
    
    ax.set_xlabel('Petal Length')
    ax.set_ylabel('Petal Width')
    ax.set_zlabel('Neural Network Output')
    ax.set_title('3D Plot of Neural Network Output')
    ax.legend()
    plt.show()


# Exercise 1e
def e1e(df, weights, bias):
    indexes = [0, 1, 2, 10, 11, 60, 61, 90, 91, 99]
    for idx in indexes:
        row = df.iloc[idx]
        petal_length = row['petal_length']
        petal_width = row['petal_width']
        nn_output = neural_network_output(petal_length, petal_width, weights, bias)
        print(f"Index: {idx}, Petal Length: {petal_length}, Petal Width: {petal_width}, Neural Network Output: {nn_output}")


# Exercise 2a
def calculate_mse(data_vectors, weights, bias, pattern_classes):
    # Calculate the neural network output for each data point
    outputs = np.array([neural_network_output(row['petal_length'], row['petal_width'], weights, bias) for _, row in data_vectors.iterrows()])
    
    # Calculate the mean-squared error
    mse = np.mean((outputs - pattern_classes) ** 2)
    
    return mse


# Exercise 2b
def e2b(df):
    weights = np.array([0.35, 0.35])
    bias = -2.3
    mse = calculate_mse(df, weights, bias, df['pattern_class'])
    print(f"Mean-Squared Error: {mse}")
    plot_decision_boundary(df, weights, bias, f'Decision Boundary with MSE: {mse}')
    weights = np.array([0.5, -0.5])
    bias = 0.0
    mse = calculate_mse(df, weights, bias, df['pattern_class'])
    print(f"Mean-Squared Error: {mse}")
    plot_decision_boundary(df, weights, bias, f'Decision Boundary with MSE: {mse}')


def calculate_gradients(data_vectors, weights, bias, pattern_classes):
    N = len(data_vectors)
    gradients = np.zeros_like(weights)
    bias_gradient = 0.0

    for _, row in data_vectors.iterrows():
        petal_length = row['petal_length']
        petal_width = row['petal_width']
        y_true = row['pattern_class']
        y_pred = neural_network_output(petal_length, petal_width, weights, bias)
        
        error = y_true - y_pred
        gradient_common_term = error * y_pred * (1 - y_pred)
        
        gradients[0] += gradient_common_term * petal_length
        gradients[1] += gradient_common_term * petal_width
        bias_gradient += gradient_common_term

    gradients = -2 * gradients / N
    bias_gradient = -2 * bias_gradient / N

    return gradients, bias_gradient


# Exercise 2d
def e2d(df):
    weights = np.array([-.11987, 0.3998])
    bias = 0.42
    mse = calculate_mse(df, weights, bias, df['pattern_class'])
    print(f"Mean-Squared Error: {mse}")
    plot_decision_boundary(df, weights, bias, f'Decision Boundary with MSE: {mse}')
    weights = np.array([-.19534, 0.3876])
    bias = 0.39
    mse = calculate_mse(df, weights, bias, df['pattern_class'])
    print(f"Mean-Squared Error: {mse}")
    plot_decision_boundary(df, weights, bias, f'Decision Boundary with MSE: {mse}')


def gradient_descent(weights, bias, data_vectors, pattern_classes, learning_rate=0.5, iter=10000, epsilon=0.00000001):
    
    mse_history = []
    sample = [0, 1, 2, 10, 11, 60, 61, 90, 91, 99]
    sample = [0, 1, 2]
    half_flag = True
    for i in range(iter):
        gradients, bias_gradient = calculate_gradients(data_vectors, weights, bias, pattern_classes)
        weights -= learning_rate * gradients
        bias -= learning_rate * bias_gradient
        mse = calculate_mse(data_vectors, weights, bias, pattern_classes)
        mse_history.append(mse)
        print(f"Iteration {i+1}/{iter}, MSE: {mse}") 
        if i in sample or (half_flag and mse < mse_history[0]/2) or (i > 3 and abs(mse - mse_history[i-1]) < epsilon):
            if half_flag and mse < mse_history[0]/2:
                half_flag = False
            if i > 3 and abs(mse - mse_history[i-1]) < epsilon:
                print('Convergence')
                return weights, bias, mse_history
            print(mse-mse_history[i-1])
            print(weights, bias)
            plot_decision_boundary(data_vectors, weights, bias, f'Decision Boundary after Iteration {i+1}')
            plot_learning_curve(mse_history, f'Learning Curve after Iteration {i+1}')
    return weights, bias, mse_history


def plot_learning_curve(mse_history, title='Learning Curve'):
    plt.plot(mse_history)
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.title(title)
    plt.show()


def main():
    # Load the iris data
    df = pd.read_csv('irisdata.csv')
    df = df[df['species'].isin(['versicolor', 'virginica'])]
    df['pattern_class'] = df['species'].apply(lambda x: 0 if x == 'versicolor' else 1)

    # Exercise 1a
    # plot_iris_data(df)

    # Exercise 1c
    # plot_decision_boundary(df, np.array([3.955, 7.1877]), -31.085, 'Decision Boundary Example')

    # Exercise 1d
    # plot_3d_output(df, np.array([3.955, 7.1877]), -31.085)

    # Exercise 1e
    # e1e(df, np.array([3.955, 7.1877]), -31.085)

    # Exercise 2b
    # e2b(df)

    # Exercise 2d
    e2d(df)


    # Exercise 3c
    weights = np.random.uniform(-.5, .5, 2)
    bias = np.random.uniform(-.5, .5)
    weights, bias, mse_history = gradient_descent(weights, bias, df, df['pattern_class'], learning_rate=2.0, iter=1000, epsilon=0.001)
    print(weights, bias)
    plot_decision_boundary(df, weights, bias, 'Decision Boundary after Gradient Descent')
    plot_learning_curve(mse_history)

    weights = np.random.uniform(-.5, .5, 2)
    bias = np.random.uniform(-.5, .5)
    weights, bias, mse_history = gradient_descent(weights, bias, df, df['pattern_class'], learning_rate=.1, iter=1000, epsilon=.01)
    print(weights, bias)
    plot_decision_boundary(df, weights, bias, 'Decision Boundary after Gradient Descent')
    plot_learning_curve(mse_history)

    # Exercise 3d
    weights = np.random.uniform(-.5, .5, 2)
    bias = np.random.uniform(-.5, .5)
    weights, bias, mse_history = gradient_descent(weights, bias, df, df['pattern_class'], learning_rate=0.5, iter=100000)
    print(weights, bias)
    plot_decision_boundary(df, weights, bias, 'Decision Boundary after Gradient Descent')
    plot_learning_curve(mse_history)


if __name__ == '__main__':
    main()