import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot_iris_data_1a(df):

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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def neural_network_output(petal_length, petal_width, weights, bias):
    # Combine petal length and width into a single input array
    inputs = np.array([petal_length, petal_width])
    
    # Compute the weighted sum
    weighted_sum = np.dot(weights, inputs) + bias
    
    # Apply the sigmoid non-linearity
    output = sigmoid(weighted_sum)
    
    return output


def plot_decision_boundary(df, weights, bias):
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
    plt.title('Decision Boundary for Neural Network')
    plt.legend()
    plt.show()


def plot_3d_output(df, weights, bias):
    # Create a mesh grid
    x_min, x_max = df['petal_length'].min() - 1, df['petal_length'].max() + 1
    y_min, y_max = df['petal_width'].min() - 1, df['petal_width'].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Compute the neural network output for each point in the mesh grid
    Z = np.array([neural_network_output(x, y, weights, bias) for x, y in zip(np.ravel(xx), np.ravel(yy))])
    Z = Z.reshape(xx.shape)
    
    # Create a 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, Z, cmap='viridis', alpha=0.8)
    
    # # Plot the original data points
    # for species in df['species'].unique():
    #     subset = df[df['species'] == species]
    #     ax.scatter(subset['petal_length'], subset['petal_width'], subset['nn_output'], label=species)
    
    ax.set_xlabel('Petal Length')
    ax.set_ylabel('Petal Width')
    ax.set_zlabel('Neural Network Output')
    ax.set_title('3D Plot of Neural Network Output')
    ax.legend()
    plt.show()


def p1b(df):
    weights = np.array([0.35, 0.35])
    bias = -2.3
    df['nn_output'] = df.apply(lambda row: neural_network_output(row['petal_length'], row['petal_width'], weights, bias), axis=1)
    
    print(df[['petal_length', 'petal_width', 'nn_output']])

    # Plot the neural network output
    # plt.figure(figsize=(10, 6))
    # plt.scatter(df['petal_length'], df['petal_width'], c=df['nn_output'], cmap='viridis')
    # plt.colorbar(label='Neural Network Output')
    # plt.xlabel('Petal Length')
    # plt.ylabel('Petal Width')
    # plt.title('Neural Network Output for Petal Length and Width')
    # plt.show()
    # plot_decision_boundary(df, weights, bias)
    plot_3d_output(df, weights, bias)


def main():
    df = pd.read_csv('irisdata.csv')
    df = df[df['species'].isin(['versicolor', 'virginica'])]
    # plot_iris_data_1a(df)
    p1b(df)
    

if __name__ == '__main__':
    main()