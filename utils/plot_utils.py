import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable, Optional, Dict


def plot_results(best_x: np.array, best_cost: float, x_history: List[np.array], cost_history: List[float], cost_function: Callable, x_range: Optional[List[List[float]]] = None) -> None:
    x1_history = [x[0] for x in x_history]
    x2_history = [x[1] for x in x_history]

    # Create a 3D plot of the optimization landscape
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Generate a grid of x1 and x2 values for plotting the surface
    if x_range is not None:
        x1_range = np.linspace(x_range[0][0], x_range[0][1], 500)
        x2_range = np.linspace(x_range[1][0], x_range[1][1], 500)
    else:
        x1_range = np.linspace(min(x1_history) - 2, max(x1_history) + 2, 500)
        x2_range = np.linspace(min(x2_history) - 2, max(x2_history) + 2, 500)
    X1, X2 = np.meshgrid(x1_range, x2_range)

    # Initialize an empty array to store the cost values
    Z = np.zeros_like(X1)

    # Calculate the cost for each combination of X1 and X2
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i][j] = cost_function([X1[i][j], X2[i][j]])

    # Plot the surface
    ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)

    # Plot the optimization path
    ax.plot(x1_history, x2_history, cost_history, marker='o', linestyle='-', color='red', label='Optimization path')
    ax.plot(best_x[0], best_x[1], best_cost, marker='o', linestyle='-', color='blue', label='Best solution')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Cost')
    ax.set_title('Cost function and optimization')
    plt.legend()
    plt.show()

    # Calculate the extent for the 2D heatmap plot based on the actual range of the data
    x1_min, x1_max = min(x1_range), max(x1_range)
    x2_min, x2_max = min(x2_range), max(x2_range)

    # Create a 2D heatmap plot
    plt.figure(figsize=(8, 6))
    plt.imshow(Z, extent=(x1_min, x1_max, x2_min, x2_max), origin='lower', cmap='viridis', interpolation='bilinear')
    plt.colorbar(label='Cost')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Cost function and optimization')
    plt.grid(True)

    # Overlay the optimization path on the heatmap as red dots
    plt.plot(x1_history, x2_history, c='red', marker='o', linestyle='-', label='Optimization path')
    plt.plot(best_x[0], best_x[1], c='blue', marker='o', linestyle='-', label='Best solution')
    plt.legend()
    plt.show()

def plot_results_with_population(best_x: np.array, individuals: List[Dict], cost_function: Callable, x_range: Optional[List[List[float]]] = None) -> None:
    # Generate a grid of x1 and x2 values for plotting the surface
    x1_range = np.linspace(x_range[0][0], x_range[0][1], 500)
    x2_range = np.linspace(x_range[1][0], x_range[1][1], 500)
    X1, X2 = np.meshgrid(x1_range, x2_range)

    # Initialize an empty array to store the cost values
    Z = np.zeros_like(X1)

    # Calculate the cost for each combination of X1 and X2
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i][j] = cost_function([X1[i][j], X2[i][j]])

    # Calculate the extent for the 2D heatmap plot based on the actual range of the data
    x1_min, x1_max = min(x1_range), max(x1_range)
    x2_min, x2_max = min(x2_range), max(x2_range)

    # Create a 2D heatmap plot
    plt.figure(figsize=(8, 6))
    plt.imshow(Z, extent=(x1_min, x1_max, x2_min, x2_max), origin='lower', cmap='viridis', interpolation='bilinear')
    plt.colorbar(label='Cost')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Cost function and optimization')
    plt.grid(True)

    # Overlay the optimization path on the heatmap as red dots
    for i, individual_index in enumerate(range(len(individuals))):
        individuals[individual_index]['position_history'] = np.asarray(individuals[individual_index]['position_history'])
        if i == 0:
            label = 'Candidate solutions'
        else:
            label = None
        plt.plot(individuals[individual_index]['position_history'][:, 0], individuals[individual_index]['position_history'][:, 1], c='red', marker='o', linestyle='-', label=label)
    plt.plot(best_x[0], best_x[1], c='blue', marker='o', label='Best solution')
    plt.legend()
    plt.show()