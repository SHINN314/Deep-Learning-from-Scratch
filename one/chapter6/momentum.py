import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import common.optimizer as optimizer
import common.gradient as gradient
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x):
    """
    Two dimensional function x**2 / 10 + y**2
    where x[0] is the first dimension and x[1] is the second dimension.

    Parameters:
        x (list or np.ndarray): Input vector with two elements.
    Returns:
        float: The value of the function at the input vector.
    """
    return (x[0] ** 2) / 10 + x[1] ** 2


def plot_optimization_3d(history):
    """
    Plot the optimization process in 3D.
    
    Parameters:
        history: list of (step, params, function_value) tuples
    """
    # Extract data from history
    steps = [h[0] for h in history]
    x_vals = [h[1][0] for h in history]
    y_vals = [h[1][1] for h in history]
    z_vals = [h[2] for h in history]
    
    # Create 3D plot
    fig = plt.figure(figsize=(15, 5))
    
    # 3D surface plot
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Create meshgrid for surface plot
    x_range = np.linspace(-8, 1, 100)
    y_range = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = (X ** 2) / 10 + Y ** 2
    
    # Plot surface
    ax1.plot_surface(X, Y, Z, alpha=0.3, cmap='viridis')
    
    # Plot optimization path
    ax1.plot(x_vals, y_vals, z_vals, 'ro-', markersize=3, linewidth=2, label='Optimization Path')
    ax1.scatter(x_vals[0], y_vals[0], z_vals[0], color='green', s=100, label='Start')
    ax1.scatter(x_vals[-1], y_vals[-1], z_vals[-1], color='red', s=100, label='End')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('f(X,Y)')
    ax1.set_title('3D Optimization Path')
    ax1.legend()
    
    # 2D contour plot
    ax2 = fig.add_subplot(132)
    contour = ax2.contour(X, Y, Z, levels=20)
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.plot(x_vals, y_vals, 'ro-', markersize=3, linewidth=2, label='Optimization Path')
    ax2.scatter(x_vals[0], y_vals[0], color='green', s=100, label='Start')
    ax2.scatter(x_vals[-1], y_vals[-1], color='red', s=100, label='End')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('2D Contour Plot with Path')
    ax2.legend()
    ax2.grid(True)
    
    # Function value over iterations
    ax3 = fig.add_subplot(133)
    ax3.plot(steps, z_vals, 'b-', linewidth=2)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Function Value')
    ax3.set_title('Function Value vs Iteration')
    ax3.grid(True)
    ax3.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('momentum_optimization_3d.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("3D visualization saved as 'momentum_optimization_3d.png'")

def main():
    print("=== Momentum Optimizer Test ===")
    print("Function: f = x^2/10 + y^2")
    print("Starting point: (-7.0, 2.0)")
    print()
    
    # Initialize parameters and optimizer
    param = {
        "test": np.array([-7.0, 2.0]),
    }
    
    opt = optimizer.Momentum(lr=0.01, momentum=0.9)
    
    # Store optimization history
    history = []

    for i in range(500):
        # Calculate gradient
        grad = {
            "test": gradient.numerical_gradient(f, param["test"]),
        }
        
        # Store current state before update
        current_params = param["test"].copy()
        current_value = f(current_params)
        history.append((i, current_params, current_value))
        
        # Update parameters using optimizer
        opt.update(param, grad)
        
        # Print progress every 100 steps
        if i % 100 == 0:
            print(f"Step {i:3d}: x={current_params[0]:7.4f}, y={current_params[1]:7.4f}, f(x,y)={current_value:8.6f}")

    final_value = f(param["test"])
    print(f"Final   : x={param['test'][0]:7.4f}, y={param['test'][1]:7.4f}, f(x,y)={final_value:8.6f}")
    print(f"\nOptimization completed! Final value: {final_value:.8f}")
    print("Theoretical minimum: f(0,0) = 0")
    print()
    
    # Create 3D visualization
    print("Creating 3D visualization...")
    plot_optimization_3d(history)

if __name__=="__main__":
    main()