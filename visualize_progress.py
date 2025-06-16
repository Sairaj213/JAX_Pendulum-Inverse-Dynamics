import matplotlib.pyplot as plt
from target_trajectory import TRUE_INITIAL_THETA, TRUE_INITIAL_OMEGA
from optimize import loss_history, theta_history, omega_history

def plot_optimization_progress():
    
    plt.figure(figsize=(14, 6))    
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.title('Loss History During Optimization')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error Loss')
    plt.grid(True)
    plt.yscale('log')  
    plt.subplot(1, 2, 2)
    plt.plot(theta_history, label='Optimized Initial Theta')
    plt.plot(omega_history, label='Optimized Initial Omega')
    plt.axhline(TRUE_INITIAL_THETA, color='r', linestyle='--', label='True Initial Theta')
    plt.axhline(TRUE_INITIAL_OMEGA, color='g', linestyle='--', label='True Initial Omega')
    plt.title('Evolution of Initial Conditions')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_optimization_progress()
