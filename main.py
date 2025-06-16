from setup_imports import *
from pendulum_dynamics import *
from simulate import *
from target_trajectory import *
from loss import *
from objective import *
from optimize import *
from visualize_progress import plot_optimization_progress
from compare_trajectory import compare_optimized_with_target

if __name__ == "__main__":
    print("=== Pendulum Trajectory Optimization ===\n")
    print(f"Using JAX backend: {jax.default_backend()}")
    print(f"JAX version: {jax.__version__}")
    print(f"64-bit precision: {jax.config.read('jax_enable_x64')}")
    print(f"Target initial conditions: theta={TRUE_INITIAL_THETA:.4f}, omega={TRUE_INITIAL_OMEGA:.4f}")
    print(f"Optimized initial conditions: {optimized_initial_conditions}")
    print(f"\nFinal loss: {overall_objective_function(optimized_initial_conditions):.6f}\n")

    print(">> Plotting optimization progress...")
    plot_optimization_progress()

    print("\n>> Comparing optimized vs. target trajectory...")
    compare_optimized_with_target()
