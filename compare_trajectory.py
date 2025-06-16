import matplotlib.pyplot as plt
from simulate import simulate_pendulum, TIME_POINTS
from pendulum_dynamics import G, L_PENDULUM
from target_trajectory import target_angles
from objective import overall_objective_function
from optimize import optimized_initial_conditions

def compare_optimized_with_target():
    
    optimized_trajectory = simulate_pendulum(
        optimized_initial_conditions,
        TIME_POINTS,
        G,
        L_PENDULUM
    )
    optimized_angles = optimized_trajectory[:, 0]

    plt.figure(figsize=(10, 6))
    plt.plot(TIME_POINTS, target_angles, label='Target Trajectory (Angle)', color='blue', linestyle='--')
    plt.plot(TIME_POINTS, optimized_angles, label='Optimized Trajectory (Angle)', color='red', alpha=0.7)
    plt.title('Comparison of Optimized vs. Target Pendulum Trajectory')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.grid(True)
    plt.legend()
    plt.show()

    final_loss = overall_objective_function(optimized_initial_conditions)
    print(f"Final loss with optimized conditions: {final_loss:.6f}")

if __name__ == "__main__":
    compare_optimized_with_target()
