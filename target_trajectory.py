import jax.numpy as jnp
import matplotlib.pyplot as plt
from simulate import simulate_pendulum, TIME_POINTS
from pendulum_dynamics import G, L_PENDULUM

TRUE_INITIAL_THETA = jnp.pi / 4.0  
TRUE_INITIAL_OMEGA = 0.0          
true_initial_state = jnp.array([TRUE_INITIAL_THETA, TRUE_INITIAL_OMEGA])

target_trajectory = simulate_pendulum(true_initial_state, TIME_POINTS, G, L_PENDULUM)
target_angles = target_trajectory[:, 0]  

print(f"Target trajectory generated from initial_theta={TRUE_INITIAL_THETA:.2f} rad, initial_omega={TRUE_INITIAL_OMEGA:.2f} rad/s")

def plot_target_trajectory():
    plt.figure(figsize=(10, 4))
    plt.plot(TIME_POINTS, target_angles, label='Target Angle (rad)')
    plt.title('Target Pendulum Trajectory (Angle vs. Time)')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_target_trajectory()
