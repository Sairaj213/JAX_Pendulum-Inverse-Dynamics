import jax
import jax.numpy as jnp
from simulate import simulate_pendulum, TIME_POINTS
from pendulum_dynamics import G, L_PENDULUM
from loss import compute_loss
from target_trajectory import target_angles

@jax.jit
def overall_objective_function(initial_conditions_to_optimize):

    predicted_trajectory = simulate_pendulum(
        initial_conditions_to_optimize,
        TIME_POINTS,
        G,
        L_PENDULUM
    )
    predicted_angles = predicted_trajectory[:, 0]
    loss = compute_loss(predicted_angles, target_angles)
    return loss

get_gradients = jax.grad(overall_objective_function)

print("Overall objective function defined (simulates + computes loss).")
print("Gradient function created using jax.grad(overall_objective_function).")
