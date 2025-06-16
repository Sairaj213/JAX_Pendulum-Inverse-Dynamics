import jax
import jax.numpy as jnp
from setup_imports import key
from objective import overall_objective_function, get_gradients
from target_trajectory import TRUE_INITIAL_THETA, TRUE_INITIAL_OMEGA

LEARNING_RATE = 0.1
NUM_EPOCHS = 1000

key, subkey = jax.random.split(key)
initial_guess = jax.random.uniform(subkey, shape=(2,), minval=-jnp.pi, maxval=jnp.pi)
initial_guess = jnp.array([initial_guess[0], initial_guess[1]/jnp.pi])

current_initial_conditions = initial_guess
print(f"Initial guess for [theta_0, omega_0]: {current_initial_conditions}")

loss_history = []
theta_history = []
omega_history = []

print("\nStarting optimization...")
for epoch in range(NUM_EPOCHS):
    loss = overall_objective_function(current_initial_conditions)
    gradients = get_gradients(current_initial_conditions)
    current_initial_conditions = current_initial_conditions - LEARNING_RATE * gradients

    loss_history.append(loss)
    theta_history.append(current_initial_conditions[0])
    omega_history.append(current_initial_conditions[1])

    if epoch % (NUM_EPOCHS // 10) == 0 or epoch == NUM_EPOCHS - 1:
        print(f"Epoch {epoch:4d} | Loss: {loss:.6f} | Current [theta_0, omega_0]: {current_initial_conditions[0]:.4f}, {current_initial_conditions[1]:.4f}")

print("\nOptimization complete.")
optimized_initial_conditions = current_initial_conditions
print(f"Optimized initial conditions: {optimized_initial_conditions}")
print(f"True initial conditions:      [{TRUE_INITIAL_THETA:.4f}, {TRUE_INITIAL_OMEGA:.4f}]")

loss_history = jnp.array(loss_history)
theta_history = jnp.array(theta_history)
omega_history = jnp.array(omega_history)

