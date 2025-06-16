import jax
import jax.numpy as jnp

@jax.jit
def compute_loss(predicted_angles, target_angles):
    return jnp.mean(jnp.square(predicted_angles - target_angles))

print("Loss function defined: Mean Squared Error on angles.")
