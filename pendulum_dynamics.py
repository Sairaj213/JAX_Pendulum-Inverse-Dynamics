import jax.numpy as jnp

def pendulum_dynamics(state, t, g, L):

    theta, omega = state
    d_theta_dt = omega
    d_omega_dt = -(g / L) * jnp.sin(theta)
    return jnp.array([d_theta_dt, d_omega_dt])

G = 9.81         
L_PENDULUM = 1.0 
print(f"Pendulum constants: g={G} m/s^2, L={L_PENDULUM} m")
