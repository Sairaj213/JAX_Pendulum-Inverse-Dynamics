import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from pendulum_dynamics import pendulum_dynamics

@jax.jit
def simulate_pendulum(initial_state, time_points, g, L):
    return odeint(pendulum_dynamics, initial_state, time_points, g, L)


TOTAL_TIME = 10.0  
NUM_STEPS = 500    
TIME_POINTS = jnp.linspace(0, TOTAL_TIME, NUM_STEPS)

print(f"Simulation will run for {TOTAL_TIME} seconds over {NUM_STEPS} steps.")
