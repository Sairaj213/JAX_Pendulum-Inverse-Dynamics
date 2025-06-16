import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML  
import yaml

print("JAX version:", jax.__version__)
print("JAX backend:", jax.default_backend())
key = jax.random.PRNGKey(0)
jax.config.update("jax_enable_x64", True)
print("JAX 64-bit precision enabled:", jax.config.read("jax_enable_x64"))

# Load config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Constants loaded from config
G = config["simulation"]["g"]
L_PENDULUM = config["simulation"]["L"]

TOTAL_TIME = config["simulation"]["total_time"]
NUM_STEPS = config["simulation"]["num_steps"]

TRUE_INITIAL_THETA = config["target_trajectory"]["initial_theta"]
TRUE_INITIAL_OMEGA = config["target_trajectory"]["initial_omega"]

EPOCHS = config["optimization"]["epochs"]
LEARNING_RATE = config["optimization"]["learning_rate"]
SEED = config["optimization"]["seed"]

# Derived constants
DT = TOTAL_TIME / NUM_STEPS
TIME_POINTS = jnp.linspace(0, TOTAL_TIME, NUM_STEPS)
