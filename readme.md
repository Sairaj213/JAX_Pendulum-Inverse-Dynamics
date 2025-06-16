# 🕰️ JAX_Pendulum-Inverse-Dynamics

The Project uses physics-based simulation and optimization implemented using JAX, it demonstrates how to recover the initial conditions of a pendulum system using gradient-based optimization from JAX. A simulated target trajectory is first generated using known initial values for the pendulum’s angle and angular velocity.using only this trajectory, the project attempts to learn the original initial conditions by minimizing the mean squared error between a predicted trajectory (from guessed values) and the target one. The differential equations are solved using odeint, gradients are computed using jax.grad, and optimization is done via simple gradient descent.


# 🗂️ Project Structure
```markdown-tree
📁 JAX_Pendulum-Inverse-Dynamics
├── main.py                         # Entry point
├── setup_imports.py                # Handles JAX imports, precision config, and random seed
├── pendulum_dynamics.py            # Defines pendulum ODE and physical constants
├── simulate.py                     # JIT-compiled function to run pendulum simulation
├── target_trajectory.py            # Generates and stores the target trajectory
├── loss.py                         # Mean squared error computation
├── objective.py                    # Objective function and gradient computation
├── optimize.py                     # Gradient descent optimization loop
├── visualize_progress.py           # Loss + parameter history visualization
├── compare_trajectory.py           # Final comparison plot between target and optimized output
├── requirements.txt                # Dependencies list
└── README.md                       # Project documentation
```


# 🚀 Getting Started 

### 📥 Clone the Repository
First, clone this repository to your local machine:
```bash
git clone https://github.com/Sairaj213/JAX_Pendulum-Inverse-Dynamics.git

cd JAX_Pendulum-Inverse-Dynamics
```
### ⚙️ Install Requirements

Ensure you’re using Python 3.9+. Then, install all necessary dependencies:

```bash
pip install -r requirements.txt
```

For GPU acceleration, refer to [JAX's official installation guide](https://github.com/jax-ml/jax#installation) based on your CUDA version.

### ▶️ Run the Simulation

Execute the main script:

```bash
python main.py
```
You’ll see:

* Basic setup logs (JAX version, constants)

* Target pendulum simulation

* Optimization progress

* Visualization of convergence

* Final comparison of optimized vs. true trajectory



