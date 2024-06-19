# scripts/jax/optimization.py

# EvoJAX example (using a simple evolutionary strategy for optimization)
from evosax import OpenES, FitnessFunction
import jax

# Define fitness function
class MyFitness(FitnessFunction):
    def __init__(self, rng_key):
        self.rng_key = rng_key

    def reset(self):
        pass

    def evaluate(self, params, rng_key):
        # Example: fitness is negative loss
        loss = jnp.sum(params ** 2)
        return -loss

rng_key = jax.random.PRNGKey(0)
fitness = MyFitness(rng_key)
optimizer = OpenES(popsize=50, num_dims=2)

# Optimize
final_params, fitness_history = optimizer.run(fitness, rng_key, max_iter=100)
print("Optimized Parameters:", final_params)
