import jax
import jax.numpy as jnp
import os

print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")

@jax.jit
def f(x):
    return x + 1

print("Testing JIT compilation...")
try:
    res = f(jnp.ones(3))
    print(f"Result: {res}")
    print("JIT compilation successful!")
except Exception as e:
    print(f"Error during JIT: {e}")
