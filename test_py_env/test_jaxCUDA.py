import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random

print("✅ Devices:", jax.devices())

# Basic ops
a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
print("✅ Dot:", jnp.dot(a, b))

# Grad test
f = lambda x: x ** 2 + 3 * x + 2
print("✅ grad(f)(2.0):", grad(f)(2.0))

# JIT test
@jit
def compute(x):
    return jnp.sin(x) + jnp.cos(x)

x = jnp.linspace(0, 10, 1000000)
print("✅ JIT compute(x).shape:", compute(x).shape)

# RNG test
key = random.PRNGKey(0)
print("✅ RNG normal:", random.normal(key, (3,)))

# vmap test
print("✅ vmap square:", vmap(lambda x: x**2)(jnp.array([1., 2., 3.])))

# Grad + JIT test
@jit
def loss_fn(w, x, y):
    pred = jnp.dot(x, w)
    return jnp.mean((pred - y) ** 2)

x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
y = jnp.array([1.0, 2.0])
w = jnp.array([0.1, -0.2])
print("✅ loss grad:", grad(loss_fn)(w, x, y))
