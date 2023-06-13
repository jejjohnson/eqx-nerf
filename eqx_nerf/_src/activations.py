import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import static_field
from jaxtyping import Array


class Tanh(eqx.Module):
    """Tanh activation function."""

    def __call__(self, x: Array) -> Array:
        return jax.nn.tanh(x)


class ReLU(eqx.Module):
    def __call__(self, x: Array) -> Array:
        return jax.nn.relu(x)


class Swish(eqx.Module):
    """Swish activation Function"""

    beta: float = static_field()

    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def __call__(self, x: Array) -> Array:
        return x * jax.nn.sigmoid(self.beta * x)


class Sine(eqx.Module):
    """Sine activation function."""

    w0: float = eqx.static_field()

    def __init__(self, w0: float):
        """
        Args:
            w0 (int): the amplitude factor for the sine activation
        """
        self.w0 = w0

    def __call__(self, x: Array) -> Array:
        return jnp.sin(self.w0 * x)


def sine_activation(x: Array, w0: float) -> Array:
    return w0 * x
