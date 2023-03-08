from typing import Optional, Literal, Union
from jaxtyping import Float, Array
import jax
import jax.numpy as jnp
import jax.random as jrandom
from equinox.custom_types import PRNGKey
import equinox as eqx
from equinox import static_field


def sine_activation(x: Array, w0: Array) -> Array:
    return w0 * x


def siren_init():
    pass


class Sine(eqx.Module):
    w0: Array = eqx.static_field()

    def __call__(self, x: Array) -> Array:
        return jnp.sin(self.w0 * x)


class SirenLayer(eqx.nn.Linear):
    """Performs a linear transformation."""

    weight: Array
    bias: Optional[Array]
    in_features: Union[int, Literal["scalar"]] = static_field()
    out_features: Union[int, Literal["scalar"]] = static_field()
    use_bias: bool = static_field()


# class SirenLayer(eqx.nn.MLP):
#     is_first: bool = eqx.static_field()
#     layers: tp.Tuple[eqx.nn.Linear, ...]
#     final_activation: Sine
#     weight: Array
#     bias: tp.Optional[Array]
#     in_size: tp.Union[int, tp.Literal["scalar"]] = eqx.static_field()
#     out_size: tp.Union[int, tp.Literal["scalar"]] = eqx.static_field()
#     width_size: int = eqx.static_field()
#     use_bias: bool = eqx.static_field()

#     def __init__(
#         self,
#         in_size: tp.Union[int, tp.Literal["scalar"]],
#         out_size: tp.Union[int, tp.Literal["scalar"]],
#         width_size: int,
#         depth: int,
#         activation: tp.Callable = jnn.relu,
#         final_activation: tp.Callable = _identity,
#         *,
#         key: PRNGKey,
#         **kwargs,
#     ):
#         pass
