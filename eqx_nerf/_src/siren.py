from typing import Optional, Literal, Union, Tuple, Callable
from jaxtyping import Float, Array
import math
import jax
import jax.numpy as jnp
import jax.random as jrandom

import equinox as eqx
from equinox import static_field
from equinox.nn.linear import Identity

PRNGKey = jax.random.PRNGKey


def sine_activation(x: Array, w0: float) -> Array:
    return w0 * x


def get_siren_init(dim: int, c: float, w0: float, is_first: bool):
    return (1 / dim) if is_first else (math.sqrt(c / dim) / w0)


class Sine(eqx.Module):
    w0: float = eqx.static_field()

    def __init__(self, w0: float):
        self.w0 = w0

    def __call__(self, x: Array) -> Array:
        return jnp.sin(self.w0 * x)


class Siren(eqx.Module):
    """Performs a linear transformation."""

    w0: float = static_field()
    c: float = static_field()
    weight: Array
    bias: Optional[Array]
    in_features: Union[int, Literal["scalar"]] = static_field()
    out_features: Union[int, Literal["scalar"]] = static_field()
    use_bias: bool = static_field()
    is_first: bool = static_field()

    def __init__(
        self,
        in_features: Union[int, Literal["scalar"]],
        out_features: Union[int, Literal["scalar"]],
        use_bias: bool = True,
        is_first: bool = False,
        w0: float = 1.0,
        c: float = 6.0,
        *,
        key: PRNGKey,
    ):
        super().__init__()
        wkey, bkey = jrandom.split(key, 2)
        in_features_ = 1 if in_features == "scalar" else in_features
        out_features_ = 1 if out_features == "scalar" else out_features

        lim = get_siren_init(dim=in_features_, c=c, w0=w0, is_first=is_first)

        self.weight = jrandom.uniform(
            wkey, (out_features_, in_features_), minval=-lim, maxval=lim
        )
        if use_bias:
            self.bias = jrandom.uniform(bkey, (out_features_,), minval=-lim, maxval=lim)
        else:
            self.bias = None

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.w0 = w0
        self.c = c
        self.is_first = is_first

    def __call__(self, x: Array, *, key: Optional[PRNGKey] = None) -> Array:
        if self.in_features == "scalar":
            if jnp.shape(x) != ():
                raise ValueError("x must have scalar shape")
            x = jnp.broadcast_to(x, (1,))
        x = self.weight @ x
        if self.bias is not None:
            x = x + self.bias
        if self.out_features == "scalar":
            assert jnp.shape(x) == (1,)
            x = jnp.squeeze(x)
        return sine_activation(x, self.w0)


class SirenNet(eqx.Module):
    """Standard Multi-Layer Perceptron; also known as a feed-forward network.
    !!! faq
        If you get a TypeError saying an object is not a valid JAX type, see the
            [FAQ](https://docs.kidger.site/equinox/faq/)."""

    layers: Tuple[Siren, ...]
    final_activation: Callable
    in_size: Union[int, Literal["scalar"]] = static_field()
    out_size: Union[int, Literal["scalar"]] = static_field()
    width_size: int = static_field()
    depth: int = static_field()
    w0_init: float = static_field()
    w0: float = static_field()
    c: float = static_field()

    def __init__(
        self,
        in_size: Union[int, Literal["scalar"]],
        out_size: Union[int, Literal["scalar"]],
        width_size: int,
        depth: int,
        w0_initial: float = 30.0,
        w0: float = 1.0,
        c: float = 6.0,
        final_activation: Callable = Identity(),
        *,
        key: PRNGKey,
        **kwargs,
    ):
        """**Arguments**:
        - `in_size`: The input size. The input to the module should be a vector of
            shape `(in_features,)`
        - `out_size`: The output size. The output from the module will be a vector
            of shape `(out_features,)`.
        - `width_size`: The size of each hidden layer.
        - `depth`: The number of hidden layers.
        - `activation`: The activation function after each hidden layer. Defaults to
            ReLU.
        - `final_activation`: The activation function after the output layer. Defaults
            to the identity.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        Note that `in_size` also supports the string `"scalar"` as a special value.
        In this case the input to the module should be of shape `()`.
        Likewise `out_size` can also be a string `"scalar"`, in which case the
        output from the module will have shape `()`.
        """

        super().__init__(**kwargs)
        keys = jrandom.split(key, depth + 1)
        layers = []
        if depth == 0:
            layers.append(Siren(in_size, out_size, w0=w0_initial, c=c, key=keys[0]))
        else:
            layers.append(Siren(in_size, width_size, w0=w0_initial, c=c, key=keys[0]))
            for i in range(depth - 1):
                layers.append(
                    Siren(width_size, width_size, w0=w0, c=c, key=keys[i + 1])
                )
            layers.append(Siren(width_size, out_size, w0=w0, c=c, key=keys[-1]))
        self.layers = tuple(layers)
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        self.final_activation = final_activation
        self.w0_init = w0_initial
        self.w0 = w0
        self.c = c

    def __call__(self, x: Array, *, key: Optional[PRNGKey] = None) -> Array:
        """**Arguments:**
        - `x`: A JAX array with shape `(in_size,)`. (Or shape `()` if
            `in_size="scalar"`.)
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)
        **Returns:**
        A JAX array with shape `(out_size,)`. (Or shape `()` if `out_size="scalar"`.)
        """
        for layer in self.layers[:-1]:
            x = layer(x)
        x = self.layers[-1](x)
        x = self.final_activation(x)
        return x
