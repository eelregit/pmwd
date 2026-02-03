"""Derivative-based Global Sensitivity Measure (DGSM) using JAX."""

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from typing import Callable, Any, Literal


def dgsm(
    func: Callable,
    inputs: Any,
    normalization: Literal['variance', 'elasticity', None] = None,
) -> dict:
    """Compute Derivative-based Global Sensitivity Measures (DGSM).

    DGSM measures the sensitivity of a function's output to its inputs using
    gradient information. Assumes scalar output function with PyTree inputs.

    Args
    ----
    func : Callable
        A JAX-compatible function: f(inputs) -> scalar, where inputs is a PyTree.
        Each leaf of the PyTree should have shape (n_inputs_i, ...).
    inputs : PyTree
        Input samples as a PyTree. The leading axis of each leaf corresponds
        to the sample dimension, shape (n_samples, ...) for each leaf.
    normalization : {'variance', 'elasticity', None}, default=None
        Normalization method for the sensitivity indices:
        - 'variance': Normalize by input variance and output variance (standard DGSM).
          Results in v = E[(∂f/∂x)²] * Var(x) / Var(y).
        - 'elasticity': Use elasticity (sample-wise normalization).
          Results in v = E[(∂f/∂x * x/y)²] = E[ε²], where ε = ∂ln(f)/∂ln(x).
        - None: No normalization, return raw gradient statistics.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'v': Mean squared sensitivity index, same PyTree structure as inputs
        - 'v_std': Std of squared sensitivity index
        - 'gradients': Raw gradients, PyTree with shape (n_samples, ...) per leaf
        - 'elasticity': Per-sample elasticity ε = (∂f/∂x)*(x/y), None if not 'elasticity' mode
        - 'outputs': Function outputs
    """
    func_valgrad = jax.value_and_grad(func)
    outputs, gradients = jax.vmap(func_valgrad)(inputs)

    if normalization == 'elasticity':
        # Elasticity: sample-wise normalization ε = (∂f/∂x) * (x / y)
        def compute_elasticity(grad, x):
            y_broadcast = outputs.reshape((-1,) + (1,) * (grad.ndim - 1))
            return grad * x / y_broadcast

        g = tree_map(compute_elasticity, gradients, inputs)
    else:
        g = gradients

    # DGSM indices
    v = tree_map(lambda g: jnp.mean(g**2, axis=0), g)
    v_std = tree_map(lambda g: jnp.std(g**2, axis=0), g)

    # Variance normalization
    if normalization == 'variance':
        input_vars = tree_map(lambda x: jnp.var(x, axis=0), inputs)
        output_var = jnp.var(outputs)
        v = tree_map(lambda val, var: val * var / output_var, v, input_vars)
        v_std = tree_map(lambda val, var: val * var / output_var, v_std, input_vars)

    return {
        'v': v,
        'v_std': v_std,
        'gradients': gradients,
        'elasticity': g if normalization == 'elasticity' else None,
        'outputs': outputs,
    }
