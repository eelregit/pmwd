"""Derivative-based Global Sensitivity Measure (DGSM) using JAX."""

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from typing import Callable, Any


def dgsm(
    func: Callable,
    inputs: Any,
    normalize_input: bool = True,
    normalize_output: bool = True,
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
    normalize_input : bool, default=True
        Normalize indices by input variance. This is recommended when inputs
        have different scales/units.
    normalize_output : bool, default=True
        Normalize indices by output variance.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'v': Mean squared gradients (importance), same PyTree structure as inputs
        - 'v_abs': Mean absolute gradients (robustness)
        - 'sigma': Std of gradients (interaction)
        - 'mean_grad': Mean gradient
        - 'gradients': Raw gradients, PyTree with shape (n_samples, ...) per leaf
        - 'outputs': Function outputs
    """
    func_valgrad = jax.value_and_grad(func)
    outputs, gradients = jax.vmap(func_valgrad)(inputs)

    # DGSM indices (computed per leaf)
    v = tree_map(lambda g: jnp.mean(g**2, axis=0), gradients)
    v_abs = tree_map(lambda g: jnp.mean(jnp.abs(g), axis=0), gradients)
    sigma = tree_map(lambda g: jnp.std(g, axis=0), gradients)
    mean_grad = tree_map(lambda g: jnp.mean(g, axis=0), gradients)

    # Normalize by input variance (Chain Rule correction)
    if normalize_input:
        input_vars = tree_map(lambda x: jnp.var(x, axis=0), inputs)
        v = tree_map(lambda val, var: val * var, v, input_vars)

        # For linear metrics (v_abs, sigma, mean_grad), scale by std
        input_stds = tree_map(lambda x: jnp.sqrt(x), input_vars)
        v_abs = tree_map(lambda val, std: val * std, v_abs, input_stds)
        sigma = tree_map(lambda val, std: val * std, sigma, input_stds)
        mean_grad = tree_map(lambda val, std: val * std, mean_grad, input_stds)

    # Normalize by output variance
    if normalize_output:
        output_var = jnp.var(outputs)
        if output_var > 1e-10:
            v = tree_map(lambda x: x / output_var, v)
            v_abs = tree_map(lambda x: x / jnp.sqrt(output_var), v_abs)
            sigma = tree_map(lambda x: x / jnp.sqrt(output_var), sigma)
            mean_grad = tree_map(lambda x: x / jnp.sqrt(output_var), mean_grad)

    return {
        'v': v,
        'v_abs': v_abs,
        'sigma': sigma,
        'mean_grad': mean_grad,
        'gradients': gradients,
        'outputs': outputs,
    }
