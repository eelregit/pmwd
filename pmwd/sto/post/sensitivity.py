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
        - 'v_std': Std of squared gradients (spread of g²)
        - 'grad_std': Std of gradients (interaction)
        - 'grad_mean': Mean gradient
        - 'grad_abs_mean': Mean absolute gradient
        - 'gradients': Raw gradients, PyTree with shape (n_samples, ...) per leaf
        - 'outputs': Function outputs
    """
    func_valgrad = jax.value_and_grad(func)
    outputs, gradients = jax.vmap(func_valgrad)(inputs)

    # DGSM indices (computed per leaf)
    v = tree_map(lambda g: jnp.mean(g**2, axis=0), gradients)
    v_std = tree_map(lambda g: jnp.std(g**2, axis=0), gradients)
    grad_std = tree_map(lambda g: jnp.std(g, axis=0), gradients)
    grad_mean = tree_map(lambda g: jnp.mean(g, axis=0), gradients)
    grad_abs_mean = tree_map(lambda g: jnp.mean(jnp.abs(g), axis=0), gradients)

    # Normalize by input variance (Chain Rule correction)
    if normalize_input:
        input_vars = tree_map(lambda x: jnp.var(x, axis=0), inputs)
        v = tree_map(lambda val, var: val * var, v, input_vars)
        v_std = tree_map(lambda val, var: val * var, v_std, input_vars)

        # For linear metrics (grad_std, grad_mean), scale by std
        input_stds = tree_map(lambda x: jnp.sqrt(x), input_vars)
        grad_std = tree_map(lambda val, std: val * std, grad_std, input_stds)
        grad_mean = tree_map(lambda val, std: val * std, grad_mean, input_stds)
        grad_abs_mean = tree_map(lambda val, std: val * std, grad_abs_mean, input_stds)

    # Normalize by output variance
    if normalize_output:
        output_var = jnp.var(outputs)
        v = tree_map(lambda x: x / output_var, v)
        v_std = tree_map(lambda x: x / output_var, v_std)
        grad_std = tree_map(lambda x: x / jnp.sqrt(output_var), grad_std)
        grad_mean = tree_map(lambda x: x / jnp.sqrt(output_var), grad_mean)
        grad_abs_mean = tree_map(lambda x: x / jnp.sqrt(output_var), grad_abs_mean)

    return {
        'v': v,
        'v_std': v_std,
        'grad_std': grad_std,
        'grad_mean': grad_mean,
        'grad_abs_mean': grad_abs_mean,
        'gradients': gradients,
        'outputs': outputs,
    }


def dgsm_elasticity(
    func: Callable,
    inputs: Any,
) -> dict:
    """Compute elasticity-based Derivative Global Sensitivity Measures.

    This variant uses the elasticity formulation where normalization by x²/y²
    happens inside the expectation, giving the mean squared elasticity:
        E[(∂f/∂x)² * x² / y²]

    The elasticity ε = (∂f/∂x) * (x/y) measures the percentage change in output
    per percentage change in input, making this a scale-invariant sensitivity
    measure.

    Args
    ----
    func : Callable
        A JAX-compatible function: f(inputs) -> scalar, where inputs is a PyTree.
        Each leaf of the PyTree should have shape (n_inputs_i, ...).
    inputs : PyTree
        Input samples as a PyTree. The leading axis of each leaf corresponds
        to the sample dimension, shape (n_samples, ...) for each leaf.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'v': Mean squared elasticity E[ε²], same PyTree structure as inputs
        - 'v_std': Std of squared elasticity (spread of ε²)
        - 'elasticity_std': Std of elasticity
        - 'elasticity_mean': Mean elasticity E[ε]
        - 'elasticity_abs_mean': Mean absolute elasticity E[|ε|]
        - 'elasticities': Raw elasticities, PyTree with shape (n_samples, ...) per leaf
        - 'gradients': Raw gradients
        - 'outputs': Function outputs
    """
    func_valgrad = jax.value_and_grad(func)
    outputs, gradients = jax.vmap(func_valgrad)(inputs)

    # Compute elasticity per sample: ε = (∂f/∂x) × (x / y)
    # For each leaf, we broadcast x (shape: n_samples, ...) and y (shape: n_samples,)
    def compute_elasticity(grad, x):
        # grad shape: (n_samples, ...), x shape: (n_samples, ...), outputs shape: (n_samples,)
        # Need to broadcast outputs to match grad/x shape
        y_broadcast = outputs.reshape((-1,) + (1,) * (grad.ndim - 1))
        return grad * x / y_broadcast

    elasticities = tree_map(compute_elasticity, gradients, inputs)

    # Elasticity-based DGSM indices
    v = tree_map(lambda e: jnp.mean(e**2, axis=0), elasticities)
    v_std = tree_map(lambda e: jnp.std(e**2, axis=0), elasticities)
    elasticity_std = tree_map(lambda e: jnp.std(e, axis=0), elasticities)
    elasticity_mean = tree_map(lambda e: jnp.mean(e, axis=0), elasticities)
    elasticity_abs_mean = tree_map(lambda e: jnp.mean(jnp.abs(e), axis=0), elasticities)

    return {
        'v': v,
        'v_std': v_std,
        'elasticity_std': elasticity_std,
        'elasticity_mean': elasticity_mean,
        'elasticity_abs_mean': elasticity_abs_mean,
        'elasticities': elasticities,
        'gradients': gradients,
        'outputs': outputs,
    }
