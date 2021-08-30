


pmwd: particle mesh with derivatives
====================================

``pmwd`` is a differentiable cosmological particle-mesh forward model.
The C\ :sub:`2` symmetry of the name symbolizes the reversibility of the
model, which helps to dramatically reduce the memory cost when used with
the adjoint method.
It is built on ``jax`` for full differentiability.
