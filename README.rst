.. image:: https://raw.githubusercontent.com/eelregit/pmwd/master/assets/logo.svg?token=ABXY56TX3W73M5KITHFOPB3BHO3HM
   :alt: logo


particle mesh with derivatives
==============================

``pmwd`` is a differentiable cosmological particle-mesh forward model.
The C\ :sub:`2` symmetry of the name symbolizes the reversibility of the
model, which helps to dramatically reduce the memory cost when used with
the adjoint method.
It is built on ``jax`` for full differentiability.
