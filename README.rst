.. image:: assets/logo.svg?raw=true
   :width: 42 em
   :align: center
   :alt: logo


particle mesh with derivatives
==============================

``pmwd`` is a differentiable cosmological particle-mesh forward model.
The C\ :sub:`2` symmetry of the name symbolizes the reversibility of the
model, which helps to dramatically reduce the memory cost when used with
the adjoint method.
Based on ``JAX``, pmwd is fully differentiable, and is highly performant
on GPUs.

Particles align on the initial grid after evolving forward and then
backward in time.

.. raw:: html

  <video src="https://user-images.githubusercontent.com/7311098/212061133-a848247e-5a27-49a5-8846-28a7b4a7e4b4.mp4"></video>

Optimizing the initial conditions by gradient descent to make some
interesting projected patterns.

.. raw:: html

  <video src="https://user-images.githubusercontent.com/7311098/212061152-2b1be0ac-bfc4-4b57-87fe-d5b9b5c38e8c.mp4"></video>


Installation
------------

.. code:: sh

  pip install -e .  # to install in editable/develop mode

..
  pip install pmwd
  pip install -e .[dev]  # to install development dependencies


Examples
--------

See `docs/examples <docs/examples>`_.


..
  Testing
  -------

  .. code:: sh

    XLA_PYTHON_CLIENT_MEM_FRACTION=.05 python -m pytest --cov --cov-report=term-missing:skip-covered --durations=5 -n 16

  where `XLA_PYTHON_CLIENT_MEM_FRACTION=.05` makes JAX preallocate 5% of
  currently-available GPU memory, instead of the default 90%.

  .. code:: sh

    CUDA_VISIBLE_DEVICES= python -m pytest --cov --cov-report=term-missing:skip-covered --durations=5 -n 16

  disables CUDA (to run tests on CPUs).

  .. code:: sh

    python -m pytest --durations=5 --benchmark-columns=mean,ops,rounds,iterations tests/benchmark.py


..
  References & Citations
  ----------------------

  We refer the users to the following references for ...
  Please cite the following papers:

  .. code:: bibtex

    .. include:: CITATIONS.bib
