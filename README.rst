.. image:: assets/logo.svg?raw=true
   :alt: logo


particle mesh with derivatives
==============================

``pmwd`` is a differentiable cosmological particle-mesh forward model.
The C\ :sub:`2` symmetry of the name symbolizes the reversibility of the
model, which helps to dramatically reduce the memory cost when used with
the adjoint method.
Based on ``JAX``, pmwd is fully differentiable, and is highly performant
on GPUs.


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
