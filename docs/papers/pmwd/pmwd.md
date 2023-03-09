---
logo_path: joss-logo.png
aas_logo_path: aas-logo.png
bibliography: pmwd.bib
title: "\\pmwd: A Differentiable Cosmological Particle-Mesh $N$-body Library"
tags:
  - Python
  - cosmology
  - forward modeling
  - differentiable simulation
authors:
  - name: Yin Li
    orcid: 0000-0002-0701-1410
    affiliation: 1, 2, 3
  - name: Libin Lu
    orcid: 0000-0003-0745-9431
    affiliation: 2
  - name: Chirag Modi
    orcid: 0000-0002-1670-2248
    affiliation: 2, 3
  - name: Drew Jamieson
    orcid: 0000-0001-5044-7204
    affiliation: 4
  - name: Yucheng Zhang
    orcid: 0000-0002-9300-2632
    affiliation: 1, 5
  - name: Yu Feng
    orcid: 0000-0001-5590-0581
    affiliation: 6
  - name: Wenda Zhou
    orcid: 0000-0001-5549-7884
    affiliation: 2, 7
  - name: Ngai Pok Kwan
    orcid: 0000-0002-5929-6905
    affiliation: 8, 3
  - name: François Lanusse
    orcid: 0000-0001-7956-0542
    affiliation: 9
  - name: Leslie Greengard
    orcid: 0000-0003-2895-8715
    affiliation: 1, 10
affiliations:
  - name: Department of Mathematics and Theory, Peng Cheng Laboratory,
          Shenzhen, Guangdong 518066, China
    index: 1
  - name: Center for Computational Mathematics, Flatiron Institute, New
          York, New York 10010, USA
    index: 2
  - name: Center for Computational Astrophysics, Flatiron Institute, New
          York, New York 10010, USA
    index: 3
  - name: Max Planck Institute for Astrophysics, 85748 Garching bei
          München, Germany
    index: 4
  - name: Center for Cosmology and Particle Physics, Department of
          Physics, New York University, New York, New York 10003, USA
    index: 5
  - name: Berkeley Center for Cosmological Physics, University of
          California, Berkeley, California 94720, USA
    index: 6
  - name: Center for Data Science, New York University, New York, New
          York 10011, USA
    index: 7
  - name: Department of Physics, The Chinese University of Hong Kong,
          Hong Kong
    index: 8
  - name: AIM, CEA, CNRS, Université Paris-Saclay, Université Paris
          Diderot, Sorbonne Paris Cité, F-91191 Gif-sur-Yvette, France
    index: 9
  - name: Courant Institute, New York University, New York, New York
          10012, USA
    index: 10
citation_author: Li et al.
year: 2022
journal_name: to be submitted to Journal of Open Source Software
volume: XX
issue: X
page: XXXX
formatted_doi: 10.21105/joss.XXXXX
repository: https://github.com/eelregit/pmwd
archive_doi: https://doi.org/10.5281/zenodo.XXXXXXX
review_issue_url: https://github.com/openjournals/joss-reviews/issues/XXXX
editor_name: Editor Name
editor_url: https://editor_url.xyz
reviewers:
  - Reviewer A
  - Reviewer B
submitted: 2022-XX-XX
published: 2023-XX-XX
aas-doi: 10.3847/XXXXX
aas-journal: to be submitted to Astrophysical Journal Supplement Series
...


\vspace{2em}


# Summary

The formation of the large-scale structure, the evolution and
distribution of galaxies, quasars, and dark matter on cosmological
scales, requires numerical simulations.
Differentiable simulations provide gradients of the cosmological
parameters, that can accelerate the extraction of physical information
from statistical analyses of observational data.
The deep learning revolution has brought not only myriad powerful neural
networks, but also breakthroughs including automatic differentiation
(AD) tools and computational accelerators like GPUs, facilitating
forward modeling of the Universe with differentiable simulations.
Because AD needs to save the whole forward evolution history to
backpropagate gradients, current differentiable cosmological simulations
are limited by memory.
Using the adjoint method, with reverse time integration to reconstruct
the evolution history, we develop a differentiable cosmological
particle-mesh (PM) simulation library \pmwd{} (particle-mesh with
derivatives) with a low memory cost.
Based on the powerful AD library `JAX`, \pmwd{} is fully differentiable,
and is highly performant on GPUs.


\vspace{2em}


![\pmwd{} logo. The C<sub>2</sub> symmetry of the name symbolizes the
reversibility of the model, which helps to dramatically reduce the
memory cost together with the adjoint method.
\label{fig:logo}](logo.pdf){width=60%}


\clearpage


# Statement of Need

Current established workflows of statistical inference from cosmological
datasets involve reducing cleaned data to summary statistics like the
power spectrum, and predicting these statistics using perturbation
theories, semi-analytic models, or simulation-calibrated emulators.
Rapid advances in accelerator technology like GPUs opens the possibility
of direct simulation-based forward modeling and inference
[@CranmerEtAl2020], even at level of the fields before their compression
into summary statistics.
The forward modeling approach naturally account for the
cross-correlation of different observables, and can easily incorporate
systematic errors.
In addition, model differentiability can accelerate parameter constraint
with gradient-based optimization and inference.
A differentiable field-level forward model combines the two features and
is able to constrain physical parameters together with the initial
conditions of the Universe.

The first differentiable cosmological simulations, such as ELUCID and
BORG-PM [@ELUCID; @BORG-PM], were developed before the advent of modern
AD systems, and were based on implementations of analytic derivatives.
Later codes including `FastPM` and `FlowPM` [@SeljakEtAl2017; @FlowPM]
compute gradients using AD engines, namely `vmad` (written by the same
authors) and `TensorFlow`, respectively.
Both analytic differentiation and AD backpropagate the gradients through
the whole history, thus requires saving the states at all time steps in
memory.
Therefore, they are subject to a trade-off between time and space/mass
resolution, and typically can integrate for only tens of time steps,
unlike the standard non-differentiable simulations.

Alternatively, the adjoint method provides systematic ways of deriving
model gradients under constraints [@Pontryagin1962], such as the
$N$-body equations of motion in the simulated Universe [@adjoint].
The adjoint method evolves a dual set of equations backward in time,
dependent on the states in the forward run, which can be recovered by
reverse time integration for reversible dynamics, thereby dramatically
reducing the memory cost [@NeuralODE].
Our logo in \autoref{fig:logo} is inspired by such reversibility as well
as the `JAX` artistic style.
Furthermore, we take the discretize-then-optimize approach [e.g.,
@ANODE] to ensure gradients propagate backward along the same discrete
trajectory as taken by the forward time integration.
We derive and validate our adjoint method in @adjoint in more details.
The table below compares the differentiable cosmological simulation
codes.

Being both computation and memory efficient, \pmwd{} enables larger and
more accurate forward modeling, and will improve gradient based
optimization and inference.
Differentiable analytic, semi-analytic, and deep learning components
can run based on or in parallel with \pmwd{} simulations.
Examples include a growth function emulator [@KwanModiEtAl2022] and our
ongoing work on spatiotemporal optimization of the PM gravity solver
[@ZhangLiEtAl].
In the future, \pmwd{} will also facilitate the modeling of various
cosmological observables and the understanding of the astrophysics at
play.


         code      OSS      gradient   mem efficient   hardware
------------- ------------ ---------- --------------- ----------
       ELUCID               analytic                      CPU
      BORG-PM               analytic                      CPU
`FastPM-vmad` $\checkmark$     AD                         CPU
     `FlowPM` $\checkmark$     AD                       GPU/CPU
      \pmwd{} $\checkmark$   adjoint    $\checkmark$    GPU/CPU


# Acknowledgements

YL and YZ were supported by The Major Key Project of PCL.
The Flatiron Institute is supported by the Simons Foundation.


# References
