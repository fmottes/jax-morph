---
title: API Reference
hide:
  - toc
---

This reference maps the curated public surface of `jax_morph`. For end-to-end examples, start with
[Basic Usage](../basic-usage.md) or the [example notebooks](../example-notebooks/README.md).

## Core

*   [`StateFieldSpec`][state] and [`BaseState`][state] declare and hold model state;
    [`build_state_from_model`][state] synthesizes a typed state class.
*   [Geometry utilities][geometry]: displacement, distance, and neighbourhood helpers.
*   [`SimulationStep`][step-primitives], [`StochasticStep`][step-primitives], and
    [`StepType`][step-primitives] define validated steps; [`check_stochastic_step`][step-primitives]
    validates a stochastic step's trace contract.
*   [`Model`][model]: compose steps into a validated simulation pipeline.
*   [`simulate`][simulate]: sample a pathwise-differentiable rollout.
*   [`trajectory_logp`][scoring] and [`transition_logp`][scoring]: score sampled rollouts.
*   [Autodiff utilities][utilities]: straight-through and numerical helpers.
*   [Field constants][constants]: prefilled state and physics field specs for custom steps.

## Physics

*   [Interaction potentials][potentials]: force-free, pairwise, Morse, compact-repulsion, harmonic,
    and Lennard-Jones energy laws.
*   [Relaxation][relaxation]: [`MechanicalRelaxation`][relaxation] and
    [`relax_equilibrium`][relaxation], FIRE relaxation with implicit equilibrium gradients.
*   [Dynamics][dynamics]: [`BrownianDynamics`][dynamics] and
    [`ActiveBrownianDynamics2D`][dynamics], scorable stochastic motion.
*   [`VirialStress`][stress]: per-cell mechanical stress sensing.
*   [`FreeScreenedDiffusion`][diffusion]: screened-diffusion signalling.
*   [`SaturatingCellGrowth`][cell-growth]: saturating radial growth.
*   [Birth and death][birth-death]: [`Death`][birth-death], [`Division`][birth-death], and
    [`reconstruct_lineage`][birth-death], discrete population events and lineage recovery.

## Control

*   [`ODEController`][control]: base class for continuous-time controllers.
*   [`GeneNetworkConnectionist`][control], [`NeuralODE`][control], and [`GeneNetworkMWC`][control]:
    concrete controller families.

## Serialization

*   [`TrajectoryRecord`][serialization]: a loaded complete trajectory and its step size.
*   [`save_model`][serialization] and [`load_model`][serialization]: persist model numeric leaves
    against a caller-provided template.
*   [`save_state`][serialization] and [`load_state`][serialization]: persist generated state
    snapshots.
*   [`save_trajectory`][serialization] and [`load_trajectory`][serialization]: persist complete
    histories with metadata.

## Visualization

*   [Visualization][viz]: static cell-cluster rendering, trajectory animation, and per-cell field
    time series. Install the optional ``viz`` extra to render with matplotlib.

[state]: core/state.md
[geometry]: core/geometry.md
[step-primitives]: core/step-primitives.md
[model]: core/model.md
[simulate]: core/simulate.md
[scoring]: core/scoring.md
[utilities]: core/utilities.md
[constants]: core/constants.md
[potentials]: physics/mechanics/potentials.md
[relaxation]: physics/mechanics/relaxation.md
[dynamics]: physics/mechanics/dynamics.md
[stress]: physics/mechanics/stress.md
[diffusion]: physics/diffusion.md
[cell-growth]: physics/cell-growth.md
[birth-death]: physics/birth-death.md
[control]: control.md
[serialization]: serialization.md
[viz]: viz.md
