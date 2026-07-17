# Field constants

Prefilled [`StateFieldSpec`][state] handles for the fields a step reads or writes. Use them directly
when declaring a step's `state_reads`/`state_writes`, or call one to override an attribute inline
(for example `POSITION(heritable=False)`). The right column gives the underlying state field name.

## State fields

Importable from `jax_morph`. Present on every generated state.

| Constant | Field | Notes |
| --- | --- | --- |
| `POSITION` | `position` | Cell-centre coordinates, shape `(n_space_dim,)`; filled by the constructor. |
| `RADIUS` | `radius` | Per-cell radius. |
| `CELLTYPE` | `celltype` | One-hot cell type, shape `(n_types,)`; filled by the constructor. |
| `ALIVE` | `alive` | Boolean liveness mask; `dtype=bool`, defaults to `False`. |
| `TIME` | `t` | Simulation time; `global` scope, defaults to `0.0`. |

## Physics fields

Importable from `jax_morph.physics`. Added to the state by the physics steps that consume them; all
are per-cell and heritable.

| Constant | Field | Notes |
| --- | --- | --- |
| `GROWTH_RATE` | `growth_rate` | Per-cell radial growth rate, read by [`SaturatingCellGrowth`][cell-growth]. |
| `DIVISION_RATE` | `division_rate` | Per-cell division propensity, read by [`Division`][birth-death]. |
| `DEATH_RATE` | `death_rate` | Per-cell death propensity, read by [`Death`][birth-death]. |
| `ACTIVE_SPEED` | `active_speed` | Per-cell self-propulsion speed, read by [`ActiveBrownianDynamics2D`][dynamics]. |
| `ACTIVE_HEADING` | `active_heading` | Per-cell heading angle, read by [`ActiveBrownianDynamics2D`][dynamics]. |

[state]: state.md
[cell-growth]: ../physics/cell-growth.md
[dynamics]: ../physics/mechanics/dynamics.md
[birth-death]: ../physics/birth-death.md
