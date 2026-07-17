---
title: Visualization
---

Install the optional matplotlib backend before calling these functions:

```bash
pip install 'jax-morph[viz]'
```

With uv, use `uv add 'jax-morph[viz]'`. The base package keeps matplotlib optional, while
`import jax_morph.viz` remains safe without the extra.

All functions return native matplotlib objects, so normal axes customization, notebook display,
and animation export remain available to the caller. `animate(..., ax=ax)` can target an existing
compatible axes; frame updates replace only the animation's own cell artist and categorical legend,
leaving caller-added artists, labels, and annotations intact.

## Static rendering

---
::: jax_morph.viz.draw

## Animation

---
::: jax_morph.viz.animate

## Per-cell time series

---
::: jax_morph.viz.plot_timeseries
