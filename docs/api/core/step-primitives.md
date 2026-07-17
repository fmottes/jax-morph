::: jax_morph.SimulationStep
    options:
        members:
            - state_reads
            - state_writes
            - state_requires
            - is_stochastic

---

::: jax_morph.StochasticStep
    options:
        members:
            - state_requires
            - trace_writes
            - sample_trace
            - trace_from_state
            - replay
            - logp

---

::: jax_morph.StepType

---

::: jax_morph.check_stochastic_step
