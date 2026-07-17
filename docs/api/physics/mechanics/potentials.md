::: jax_morph.physics.Potential
    options:
        members:
            - total_energy
            - forces
            - state_reads

---

::: jax_morph.physics.NoForce
    options:
        members:
            - total_energy
            - forces

---

::: jax_morph.physics.PairwisePotential
    options:
        members:
            - pair_energy
            - pair_params
            - mix
            - state_reads
            - total_energy
            - virial_pressure

---

::: jax_morph.physics.Morse
    options:
        members:
            - pair_params
            - pair_energy

---

::: jax_morph.physics.SoftSphere
    options:
        members:
            - pair_params
            - pair_energy

---

::: jax_morph.physics.Hertzian
    options:
        members:
            - pair_params
            - pair_energy

---

::: jax_morph.physics.Harmonic
    options:
        members:
            - pair_params
            - pair_energy

---

::: jax_morph.physics.LennardJones
    options:
        members:
            - pair_params
            - pair_energy
