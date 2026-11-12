"""Settings of laboratory 3: candidate edges, intervention and sampler."""

from dataclasses import dataclass

from .lab2_config import OFFSET_BOUNDS

KERNEL_NAME = "emlab-lab3"
KERNEL_DISPLAY_NAME = "Python (EMLab — lab3)"
NOTEBOOK_PATH = "notebooks/lab3_metropolis_hastings.ipynb"
OUTPUT_PATH = "results/lab3"
CHAIN_SEED_OFFSET = 300

# Candidate couplings added to the laboratory 2 circuit. The data contain
# none of them, so the answer "absent" is known in advance. Bounds are the
# largest values that still look like a parasitic effect next to the main
# circuit: Rm below R1 = 2 ohm, a coupling capacitor above 5 nF, a pickup
# below half of the 1 V source.
MUTUAL_RESISTANCE_BOUNDS = (-1.0, 1.0)
MUTUAL_ELASTANCE_BOUNDS = (-200.0, 200.0)
PICKUP_BOUNDS = (-500.0, 500.0)

# Polynomial (Schott) terms c_p * x^p of the wavelength x, offered next to
# the Sellmeier resonances; p = 0 is the constant c0 of laboratory 2 and all
# coefficients share its bounds.
POLYNOMIAL_POWERS = (0, 2, -2, -4)
POLYNOMIAL_BOUNDS = OFFSET_BOUNDS


@dataclass(frozen=True)
class Lab3Config:
    """Intervention, objective and sampler settings shared by all problems."""

    seed: int = 20261029
    bootstrap_replicates: int = 200
    intervention_alpha: float = 2.0
    intervention_repeats: int = 3
    stability_weight: float = 1.0
    proposal_floor: float = 0.1
    swap_probability: float = 0.3
    # The second-stage target of the circuits has a rare graph (0.7% of the
    # mass) that a chain enters once in about 30 000 steps; shorter chains
    # estimate nothing but the selected graph.
    steps: int = 60000
    burn_in: int = 500
    # Graph spaces up to this size are also enumerated with the stability
    # term (a bootstrap per graph) to compare the chain with the exact target.
    exact_stability_edges: int = 5
    sequence_steps: int = 10
    stability_levels: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    weight_grid: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0)
    notebook_timeout_seconds: int = 7200

    def __post_init__(self):
        if self.bootstrap_replicates < 2 or self.intervention_repeats < 1:
            raise ValueError("Use at least two bootstrap draws and one repeat")
        if self.intervention_alpha <= 0:
            raise ValueError("The intervention must add noise")
        if self.stability_weight < 0:
            raise ValueError("The stability weight cannot be negative")
        # The floor keeps every flip possible, which irreducibility requires.
        if self.proposal_floor <= 0:
            raise ValueError("The proposal floor must be positive")
        # With no replacement move a chain of single flips can be periodic;
        # the aperiodicity argument of the notebook needs 0 < p_s < 1.
        if not 0 < self.swap_probability < 1:
            raise ValueError("The swap probability must lie in (0, 1)")
        if not 0 <= self.burn_in < self.steps:
            raise ValueError("Burn-in must be shorter than the chain")


DEFAULT_CONFIG = Lab3Config()
