"""Objective of a sampled graph: accuracy, complexity, intervention stability.

Everything statistical is delegated to laboratory 2: ``Regression.fit``,
``residual_bootstrap``, ``conditional_noise``, ``marginal_w1`` and the
deletion rule ``decide_elements``. This module only maps a graph (a mask of
candidate edges) to the regression over the parameters of its edges.

    L(G) = chi2(G) / 2 + c_m * (|E| + weight * sum_e u_e(G)),  c_m = ln(m) / 2

``u_e`` measures how far the intervention moves the bootstrap distribution
of the edge parameter relative to the distance that removes the edge.
"""

from dataclasses import dataclass, replace

import numpy as np

from .intervention import conditional_noise, marginal_w1, residual_bootstrap
from .lab2_config import (
    BOOTSTRAP_SEED_OFFSET,
    BOUNDARY_ABSOLUTE_TOLERANCE,
    BOUNDARY_RELATIVE_TOLERANCE,
    DECISION_LOSS,
    INTERVENTION_SEED_OFFSET,
)
from .lab2_config import DEFAULT_CONFIG as LAB2_CONFIG
from .lab2_experiment import decide_elements, rmse


@dataclass(frozen=True)
class Fit:
    """Most likely parameters of a graph; absent edges keep ablation values."""

    theta: np.ndarray
    chi2: float


@dataclass(frozen=True)
class Stability:
    """Response of the edge parameters of one graph to the intervention."""

    ratio: np.ndarray
    identified: np.ndarray
    instability: np.ndarray
    boundary_fraction: float


@dataclass(frozen=True)
class Loss:
    """The three terms of the objective, in nats."""

    accuracy: float
    complexity: float
    stability: float

    @property
    def total(self):
        return self.accuracy + self.complexity + self.stability


def boundary_share(samples, bounds):
    """Largest share of estimates of each parameter lying on a bound.

    ``samples`` holds the bootstrap draws before the intervention and after
    each noise realization. A parameter that sticks to a bound in any of
    them is not determined by the data, hence the maximum.
    """
    shares = []
    for draws in samples:
        on_bound = np.zeros(draws.shape, dtype=bool)
        for bound in bounds:
            on_bound |= np.isclose(
                draws,
                bound,
                atol=BOUNDARY_ABSOLUTE_TOLERANCE,
                rtol=BOUNDARY_RELATIVE_TOLERANCE,
            )
        shares.append(on_bound.mean(axis=0))
    return np.max(shares, axis=0)


class GraphScorer:
    """Evaluate graphs of one problem; every result is cached by mask.

    Bootstrap indices and intervention noise are drawn once (with the
    laboratory 2 seeds) and reused for all graphs. The loss is therefore a
    deterministic function of the graph, which Metropolis-Hastings requires,
    and the laboratory 2 graph reproduces the laboratory 2 draws.
    """

    def __init__(self, problem, config):
        self.problem = problem
        self.config = config
        model = problem.dataset.model
        size = len(model.y)
        self.ablations = np.array([edge.ablation for edge in problem.edges])
        self.edge_price = 0.5 * np.log(size)
        self.indices = np.random.default_rng(
            config.seed + BOOTSTRAP_SEED_OFFSET
        ).integers(0, size, size=(config.bootstrap_replicates, size))
        self.noise = np.random.default_rng(
            config.seed + INTERVENTION_SEED_OFFSET
        ).normal(size=(config.intervention_repeats, size))
        self._fits = {}
        self._stability = {}

    def _expand(self, mask, parameters):
        theta = self.ablations.copy()
        theta[np.array(mask, dtype=bool)] = parameters
        return theta

    def submodel(self, mask):
        """Laboratory 2 regression over the parameters of the present edges."""
        model = self.problem.dataset.model
        free = np.array(mask, dtype=bool)
        if not free.any():
            raise ValueError("The empty graph has no parameters to estimate")
        lower, upper = (
            np.asarray(bound, dtype=float) for bound in model.bounds
        )
        return replace(
            model,
            names=tuple(np.array(model.names)[free]),
            predict=lambda x, parameters: model.predict(
                x, self._expand(mask, parameters)
            ),
            initial=np.asarray(model.initial, dtype=float)[free],
            bounds=(lower[free], upper[free]),
        )

    def fit(self, mask):
        """First level of inference: parameters of a fixed graph."""
        if mask not in self._fits:
            model = self.problem.dataset.model
            theta = self.ablations.copy()
            if any(mask):
                point, _ = self.submodel(mask).fit(loss=DECISION_LOSS)
                theta = self._expand(mask, point)
            residual = (model.predict(model.x, theta) - model.y) / model.scale
            self._fits[mask] = Fit(theta, float(residual @ residual))
        return self._fits[mask]

    def stability(self, mask, alpha=None):
        """Bootstrap the graph before and after the intervention.

        rho_e = mean_r W1(P_e, P'_e,r) / W1(P_e, delta at the ablation value).
        The numerator is the laboratory 2 distance averaged over the noise
        realizations; the denominator, the mean absolute distance of the
        estimates from the value that removes the edge, turns it into
        "share of the way to deletion".
        """
        alpha = self.config.intervention_alpha if alpha is None else alpha
        key = (mask, float(alpha))
        if key in self._stability:
            return self._stability[key]
        free = np.array(mask, dtype=bool)
        count = len(mask)
        ratio = np.full(count, np.nan)
        identified = np.ones(count, dtype=bool)
        instability = np.zeros(count)
        boundary_fraction = 0.0
        if free.any():
            sub = self.submodel(mask)
            baseline, _, diagnostics = residual_bootstrap(
                sub, sub.y, self.indices, DECISION_LOSS
            )
            boundary_fraction = diagnostics["boundary_fraction"]
            samples, shifts = [baseline], []
            for noise in self.noise:
                observations = conditional_noise(
                    sub.y, sub.scale, alpha, noise
                )
                draws, _, diagnostics = residual_bootstrap(
                    sub, observations, self.indices, DECISION_LOSS
                )
                samples.append(draws)
                shifts.append(marginal_w1(baseline, draws)[0])
                boundary_fraction = max(
                    boundary_fraction, diagnostics["boundary_fraction"]
                )
            shift = np.mean(shifts, axis=0)
            on_bound = boundary_share(samples, sub.bounds)
            removal = np.abs(baseline - self.ablations[free]).mean(axis=0)
            ratio[free] = np.divide(
                shift,
                removal,
                out=np.full_like(shift, np.inf),
                where=removal > 0,
            )
            # A parameter that never moves (the model does not depend on it)
            # or sticks to a bound is not determined by the data. Its W1 is
            # small for the wrong reason, so the edge is declared unstable.
            identified[free] = (baseline.std(axis=0) > 0) & (
                on_bound <= LAB2_CONFIG.max_boundary_fraction
            )
            instability[free] = np.where(
                identified[free], np.minimum(ratio[free], 1.0), 1.0
            )
        result = Stability(ratio, identified, instability, boundary_fraction)
        self._stability[key] = result
        return result

    def parts(self, mask, weight=None):
        """Accuracy, complexity and stability of a graph.

        With ``weight == 0`` the intervention is not run at all: this is the
        first stage of the laboratory, selection without interventions.
        """
        weight = self.config.stability_weight if weight is None else weight
        stability = 0.0
        if weight > 0:
            unstable = self.stability(mask).instability.sum()
            stability = self.edge_price * weight * unstable
        return Loss(
            accuracy=0.5 * self.fit(mask).chi2,
            complexity=self.edge_price * sum(mask),
            stability=stability,
        )

    def loss(self, weight=None):
        """Loss function of the sampler for a given stability weight."""
        return lambda mask: self.parts(mask, weight).total

    def informed_weights(self, mask):
        """Proposal that uses the intervention on the current graph.

        Unstable present edges are offered for deletion more often; nothing
        is known about absent edges, so each keeps the unit weight.
        """
        weights = np.ones(len(mask))
        present = np.array(mask, dtype=bool)
        instability = self.stability(mask).instability
        weights[present] = self.config.proposal_floor + instability[present]
        return weights

    def predict(self, mask, x):
        return self.problem.dataset.model.predict(x, self.fit(mask).theta)

    def validation_rmse(self, mask):
        """Laboratory 2 accuracy measure on its held-out grid."""
        dataset = self.problem.dataset
        return rmse(
            self.predict(mask, dataset.validation_x), dataset.validation_y
        )

    def lab2_decisions(self, mask):
        """Apply the unchanged laboratory 2 deletion rule to a graph.

        Returns the laboratory 2 records keyed by the candidate edge index.
        """
        if sum(mask) < 2:
            # decide_elements refits the remaining parameters after an
            # ablation; a single-edge graph leaves nothing to refit.
            raise ValueError("The laboratory 2 rule needs at least two edges")
        present = np.flatnonzero(mask)
        dataset = replace(
            self.problem.dataset,
            model=self.submodel(mask),
            ablations={
                row: self.ablations[edge] for row, edge in enumerate(present)
            },
        )
        records = decide_elements(
            dataset,
            self.fit(mask).theta[present],
            self.stability(mask).boundary_fraction,
            LAB2_CONFIG,
        )
        return dict(zip(present.tolist(), records))
