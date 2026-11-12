"""Candidate graphs: laboratory 2 models with extra, possibly absent edges.

Every candidate edge carries one regression parameter. A graph is a subset
of the candidates; an absent edge fixes its parameter at the ablation value,
exactly as ``Regression.fit(fixed=...)`` did in laboratory 2.
"""

from dataclasses import dataclass, replace

import numpy as np

from .lab2_config import DEFAULT_CONFIG as LAB2_CONFIG
from .lab2_config import OFFSET_PARAMETER_NAME
from .lab2_data import Dataset, SellmeierPredictor, prepare_datasets
from .lab3_circuit import coupled_current
from .lab3_config import (
    MUTUAL_ELASTANCE_BOUNDS,
    MUTUAL_RESISTANCE_BOUNDS,
    PICKUP_BOUNDS,
    POLYNOMIAL_BOUNDS,
    POLYNOMIAL_POWERS,
)

OUTPUT_NODE = "n² − 1"
SUPERSCRIPTS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
# Labels repeat the notation of the formulas (R₂, c₋₂), so that figures,
# tables and text name every edge in the same way.
SUBSCRIPTS = str.maketrans("-0123456789", "₋₀₁₂₃₄₅₆₇₈₉")
MATERIAL_TITLES = {"silica": "Кварцевое стекло", "water": "Вода"}


@dataclass(frozen=True)
class Edge:
    """A candidate edge and the regression parameter attached to it."""

    name: str
    label: str
    source: str
    target: str
    kind: str
    ablation: float
    truth: bool
    curve: float = 0.0
    directed: bool = True


@dataclass(frozen=True)
class GraphProblem:
    """A laboratory 2 dataset whose parameters are candidate graph edges."""

    name: str
    title: str
    dataset: Dataset
    edges: tuple
    positions: dict
    known_edges: tuple
    x_label: str
    y_label: str
    x_factor: float = 1.0
    y_factor: float = 1.0
    y_unit: str = ""
    label_side: str = "below"
    label_at: float = 0.5
    # Free space around the nodes: left/right, below, above.
    margin: tuple = (0.5, 0.6, 0.6)

    @property
    def truth(self):
        """Graph of the model that generated the data."""
        return tuple(edge.truth for edge in self.edges)

    def describe(self, mask):
        """Short text form of a graph, e.g. ``{M, R2}``."""
        labels = [e.label for e, present in zip(self.edges, mask) if present]
        return "{" + ", ".join(labels) + "}" if labels else "∅"


@dataclass(frozen=True)
class LibraryPredictor:
    """Laboratory 2 Sellmeier terms followed by polynomial terms."""

    sellmeier: SellmeierPredictor
    powers: tuple

    def design(self, wavelengths):
        wavelengths = np.asarray(wavelengths, dtype=float)
        polynomial = [wavelengths ** float(power) for power in self.powers]
        return np.column_stack(
            [self.sellmeier.design(wavelengths), *polynomial]
        )

    def __call__(self, wavelengths, theta):
        return self.design(wavelengths) @ theta


def _extend(dataset, name, predict, edges, initial, lower, upper):
    """Append candidate parameters to a laboratory 2 regression."""
    model = dataset.model
    extended = replace(
        model,
        names=tuple(edge.name for edge in edges),
        predict=predict,
        initial=np.r_[model.initial, initial],
        bounds=(np.r_[model.bounds[0], lower], np.r_[model.bounds[1], upper]),
    )
    ablations = {index: edge.ablation for index, edge in enumerate(edges)}
    return replace(dataset, name=name, model=extended, ablations=ablations)


def rlc_problem(datasets):
    """Reader/tag circuit: which couplings and losses do the data confirm?"""
    base = datasets["rlc"]
    mutual, loss = base.model.names
    edges = (
        Edge(
            mutual,
            "M",
            "первичный контур",
            "вторичный контур",
            "индуктивная связь",
            base.ablations[0],
            True,
            directed=False,
        ),
        Edge(
            loss,
            "R₂",
            "вторичный контур",
            "вторичный контур",
            "собственные потери",
            base.ablations[1],
            True,
        ),
        Edge(
            "Rm [ohm]",
            "Rₘ",
            "первичный контур",
            "вторичный контур",
            "резистивная связь",
            0.0,
            False,
            curve=-0.6,
            directed=False,
        ),
        Edge(
            "Sm [1/uF]",
            "Sₘ",
            "первичный контур",
            "вторичный контур",
            "ёмкостная связь",
            0.0,
            False,
            curve=0.6,
            directed=False,
        ),
        Edge(
            "E2 [mV]",
            "E₂",
            "источник",
            "вторичный контур",
            "наводка источника",
            0.0,
            False,
            curve=-0.62,
        ),
    )
    bounds = (MUTUAL_RESISTANCE_BOUNDS, MUTUAL_ELASTANCE_BOUNDS, PICKUP_BOUNDS)
    dataset = _extend(
        base,
        "rlc_couplings",
        coupled_current,
        edges,
        initial=np.zeros(3),
        lower=[low for low, _ in bounds],
        upper=[high for _, high in bounds],
    )
    return GraphProblem(
        name="rlc",
        title="Связанные контуры",
        dataset=dataset,
        edges=edges,
        positions={
            "источник": (0.0, 0.0),
            "первичный контур": (0.9, 0.0),
            "вторичный контур": (2.4, 0.0),
        },
        known_edges=(("источник", "первичный контур"),),
        x_label="время x, мкс",
        y_label="ток i₂, мА",
        x_factor=1e6,
        y_factor=1e3,
        y_unit=" мА",
        margin=(0.35, 0.5, 0.9),
    )


def _subscripted(name):
    """a1 -> a₁, c-2 -> c₋₂: the letter keeps its index as a subscript."""
    return name[0] + name[1:].translate(SUBSCRIPTS)


def _power_label(power):
    return "1" if power == 0 else "x" + str(power).translate(SUPERSCRIPTS)


def dispersion_problem(datasets, name):
    """Sellmeier resonances against polynomial terms for one material.

    Nodes are basis functions of the wavelength x, edges are their weights.
    """
    base = datasets[name]
    resonances = [
        Edge(
            term,
            _subscripted(term),
            _subscripted(f"φ{index + 1}"),
            OUTPUT_NODE,
            "резонанс Зельмейера",
            0.0,
            True,
        )
        for index, term in enumerate(base.model.names)
    ]
    polynomial = [
        Edge(
            OFFSET_PARAMETER_NAME if power == 0 else f"c{power}",
            _subscripted(f"c{power}"),
            _power_label(power),
            OUTPUT_NODE,
            "полиномиальный член",
            0.0,
            False,
        )
        for power in POLYNOMIAL_POWERS
    ]
    edges = (*resonances, *polynomial)
    count = len(POLYNOMIAL_POWERS)
    dataset = _extend(
        base,
        f"{name}_library",
        LibraryPredictor(base.model.predict, POLYNOMIAL_POWERS),
        edges,
        initial=np.zeros(count),
        lower=np.full(count, POLYNOMIAL_BOUNDS[0]),
        upper=np.full(count, POLYNOMIAL_BOUNDS[1]),
    )
    positions = {
        edge.source: (0.0, 0.62 * (len(edges) - 1 - row))
        for row, edge in enumerate(edges)
    }
    positions[OUTPUT_NODE] = (2.4, 0.31 * (len(edges) - 1))
    return GraphProblem(
        name=name,
        title=MATERIAL_TITLES[name],
        dataset=dataset,
        edges=edges,
        positions=positions,
        known_edges=(),
        x_label="длина волны x, мкм",
        y_label="n² − 1",
        label_side="outside",
        label_at=0.32,
        margin=(0.75, 0.4, 0.4),
    )


def prepare_problems(root):
    """Build all problems from the unchanged laboratory 2 datasets."""
    datasets = prepare_datasets(root, LAB2_CONFIG)
    problems = [rlc_problem(datasets)]
    problems += [
        dispersion_problem(datasets, name) for name in MATERIAL_TITLES
    ]
    return {problem.name: problem for problem in problems}
