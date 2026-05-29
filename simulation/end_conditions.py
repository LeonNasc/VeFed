"""
Simulation end conditions.

Each condition is a callable-like object with a check(world) → bool method.
WorldEngine accepts one (or a combination via AnyCondition) and sets
world.is_done = True the first time check() returns True.

Available conditions
--------------------
HorizonCondition(max_days)
    Stop after a fixed number of simulated days.
    Good default for reproducible FL experiments.

ExtinctionCondition(consecutive_days=3)
    Stop when the infectious compartment (I) has been empty for
    `consecutive_days` straight. The grace period avoids false positives
    from rare stochastic re-ignitions in large populations.

NoSusceptiblesCondition()
    Stop when S = 0 — no agent can catch the disease anymore.
    The epidemic is structurally over even if infected agents are
    still recovering. Natural stopping point for attack-rate analysis.

TrainingBudgetCondition(target_events)
    Stop once the clinic queue has processed at least `target_events`
    diagnostic events. Useful for FL: "collect N labelled examples
    then stop" regardless of epidemic state.

Combinators
-----------
AnyCondition(*conditions)   — OR: stops when the first condition fires
AllConditions(*conditions)  — AND: stops when every condition has fired

Example
-------
    from simulation.end_conditions import AnyCondition, HorizonCondition, NoSusceptiblesCondition
    cond = AnyCondition(HorizonCondition(90), NoSusceptiblesCondition())
    world = WorldEngine(num_agents=50, end_condition=cond)
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class EndCondition(ABC):
    """Base class for all simulation end conditions."""

    @abstractmethod
    def check(self, world) -> bool:
        """Return True if the simulation should stop after this day."""
        ...

    @property
    @abstractmethod
    def reason(self) -> str:
        """Human-readable explanation used in logs and W&B."""
        ...

    def reset(self) -> None:
        """Reset any internal state (called if the world is reused)."""


# ── Concrete conditions ────────────────────────────────────────────────────────

class HorizonCondition(EndCondition):
    """
    Stop after a fixed number of simulated days.

    The simplest and most reproducible condition for FL experiments —
    every silo runs for exactly max_days regardless of epidemic state.
    """

    def __init__(self, max_days: int):
        if max_days < 1:
            raise ValueError("max_days must be >= 1")
        self.max_days = max_days

    def check(self, world) -> bool:
        return world.current_day >= self.max_days

    @property
    def reason(self) -> str:
        return f"time horizon reached ({self.max_days} days)"


class ExtinctionCondition(EndCondition):
    """
    Stop when I = 0 for `consecutive_days` straight.

    The grace period prevents premature stopping in large populations
    where a single re-ignition event can restart the outbreak.
    """

    def __init__(self, consecutive_days: int = 3):
        if consecutive_days < 1:
            raise ValueError("consecutive_days must be >= 1")
        self.consecutive_days = consecutive_days
        self._zero_streak: int = 0

    def check(self, world) -> bool:
        if world.sir_model.I == 0:
            self._zero_streak += 1
        else:
            self._zero_streak = 0
        return self._zero_streak >= self.consecutive_days

    @property
    def reason(self) -> str:
        return f"epidemic extinct (I=0 for {self.consecutive_days} consecutive days)"

    def reset(self) -> None:
        self._zero_streak = 0


class NoSusceptiblesCondition(EndCondition):
    """
    Stop when S = 0 — no agent can be newly infected.

    The epidemic is structurally over: transmission chain is broken
    even while infected agents may still be recovering. Natural stopping
    point when attack rate (fraction ever infected) is the outcome of interest.
    """

    def check(self, world) -> bool:
        return world.sir_model.S == 0

    @property
    def reason(self) -> str:
        return "no susceptibles remaining (S=0)"


class TrainingBudgetCondition(EndCondition):
    """
    Stop once the clinic queue has processed at least `target_events`.

    Useful for FL training: "collect N labelled (symptom_text, label) pairs
    then stop", decoupling run length from epidemic dynamics.
    Each FL silo has its own counter so silos may stop at different rounds.
    """

    def __init__(self, target_events: int):
        if target_events < 1:
            raise ValueError("target_events must be >= 1")
        self.target_events = target_events

    def check(self, world) -> bool:
        return len(world.clinic_queue.processed) >= self.target_events

    @property
    def reason(self) -> str:
        return f"training budget reached ({self.target_events} events)"


# ── Combinators ────────────────────────────────────────────────────────────────

class AnyCondition(EndCondition):
    """
    OR combinator — stops when the first child condition fires.
    The fired condition's reason is reported.
    """

    def __init__(self, *conditions: EndCondition):
        if not conditions:
            raise ValueError("AnyCondition requires at least one child")
        self._conditions = conditions
        self._fired: EndCondition | None = None

    def check(self, world) -> bool:
        for cond in self._conditions:
            if cond.check(world):
                self._fired = cond
                return True
        return False

    @property
    def reason(self) -> str:
        if self._fired:
            return self._fired.reason
        return "any condition (none fired yet)"

    def reset(self) -> None:
        self._fired = None
        for cond in self._conditions:
            cond.reset()


class AllConditions(EndCondition):
    """
    AND combinator — stops only when every child condition has fired.
    """

    def __init__(self, *conditions: EndCondition):
        if not conditions:
            raise ValueError("AllConditions requires at least one child")
        self._conditions = conditions

    def check(self, world) -> bool:
        return all(cond.check(world) for cond in self._conditions)

    @property
    def reason(self) -> str:
        return "all conditions met: " + "; ".join(c.reason for c in self._conditions)

    def reset(self) -> None:
        for cond in self._conditions:
            cond.reset()


# ── Factory for string-based config (used by fl/train.py CLI) ─────────────────

def from_config(name: str, param: int | None = None) -> EndCondition:
    """
    Build an EndCondition from a string name + optional integer parameter.

    name            param meaning         default param
    ────────────────────────────────────────────────────
    "horizon"       max_days              90
    "extinction"    consecutive_days      3
    "no_susceptibles"  —                 (ignored)
    "budget"        target_events         500
    """
    name = name.strip().lower().replace("-", "_").replace(" ", "_")
    if name == "horizon":
        return HorizonCondition(param if param is not None else 90)
    elif name == "extinction":
        return ExtinctionCondition(param if param is not None else 3)
    elif name in ("no_susceptibles", "nosusceptibles"):
        return NoSusceptiblesCondition()
    elif name == "budget":
        return TrainingBudgetCondition(param if param is not None else 500)
    else:
        raise ValueError(
            f"Unknown end condition {name!r}. "
            "Choose: horizon, extinction, no_susceptibles, budget"
        )
