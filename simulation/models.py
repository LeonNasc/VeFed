"""
Core simulation data models.
Formalizes §3 of Nascimento et al.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from simulation.symptom_language import Personality, SymptomNarrator
from simulation.case_table import CaseTable


# ─── Enumerations ────────────────────────────────────────────────────────────

class HealthStatus(Enum):
    SUSCEPTIBLE = "S"
    INFECTED    = "I"
    RECOVERING  = "R"
    RECOVERED   = "V"   # fully cleared


class LocationType(Enum):
    HOME       = "Home"
    WORK       = "Work"
    THIRD      = "ThirdPlace"
    COMMUTING  = "Commuting"
    HOSPITAL   = "Hospital"


class DiagnosticAction(Enum):
    INQUIRY     = "inquiry"       # request more behavioural context
    INVESTIGATE = "investigate"   # run virtual exams (latent state peek)
    RECOVER     = "home_recovery" # send home
    HOSPITALISE = "hospitalise"   # admit to hospital
    RESOLVE     = "resolve"       # assign diagnosis + treatment


# ─── HealthState (§3.2) ───────────────────────────────────────────────────────

class HealthState:
    """
    Trajectory-driven disease progression.

    Severity s(t) follows a skewed bell curve sampled from a
    DiseaseProgressionStrategy at infection time:
      - Logistic rise to peak_severity over rise_days
      - Exponential decay after peak at rate decay_rate

    Symptoms σ(t) = sensitivity × ds/dt — the derivative of severity.
    High on the upslope, near-zero at plateau, zero on downslope.

    The A/B matrix formulation is retired; progression is fully
    determined by the individual DiseaseTrajectory.
    """

    def __init__(self):
        self.status:       HealthStatus = HealthStatus.SUSCEPTIBLE
        self.severity:     float        = 0.0
        self.symptoms:     float        = 0.0
        self.days_infected: int         = 0
        self._trajectory                = None   # set at infect()

    def infect(self, trajectory) -> None:
        """
        Begin illness with the given DiseaseTrajectory.
        trajectory is sampled from DiseaseProgressionStrategy.sample_trajectory().
        """
        if self.status != HealthStatus.SUSCEPTIBLE:
            return
        self.status        = HealthStatus.INFECTED
        self.days_infected = 0
        self._trajectory   = trajectory
        self.severity      = 0.0
        self.symptoms      = 0.0

    def step(self) -> None:
        """Advance one simulated day along the trajectory."""
        if self.status == HealthStatus.SUSCEPTIBLE:
            return

        if self._trajectory is None:
            return

        self.days_infected += 1
        s, sigma = self._trajectory.step()
        self.severity = s
        self.symptoms = sigma

        # Status transitions driven by trajectory shape
        if self.status == HealthStatus.INFECTED:
            if self._trajectory._at_peak and self._trajectory.decay_rate > 0.005:
                self.status = HealthStatus.RECOVERING

        elif self.status == HealthStatus.RECOVERING:
            if self._trajectory.cleared:
                self.status   = HealthStatus.RECOVERED
                self.severity = 0.0
                self.symptoms = 0.0

    def apply_treatment(self, decay_boost: float = 0.15) -> None:
        """Accelerate recovery — delegates to trajectory."""
        if self._trajectory is not None:
            self._trajectory.apply_treatment(decay_boost)
            if self.status == HealthStatus.INFECTED:
                self.status = HealthStatus.RECOVERING

    @property
    def sir_compartment(self) -> str:
        if self.status == HealthStatus.SUSCEPTIBLE:
            return "S"
        if self.status == HealthStatus.INFECTED:
            return "I"
        return "R"


# ─── InnerState ───────────────────────────────────────────────────────────────

@dataclass
class InnerState:
    """
    Snapshot of an agent's current clinical picture, used for rich symptom
    narration. Combines HealthState (severity/σ), DiseaseTrajectory (trend),
    CaseTable (vitals), and personality (mood).

    Produced by Agent.inner_state; consumed by SymptomNarrator.full_opening_statement.
    """
    severity:       float          # actual disease severity s(t)
    symptoms:       float          # σ(t) post-floor
    days_infected:  int
    trend:          str            # "worsening" | "stable" | "improving"
    fatigue:        float          # 0–10 from CaseTable
    pain:           float          # 0–10 from CaseTable
    mood:           float          # 0–1; lower = more distressed
    top_vital:      tuple | None   # (variable, value, band) most abnormal vital sign


# ─── DiseaseCloud ─────────────────────────────────────────────────────────────

@dataclass
class DiseaseCloud:
    """Ambient viral load at a location — ω(l_t) in eq. 2."""
    location_id: str
    intensity: float          # 0..1
    decay_rate: float = 0.05  # loses 5 % intensity per tick

    def decay(self) -> None:
        self.intensity = max(0.0, self.intensity - self.decay_rate)

    @property
    def active(self) -> bool:
        return self.intensity > 0.01


# ─── Location ─────────────────────────────────────────────────────────────────

class Location:
    """
    Spatial node of the simulation graph.
    Manages agent occupancy and contact risk.
    """
    # Transmission modifier per location type (infection_modifier)
    TYPE_MODIFIERS = {
        LocationType.HOME:      0.6,
        LocationType.WORK:      1.0,
        LocationType.THIRD:     1.4,
        LocationType.COMMUTING: 0.8,
        LocationType.HOSPITAL:  0.3,  # PPE / isolation
    }

    def __init__(self, loc_id: str, loc_type: LocationType,
                 capacity: int = 20, row: int = 0, col: int = 0):
        self.id               = loc_id
        self.type             = loc_type
        self.capacity         = capacity
        self.row              = row
        self.col              = col
        self.agents_present:  set[str] = set()
        self.infection_modifier: float = self.TYPE_MODIFIERS[loc_type]
        self._clouds:         list[DiseaseCloud] = []

    def enter(self, agent_id: str) -> None:
        self.agents_present.add(agent_id)

    def exit(self, agent_id: str) -> None:
        self.agents_present.discard(agent_id)

    def add_cloud(self, cloud: DiseaseCloud) -> None:
        self._clouds.append(cloud)

    def ambient_exposure(self) -> float:
        """ω(l_t): sum of active disease-cloud intensities × location modifier."""
        return sum(c.intensity for c in self._clouds if c.active) * self.infection_modifier

    def tick_clouds(self) -> None:
        for c in self._clouds:
            c.decay()
        self._clouds = [c for c in self._clouds if c.active]

    @property
    def occupancy(self) -> int:
        return len(self.agents_present)

    @property
    def short_name(self) -> str:
        icons = {
            LocationType.HOME:      "H",
            LocationType.WORK:      "W",
            LocationType.THIRD:     "T",
            LocationType.COMMUTING: "~",
            LocationType.HOSPITAL:  "+",
        }
        return icons.get(self.type, "?")


# ─── DailySchedule ────────────────────────────────────────────────────────────

@dataclass
class DailySchedule:
    """
    Mobility policy π(a, t, type).
    288 ticks/day (5-min slots). Maps tick ranges → LocationType.
    """
    home_id:  str
    work_id:  str
    third_id: str

    def location_for_tick(self, tick: int) -> tuple[str, LocationType]:
        """Return (location_id, LocationType) for the given tick [0..287]."""
        # Sleep: 0–71 (00:00–06:00), Commute out: 72–83
        # Work: 84–191 (07:00–16:00), Commute back: 192–203
        # Third place: 204–239 (17:00–20:00), Home: 240–287
        if tick < 72:
            return self.home_id,  LocationType.HOME
        elif tick < 84:
            return "commute",     LocationType.COMMUTING
        elif tick < 192:
            return self.work_id,  LocationType.WORK
        elif tick < 204:
            return "commute",     LocationType.COMMUTING
        elif tick < 240:
            return self.third_id, LocationType.THIRD
        else:
            return self.home_id,  LocationType.HOME


# ─── DiagnosticEvent ──────────────────────────────────────────────────────────

@dataclass
class DiagnosticEvent:
    """
    e_i = (x_i, y_i) from §3.3.
    x_i: clinical summary (severity s, symptoms σ)
    y_i: ground-truth label (filled in by oracle / LLM)
    """
    agent_id:       str
    severity:       float
    symptoms:       float
    days_infected:  int
    action:         Optional[DiagnosticAction] = None
    oracle_label:   Optional[str] = None
    notes:          str = ""
    personality:    Optional[object] = None   # Personality enum
    conversation:   list = field(default_factory=list)  # [{role, text}]
    case_table:     Optional[object] = None   # CaseTable instance
    diagnosis:      Optional[str] = None       # LLM's diagnosis text
    ground_truth:   Optional[str] = None       # Actual diagnosis for accuracy


# ─── Agent ────────────────────────────────────────────────────────────────────

class Agent:
    """
    Individual simulation unit (§4.2).
    Maintains HealthState and behavioural parameters.
    """
    def __init__(self, agent_id: str, name: str, schedule: DailySchedule,
                 severity_threshold: float, rng: random.Random,
                 personality: Personality = Personality.NEUTRAL):
        self.id                = agent_id
        self.name              = name
        self.schedule          = schedule
        self.health_state      = HealthState()
        self.severity_threshold: float = severity_threshold  # γ_a
        self.current_location: str   = schedule.home_id
        self.current_type:     LocationType = LocationType.HOME
        self.hospitalised:     bool  = False
        self.cumulative_exposure: float = 0.0   # Σ ε_t(a) for the day
        self._rng              = rng
        self.personality       = personality
        self._narrator         = SymptomNarrator(rng)
        self._case_table       = None  # set at infection
        self._care_cooldown:   int   = 0   # days until next clinic visit allowed
        self.log:              list[str] = []

    def move_to(self, location_id: str, loc_type: LocationType) -> None:
        self.current_location = location_id
        self.current_type     = loc_type

    def update_health(self, exposure: float) -> None:
        """Accumulate tick exposure; eq. 2."""
        self.cumulative_exposure += exposure

    def apply_daily_infection(self, progression=None) -> bool:
        """
        At end of day, stochastic S→I transition: eq. 3.
        Samples a DiseaseTrajectory from progression strategy.
        Returns True if newly infected.
        """
        if self.health_state.status != HealthStatus.SUSCEPTIBLE:
            return False
        p_infect = 1.0 - math.exp(-self.cumulative_exposure)
        if self._rng.random() < p_infect:
            if progression is not None:
                traj = progression.sample_trajectory(self._rng)
            else:
                from simulation.progression import StandardFluProgression
                traj = StandardFluProgression().sample_trajectory(self._rng)
            self.health_state.infect(traj)
            # MIMIC trajectories carry their own vitals — use MimicCaseTable
            if hasattr(traj, "_record"):
                from simulation.case_table import MimicCaseTable
                self._case_table = MimicCaseTable(traj._record, self._rng)
            else:
                # Synthetic trajectory: generate correlated vitals (90d covers Slow Burn / Deadly)
                self._case_table = CaseTable(traj, self._rng, max_days=90)
            self.log.append("Contracted infection.")
            return True
        return False

    def should_seek_care(self) -> bool:
        """
        Agent seeks care under three conditions:
          1. Critically ill (INFECTED, severity ≥ 0.70) — always seek care.
          2. Still quite sick during recovery (RECOVERING, severity ≥ 0.40) —
             at treatment threshold a patient would reasonably keep visiting.
             This is the key path for slow-recovering diseases (Persistent Flu,
             Slow Burn) whose agents are RECOVERING but still unwell for weeks.
          3. Symptom signal σ ≥ personal threshold γ_a (both statuses).
        """
        if self.hospitalised:
            return False
        if self._care_cooldown > 0:
            return False
        hs = self.health_state
        if hs.status not in (HealthStatus.INFECTED, HealthStatus.RECOVERING):
            return False
        if hs.status == HealthStatus.INFECTED and hs.severity >= 0.70:
            return True
        if hs.status == HealthStatus.RECOVERING and hs.severity >= 0.30:
            return True
        return hs.symptoms >= self.severity_threshold

    def reset_daily_exposure(self) -> None:
        self.cumulative_exposure = 0.0
        if self._care_cooldown > 0:
            self._care_cooldown -= 1

    @property
    def inner_state(self) -> 'InnerState':
        """Current clinical snapshot — feeds full_opening_statement."""
        hs  = self.health_state
        day = hs.days_infected
        traj = hs._trajectory
        trend = traj.trend if traj is not None else "stable"

        fatigue    = 0.0
        pain       = 0.0
        top_vital  = None
        if self._case_table is not None:
            fatigue   = self._case_table.get("fatigue", day) or 0.0
            pain      = self._case_table.get("pain",    day) or 0.0
            top_vital = self._most_abnormal_vital(day)

        # Mood: how distressed the agent feels — drives urgency language
        _af = {Personality.STOIC: 0.35, Personality.NEUTRAL: 0.65, Personality.ANXIOUS: 1.15}
        mood = max(0.0, min(1.0, 1.0 - hs.severity * _af.get(self.personality, 0.65)))

        return InnerState(
            severity=hs.severity, symptoms=hs.symptoms,
            days_infected=day, trend=trend,
            fatigue=fatigue, pain=pain, mood=mood, top_vital=top_vital,
        )

    def _most_abnormal_vital(self, day: int) -> tuple | None:
        """Return (variable, value, band) for the most alarming vital today."""
        priority = ["temp", "RR", "SpO2", "HR"]
        for band_target in ("abnormal", "borderline"):
            for var in priority:
                b = self._case_table.band(var, day)
                if b == band_target:
                    return (var, self._case_table.get(var, day), b)
        return None

    def opening_statement(self) -> str:
        """Natural-language symptom report for the LLM doctor."""
        return self._narrator.full_opening_statement(
            self.inner_state,
            self.health_state.days_infected,
            self.personality,
        )

    def followup_answer(self, question: str) -> str:
        """Answer a doctor follow-up question in character."""
        day = self.health_state.days_infected
        return self._narrator.followup_answer(
            self.health_state.symptoms,
            self.personality,
            question,
            case_table=self._case_table,
            day=day,
        )

    def build_diagnostic_event(self) -> DiagnosticEvent:
        traj = self.health_state._trajectory
        day  = self.health_state.days_infected
        icd  = getattr(traj, "icd_code", "B99.9")

        # Management tier derived from disease-specific vital signs via NEWS2 scoring.
        # This makes the ground-truth label clinically grounded: pneumonia and sepsis
        # score high even at moderate internal severity; flu at the same severity may
        # only require home rest because SpO2 and BP are largely preserved.
        if self._case_table is not None:
            from simulation.case_table import management_from_vitals
            management = management_from_vitals(self._case_table, day)
        else:
            # Fallback: severity float threshold (agents infected before case table init)
            s = self.health_state.severity
            management = "hospitalise" if s >= 0.70 else "treat" if s >= 0.40 else "home rest"

        gt = f"{icd} / {management}"

        return DiagnosticEvent(
            agent_id      = self.id,
            severity      = self.health_state.severity,
            symptoms      = self.health_state.symptoms,
            days_infected = day,
            personality   = self.personality,
            case_table    = self._case_table,
            ground_truth  = gt,
        )

    @property
    def status(self) -> HealthStatus:
        return self.health_state.status

    @property
    def is_infectious(self) -> bool:
        """True only after the incubation period has elapsed."""
        if self.health_state.status != HealthStatus.INFECTED:
            return False
        traj = self.health_state._trajectory
        if traj is None:
            return False
        return self.health_state.days_infected > traj.incubation_days

    @property
    def display_char(self) -> str:
        return {
            HealthStatus.SUSCEPTIBLE: "·",
            HealthStatus.INFECTED:    "✶",
            HealthStatus.RECOVERING:  "○",
            HealthStatus.RECOVERED:   "·",
        }.get(self.status, "?")


# ─── SIRModel ─────────────────────────────────────────────────────────────────

class SIRModel:
    """
    Population-level observer (§3, §4.1).
    Recomputes S/I/R from individual HealthStates once per day.
    History is kept for plotting the curve in the TUI.
    """
    def __init__(self, beta: float = 0.3, gamma: float = 0.1):
        self.beta  = beta
        self.gamma = gamma
        self.S = 0
        self.I = 0
        self.R = 0
        self.history: list[tuple[int,int,int]] = []

    def step(self, agents: list[Agent]) -> None:
        self.S = sum(1 for a in agents if a.status == HealthStatus.SUSCEPTIBLE)
        self.I = sum(1 for a in agents if a.status == HealthStatus.INFECTED)
        self.R = sum(1 for a in agents
                     if a.status in (HealthStatus.RECOVERING, HealthStatus.RECOVERED))
        self.history.append((self.S, self.I, self.R))

    def infection_probability(self, density: int) -> float:
        """Tick-level risk estimate (used for display)."""
        return 1.0 - math.exp(-self.beta * density)

    @property
    def total(self) -> int:
        return self.S + self.I + self.R
