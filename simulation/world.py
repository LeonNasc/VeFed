"""
WorldEngine — central orchestrator (§4.1, §4.2).

Manages:
  - Discrete-time tick loop (288 ticks × 5 min = 24 h)
  - Agent spatial mobility via DailySchedule
  - Exposure calculation (eq. 2)
  - DiseaseCloud injection and decay
  - End-of-day SIR synchronisation and ClinicQueue population
  - WFC-inspired environment generation
"""
from __future__ import annotations
import random
import string
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO


from simulation.strategies import DiseaseStrategy, StandardFluStrategy
from simulation.progression import (
    DiseaseProgressionStrategy, StandardFluProgression,
    PROGRESSION_STRATEGIES,
)
from simulation.symptom_language import Personality
from simulation.models import (
    Agent, DailySchedule, DiseaseCloud,
    DiagnosticEvent, HealthStatus, Location, LocationType, SIRModel,
)

# ─── Wave Function Collapse stub ─────────────────────────────────────────────
# Full WFC is out of scope; we use a constraint-based random layout that
# mirrors the heterogeneity intent described in §2 / §5.

_LOC_TYPES = [
    LocationType.HOME,
    LocationType.WORK,
    LocationType.THIRD,
    LocationType.HOSPITAL,
]
_TYPE_WEIGHTS = [6, 3, 2, 1]   # homes most common


def _wfc_generate_locations(n_locations: int, rng: random.Random) -> list[Location]:
    """
    Simplified WFC: sample location types with weighted constraints,
    ensure at least 1 Hospital and 2 Work nodes.
    Returns a list of Location objects with (row, col) grid positions.
    """
    # Guarantee essential tiles
    types = [LocationType.HOSPITAL, LocationType.WORK, LocationType.WORK]
    remaining = n_locations - len(types)
    types += rng.choices(_LOC_TYPES, weights=_TYPE_WEIGHTS, k=remaining)
    rng.shuffle(types)

    cols = max(4, int(n_locations ** 0.5) + 1)
    locs = []
    for idx, lt in enumerate(types):
        row = idx // cols
        col = idx % cols
        loc = Location(
            loc_id   = f"{lt.value[:2].upper()}{idx:02d}",
            loc_type = lt,
            capacity = rng.randint(5, 30),
            row      = row,
            col      = col,
        )
        locs.append(loc)
    return locs


# ─── ClinicQueue ──────────────────────────────────────────────────────────────

class ClinicQueue:
    """
    Buffer between WorldEngine and the diagnostic layer (§4.2).
    Agents whose σ ≥ γ_a are enqueued; the DiagnosticClient processes them.
    """
    def __init__(self):
        import threading
        self.patients: deque[DiagnosticEvent] = deque()
        self.processed: list[DiagnosticEvent] = []  # outcome archive
        self.current_patient: str = ""   # name being processed right now
        self.current_status:  str = ""   # e.g. '3/10 done'
        self._status_callback = None    # set by TUI
        self._status_lock = threading.Lock()  # guards callback under concurrent use

    def set_status_callback(self, fn) -> None:
        """Register a zero-arg callable; called after each status update."""
        self._status_callback = fn

    def update_status(self, patient: str, status: str) -> None:
        with self._status_lock:
            self.current_patient = patient
            self.current_status  = status
            if self._status_callback:
                self._status_callback()

    def enqueue(self, event: DiagnosticEvent) -> None:
        self.patients.append(event)

    def dequeue(self) -> Optional[DiagnosticEvent]:
        return self.patients.popleft() if self.patients else None

    def process_all(self, diagnostic_fn,
                    max_workers: int = 4) -> list[DiagnosticEvent]:
        """
        Extension point → DiagnosticClient.
        diagnostic_fn(event) -> DiagnosticEvent with action + oracle_label filled.

        max_workers: concurrent Ollama requests in-flight. Python socket I/O
        releases the GIL, so real overlap happens even on one CPU. Tune to
        how many parallel slots your Ollama server allows
        (OLLAMA_NUM_PARALLEL env var, default 1 on CPU / 4 on GPU).
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        patients = list(self.patients)
        self.patients.clear()
        total    = len(patients)
        _done    = [0]
        _lock    = threading.Lock()

        def _process(ev: DiagnosticEvent) -> DiagnosticEvent:
            result = diagnostic_fn(ev)
            with _lock:
                _done[0] += 1
                self.update_status(
                    result.agent_id,
                    f"{_done[0]}/{total} done",
                )
            return result

        results: list[DiagnosticEvent] = []
        with ThreadPoolExecutor(max_workers=min(max_workers, total or 1)) as pool:
            futures = {pool.submit(_process, ev): ev for ev in patients}
            for fut in as_completed(futures):
                result = fut.result()
                results.append(result)
                self.processed.append(result)

        self.update_status("", "")
        return results

    def apply_outcomes(self, results: list[DiagnosticEvent],
                       agents_by_id: dict[str, Agent],
                       hospital_id: str) -> list[str]:
        """
        Translate DiagnosticAction decisions back into agent state (§4.2).
        Returns a log of applied actions.
        """
        from simulation.models import DiagnosticAction
        msgs = []
        for ev in results:
            agent = agents_by_id.get(ev.agent_id)
            if agent is None:
                continue
            action = ev.action
            if action == DiagnosticAction.HOSPITALISE:
                agent.hospitalised     = True
                agent.current_location = hospital_id
                agent.current_type     = LocationType.HOSPITAL
                agent.health_state.apply_treatment(decay_boost=0.20)
                msgs.append(f"{agent.name} → hospitalised")
            elif action == DiagnosticAction.RECOVER:
                msgs.append(f"{agent.name} → home recovery")
            elif action == DiagnosticAction.RESOLVE:
                agent.health_state.apply_treatment(decay_boost=0.15)
                msgs.append(f"{agent.name} → treatment plan assigned")
            else:
                msgs.append(f"{agent.name} → {action.value if action else 'assessed'}")
        return msgs


# ─── WorldEngine ─────────────────────────────────────────────────────────────

class WorldEngine:
    """
    Primary orchestrator (§4.2).

    State Ω = {agents, locations, current_tick, disease_clouds}
    Discrete-time Markov chain with T=288 ticks/day.
    """

    TICKS_PER_DAY = 288
    BASE_BETA     = 0.25   # baseline transmission rate (scaled by strategy)

    def __init__(self, num_agents: int = 30, seed: int = 0,
                 disease_strategy: DiseaseStrategy | None = None,
                 progression_strategy: DiseaseProgressionStrategy | None = None):
        self._seed   = seed
        self._rng    = random.Random(seed)
        self.strategy: DiseaseStrategy = disease_strategy or StandardFluStrategy()
        self.progression: DiseaseProgressionStrategy = (
            progression_strategy or StandardFluProgression()
        )

        # ── Generate world via WFC ──────────────────────────────────────
        n_locs       = max(10, num_agents // 3)
        self.locations: dict[str, Location] = {}
        for loc in _wfc_generate_locations(n_locs, self._rng):
            self.locations[loc.id] = loc

        # Virtual commute node (no physical tile)
        commute = Location("commute", LocationType.COMMUTING, capacity=999)
        self.locations["commute"] = commute

        # Categorise by type for schedule assignment
        homes     = [l for l in self.locations.values() if l.type == LocationType.HOME]
        works     = [l for l in self.locations.values() if l.type == LocationType.WORK]
        thirds    = [l for l in self.locations.values() if l.type == LocationType.THIRD]
        hospitals = [l for l in self.locations.values() if l.type == LocationType.HOSPITAL]

        self._hospital_id = hospitals[0].id if hospitals else list(self.locations.keys())[0]

        # ── Spawn agents ────────────────────────────────────────────────
        names       = self._generate_names(num_agents)
        self.agents: list[Agent] = []
        self.agents_by_id: dict[str, Agent] = {}
        for i in range(num_agents):
            h = self._rng.choice(homes)
            w = self._rng.choice(works)
            t = self._rng.choice(thirds) if thirds else h
            sched     = DailySchedule(home_id=h.id, work_id=w.id, third_id=t.id)
            threshold = self._rng.uniform(0.15, 0.65)   # γ_a heterogeneity
            personality = self._rng.choice(list(Personality))
            agent     = Agent(f"A{i:03d}", names[i], sched, threshold, self._rng, personality)
            agent.move_to(h.id, LocationType.HOME)
            h.enter(agent.id)
            self.agents.append(agent)
            self.agents_by_id[agent.id] = agent

        # ── Seed initial infections ─────────────────────────────────────
        patient_zero = self._rng.choice(self.agents)
        from simulation.case_table import CaseTable
        _traj = self.progression.sample_trajectory(self._rng)
        patient_zero.health_state.infect(_traj)
        if hasattr(_traj, "_record"):
            from simulation.case_table import MimicCaseTable
            patient_zero._case_table = MimicCaseTable(_traj._record, self._rng)
        else:
            patient_zero._case_table = CaseTable(_traj, self._rng, max_days=90)

        # ── Subsystems ──────────────────────────────────────────────────
        self.current_tick: int         = 0
        self.current_day:  int         = 0
        self.disease_clouds: list[DiseaseCloud] = []
        self.clinic_queue  = ClinicQueue()
        self.sir_model     = SIRModel(beta=self.BASE_BETA)
        self.event_log:    list[str]   = []
        self.paused:       bool        = False

        # FL extension points
        self.fl_round:     int         = 0
        self.fl_log:       list[str]   = []

        # Daily log file — appended at end of each simulated day
        from pathlib import Path
        log_path = Path(f"sim_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        self._log_file = log_path.open("w", encoding="utf-8", buffering=1)
        self._log_path = log_path
        self._log_file.write(
            f"# Federated Simulated World — started {datetime.now().isoformat()}\n"
            f"# agents={num_agents}  seed={seed}\n\n"
        )

        # Inject an initial disease cloud at patient zero's location
        self.inject_disease_cloud(patient_zero.current_location, intensity=0.4)

        # Sync initial SIR
        self.sir_model.step(self.agents)

    # ── Public API ───────────────────────────────────────────────────────────

    def step_tick(self) -> None:
        """
        Execute one 5-minute simulation update (§4.2):
        1. Move agents per mobility policy
        2. Calculate exposure per agent (eq. 2)
        3. Stochastically generate new disease clouds
        4. Decay existing clouds
        """
        t = self.current_tick

        for loc in self.locations.values():
            loc.agents_present.clear()

        for agent in self.agents:
            if agent.hospitalised:
                agent.move_to(self._hospital_id, LocationType.HOSPITAL)
                self.locations[self._hospital_id].enter(agent.id)
                continue
            loc_id, loc_type = agent.schedule.location_for_tick(t)
            agent.move_to(loc_id, loc_type)
            if loc_id in self.locations:
                self.locations[loc_id].enter(agent.id)

        # Compute exposure for every susceptible agent (eq. 2)
        for agent in self.agents:
            if agent.status != HealthStatus.SUSCEPTIBLE:
                continue
            loc = self.locations.get(agent.current_location)
            if loc is None:
                continue
            # (β · Σ I(a' ∈ I_t) + ω(l_t)) / T  — divide by ticks/day so
            # BASE_BETA is a daily rate and cumulative_exposure ≈ β at day's end.
            n_infectious = sum(
                1 for aid in loc.agents_present
                if self.agents_by_id[aid].status == HealthStatus.INFECTED
            )
            beta = self.BASE_BETA * self.strategy.transmission_rate(agent.current_type)
            eps = (beta * n_infectious + loc.ambient_exposure()) / self.TICKS_PER_DAY
            agent.update_health(eps)

        # Random disease-cloud generation (stochastic events)
        if self._rng.random() < 0.003:
            loc = self._rng.choice(list(self.locations.values()))
            self.inject_disease_cloud(loc.id, self._rng.uniform(0.1, 0.4))
            self.event_log.append(
                f"Day {self.current_day} T{t:03d}: Cloud at {loc.id}"
            )

        for loc in self.locations.values():
            loc.tick_clouds()

        self.current_tick += 1

        # ── End of day ───────────────────────────────────────────────────
        if self.current_tick >= self.TICKS_PER_DAY:
            self._end_of_day()
            # Process clinic queue separately so UI updates between
            self._process_clinic_queue()

    def inject_disease_cloud(self, location_id: str, intensity: float = 0.5) -> None:
        """Exogenous viral source injection (§4.2)."""
        if location_id not in self.locations:
            return
        cloud = DiseaseCloud(location_id=location_id, intensity=intensity)
        self.locations[location_id].add_cloud(cloud)
        self.disease_clouds.append(cloud)
        self.disease_clouds = [c for c in self.disease_clouds if c.active]

    def collect_end_of_day_cases(self) -> list[DiagnosticEvent]:
        """Returns symptomatic agents exceeding their care-seeking threshold."""
        cases = []
        for agent in self.agents:
            if agent.should_seek_care():
                cases.append(agent.build_diagnostic_event())
        return cases

    # ── Extension point: Diagnostic / FL hooks ───────────────────────────────

    def register_diagnostic_fn(self, fn) -> None:
        """
        Extension point §3.3 / §3.4.
        fn(DiagnosticEvent) -> DiagnosticEvent with .action and .oracle_label set.
        Defaults to built-in stub when not registered.
        """
        self._diagnostic_fn = fn

    def run_fl_round(self) -> None:
        """
        Extension point §3.3 eq. 6–7.
        Stub simulates FedAvg aggregation step.
        Replace body with real flwr-llm orchestration.
        """
        self.fl_round += 1
        n_cases = len(self.clinic_queue.processed)
        self.fl_log.append(
            f"Round {self.fl_round}: aggregated {n_cases} cases "
            f"(S={self.sir_model.S} I={self.sir_model.I} R={self.sir_model.R})"
        )

    def set_disease_strategy(self, strategy: DiseaseStrategy) -> None:
        """Hot-swap the transmission strategy at runtime."""
        self.strategy = strategy
        self.event_log.append(
            f"Day {self.current_day}: Transmission → {strategy.name}"
        )

    def set_progression_strategy(
        self, strategy: DiseaseProgressionStrategy
    ) -> None:
        """
        Hot-swap the progression strategy at runtime.
        Already-infected agents keep their current trajectory;
        new infections will use the new strategy.
        """
        self.progression = strategy
        self.event_log.append(
            f"Day {self.current_day}: Progression → {strategy.name}"
        )

    def attach_interaction_logger(self, logger) -> None:
        """
        Register an InteractionLogger.  Every processed DiagnosticEvent will
        be serialised to the logger's JSONL file automatically.
        """
        self._interaction_logger = logger

    # ── Private helpers ──────────────────────────────────────────────────────

    def _end_of_day(self) -> None:
        """
        Day boundary — split into fast world update + slow clinic processing.
        World update happens every day. Clinic processing is deferred if queue > 0.
        """
        self.current_tick = 0
        self.current_day += 1

        # Stochastic S→I transitions for accumulated daily exposure (eq. 3)
        newly_infected = []
        for agent in self.agents:
            if agent.apply_daily_infection(self.progression):
                newly_infected.append(agent.name)
            agent.reset_daily_exposure()

        if newly_infected:
            self.event_log.append(
                f"Day {self.current_day}: Infected → {', '.join(newly_infected)}"
            )

        # Progress internal health states (eq. 4/5)
        for agent in self.agents:
            agent.health_state.step()
            if agent.hospitalised and agent.status in (
                HealthStatus.RECOVERING, HealthStatus.RECOVERED
            ):
                agent.hospitalised = False
                self.event_log.append(f"Day {self.current_day}: {agent.name} discharged")

        # SIR observer — do this BEFORE clinic so UI shows updated counts
        self.sir_model.step(self.agents)

        # Collect symptomatic cases into queue
        cases = self.collect_end_of_day_cases()
        for c in cases:
            ag = self.agents_by_id.get(c.agent_id)
            if ag:
                c.conversation.append({
                    'role': 'patient',
                    'text': ag.opening_statement(),
                })
            self.clinic_queue.enqueue(c)

        # FL round every 7 days
        if self.current_day % 7 == 0:
            self.run_fl_round()

        # Log file daily header (clinic details written after processing)
        sir = self.sir_model
        self._log_file.write(
            f"=== Day {self.current_day:04d} | "
            f"S={sir.S:3d} I={sir.I:3d} R={sir.R:3d} | "
            f"FL round={self.fl_round}\n"
        )
        day_entries = [e for e in self.event_log if f"Day {self.current_day}:" in e]
        for entry in day_entries:
            self._log_file.write(f"  {entry}\n")

        # Keep log bounded
        if len(self.event_log) > 200:
            self.event_log = self.event_log[-200:]

    def _process_clinic_queue(self) -> None:
        """
        Process the clinic queue (slow LLM calls).
        Called separately from _end_of_day so UI can update in between.
        """
        if not self.clinic_queue.patients:
            return

        diag_fn = getattr(self, "_diagnostic_fn", _default_diagnostic)
        results = self.clinic_queue.process_all(diag_fn)
        msgs    = self.clinic_queue.apply_outcomes(
            results, self.agents_by_id, self._hospital_id
        )
        self.event_log.extend(msgs)

        logger = getattr(self, "_interaction_logger", None)
        if logger is not None:
            for ev in results:
                logger.log(ev, day=self.current_day, seed=self._seed)

        # Write clinic details to log file
        clinic_today = [ev for ev in results if ev.oracle_label is not None]
        for ev in clinic_today[-20:]:
            ag = self.agents_by_id.get(ev.agent_id)
            name = ag.name if ag else ev.agent_id
            ptrait = ev.personality.value if ev.personality else "?"
            self._log_file.write(
                f"  clinic | {name:<20} [{ptrait:<7}] "
                f"action={ev.action.value if ev.action else '?':<13} "
                f"label={ev.oracle_label}\n"
            )
            self._log_file.write(
                f"    GT: {ev.ground_truth or 'unknown'}  "
                f"LLM: {ev.diagnosis or 'none'}\n"
            )
            for turn in ev.conversation:
                role = "P" if turn["role"] == "patient" else "D"
                self._log_file.write(f"    {role}: {turn['text']}\n")
            self._log_file.write("\n")
        self._log_file.write("\n")

    def _generate_names(self, n: int) -> list[str]:
        first = ["Aivar","Andres","Anni","Dmitri","Ene","Erik","Hannes","Helgi",
                 "Indrek","Jaan","Kadri","Kalev","Katrin","Kristel","Lauri","Lembit",
                 "Liina","Maie","Mare","Margit","Mati","Moonika","Peeter","Piret",
                 "Raivo","Reet","Siiri","Tarvo","Tiina","Toomas"]
        last  = ["Kask","Tamm","Mägi","Sepp","Lepp","Pärn","Oja","Rand",
                 "Kallas","Saar","Leppik","Kukk","Mets","Vaher","Rebane"]
        self._rng.shuffle(first)
        return [f"{first[i % len(first)]} {self._rng.choice(last)}" for i in range(n)]


# ─── Default diagnostic stub (extension point placeholder) ───────────────────

def _default_diagnostic(event: DiagnosticEvent) -> DiagnosticEvent:
    """
    Stub oracle — mimics a simple rule-based triage until a real LLM is wired in.
    Replace via world.register_diagnostic_fn(my_llm_fn).

    Decision rules (simplistic stand-in for MedAlpaca / fine-tuned model):
      severity ≥ 0.7  → HOSPITALISE
      severity ≥ 0.4  → RESOLVE (treatment plan)
      else            → RECOVER (home rest)
    """
    from simulation.models import DiagnosticAction
    s = event.severity

    if s >= 0.70:
        event.action       = DiagnosticAction.HOSPITALISE
        event.oracle_label = "severe"
        event.notes        = "High severity — hospital admission required."
    elif s >= 0.40:
        event.action       = DiagnosticAction.RESOLVE
        event.oracle_label = "moderate"
        event.notes        = "Moderate — antiviral prescribed, monitor symptoms."
    else:
        event.action       = DiagnosticAction.RECOVER
        event.oracle_label = "mild"
        event.notes        = "Mild — rest at home, stay hydrated."
    return event
