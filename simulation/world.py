"""
WorldEngine — pure simulation orchestrator (§4.1, §4.2).

Manages:
  - Discrete-time tick loop (288 ticks × 5 min = 24 h)
  - Agent spatial mobility via DailySchedule
  - Exposure calculation (eq. 2)
  - DiseaseCloud injection and decay
  - End-of-day SIR synchronisation and ClinicQueue population
  - WFC-inspired environment generation

FL training, end conditions, and learner weights are NOT part of WorldEngine.
They live in fl/silo.py (FLSilo), which wraps this class.
"""
from __future__ import annotations
import random
import string
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

from simulation.strategies import DiseaseStrategy, StandardFluStrategy, STRATEGIES
from simulation.progression import (
    DiseaseProgressionStrategy, InfluenzaProgression,
    PROGRESSION_STRATEGIES,
)
from simulation.symptom_language import Personality
from simulation.models import (
    Agent, DailySchedule, DiseaseCloud,
    DiagnosticEvent, HealthStatus, Location, LocationType, SIRModel,
    REFRACTORY_DAYS, severity_bucket,
)
from simulation.world_config import WorldConfig, AgentConfig, EpidemicConfig

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


def _wfc_generate_locations(n_locations: int, rng: random.Random,
                             n_hospitals: int = 1) -> list[Location]:
    """
    Simplified WFC: sample location types with weighted constraints,
    ensure exactly n_hospitals Hospital tiles and 2 Work nodes.
    Returns a list of Location objects with (row, col) grid positions.
    """
    types = [LocationType.HOSPITAL] * n_hospitals + [LocationType.WORK, LocationType.WORK]
    remaining = n_locations - len(types)
    if remaining > 0:
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
                agent._care_cooldown   = 7   # review in a week
                msgs.append(f"{agent.name} → hospitalised")
            elif action == DiagnosticAction.RECOVER:
                agent._care_cooldown   = 5   # rest at home, return if worse
                msgs.append(f"{agent.name} → home recovery")
            elif action == DiagnosticAction.RESOLVE:
                # Small boost — management, not cure; preserves chronic trajectories
                agent.health_state.apply_treatment(decay_boost=0.003)
                agent._care_cooldown   = 7   # follow-up in a week
                msgs.append(f"{agent.name} → treatment plan assigned")
            else:
                agent._care_cooldown   = 3
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
    BASE_BETA     = 1.50   # baseline transmission rate (scaled by strategy)

    # Meaningful close contacts per day per location type.
    # ~2.5× POLYMOD community baseline (Mossong et al. 2008) — scaled up for a
    # closed simulated population where the same agents interact daily, giving
    # higher within-group secondary attack rates than community-level surveys
    # capture. Targets R0 ≈ 1.5 for a self-sustaining epidemic arc.
    DAILY_CONTACTS = {
        LocationType.HOME:      17,
        LocationType.WORK:      27,
        LocationType.THIRD:     33,
        LocationType.COMMUTING: 11,
        LocationType.HOSPITAL:  13,
    }

    # Expected ticks per day an agent spends at each location type (from DailySchedule).
    # Used to normalise the per-tick exposure so that daily total = beta * prevalence
    # * daily_contacts / TICKS_PER_DAY regardless of how many ticks are spent there.
    _TICKS_AT_LOC = {
        LocationType.HOME:      120,   # 0-71 (sleep) + 240-287 (evening)
        LocationType.WORK:       84,   # 96-179
        LocationType.THIRD:      36,   # 204-239
        LocationType.COMMUTING:  48,   # 72-95 + 180-203 (2 h each way)
        LocationType.HOSPITAL:  288,   # full day
    }

    # During commute, agents randomly encounter this many strangers from the full
    # commute pool — simulates a bus/metro carriage rather than a fixed social group.
    # Random sampling provides cross-household/-workplace mixing that helps the
    # epidemic bridge clusters.
    COMMUTE_SAMPLE_SIZE = 25

    def __init__(
        self,
        config: WorldConfig,
        seed: int = 0,
        # Shared-world extension: multiple hospital districts in one world.
        # Each district maps to one FL silo.  Kept here because it governs
        # how agents are spatially assigned, not how FL trains on them.
        n_hospitals: int = 1,
        hospital_disease_weights: list[list[float]] | None = None,
    ):
        self._config = config
        epi   = config.epidemic
        agents_cfg = config.agents

        self._seed = seed + config.seed_offset
        self._rng  = random.Random(self._seed)

        self.strategy: DiseaseStrategy = (
            STRATEGIES.get(epi.disease_strategy, StandardFluStrategy())
            if hasattr(epi, "disease_strategy") else StandardFluStrategy()
        )
        self._beta_scale           = epi.beta_scale
        self._contact_rate_sigma   = epi.contact_rate_sigma
        self._disease_weights      = epi.disease_weights
        self.reveal_incubating_icd = config.reveal_incubating_icd
        self.background_visit_rate = agents_cfg.background_visit_rate

        # Resolve progression strategies from names
        self.progressions: list[DiseaseProgressionStrategy] = [
            PROGRESSION_STRATEGIES[p] for p in epi.progressions
            if p in PROGRESSION_STRATEGIES
        ] or [InfluenzaProgression()]
        self.progression = self.progressions[0]  # backward-compat alias

        # ── Generate world via WFC ──────────────────────────────────────
        n_agents = agents_cfg.num_agents
        n_locs   = max(10 + n_hospitals - 1, n_agents // 3)
        self.locations: dict[str, Location] = {}
        for loc in _wfc_generate_locations(n_locs, self._rng, n_hospitals=n_hospitals):
            self.locations[loc.id] = loc

        commute = Location("commute", LocationType.COMMUTING, capacity=999)
        self.locations["commute"] = commute

        homes     = [l for l in self.locations.values() if l.type == LocationType.HOME]
        works     = [l for l in self.locations.values() if l.type == LocationType.WORK]
        thirds    = [l for l in self.locations.values() if l.type == LocationType.THIRD]
        hospitals = [l for l in self.locations.values() if l.type == LocationType.HOSPITAL]

        self._hospital_ids: list[str] = [h.id for h in hospitals] or [list(self.locations.keys())[0]]
        self._hospital_id = self._hospital_ids[0]

        # ── Spawn agents ────────────────────────────────────────────────
        names = self._generate_names(n_agents)
        self.agents: list[Agent] = []
        self.agents_by_id: dict[str, Agent] = {}
        self._agent_disease_weights: dict[str, list[float]] = {}
        n_hosp = len(self._hospital_ids)
        for i in range(n_agents):
            h = self._rng.choice(homes)
            w = self._rng.choice(works)
            t = self._rng.choice(thirds) if thirds else h
            sched       = DailySchedule(home_id=h.id, work_id=w.id, third_id=t.id)
            threshold   = self._rng.uniform(0.15, 0.65)
            personality = self._rng.choice(list(Personality))
            if self._contact_rate_sigma > 0:
                sigma       = self._contact_rate_sigma
                sociability = self._rng.lognormvariate(-(sigma ** 2) / 2, sigma)
            else:
                sociability = 1.0
            agent    = Agent(f"A{i:03d}", names[i], sched, threshold, self._rng,
                             personality, sociability)
            hosp_idx = i % n_hosp
            agent.home_hospital_id = self._hospital_ids[hosp_idx]
            if hospital_disease_weights is not None and hosp_idx < len(hospital_disease_weights):
                self._agent_disease_weights[agent.id] = hospital_disease_weights[hosp_idx]
            agent.move_to(h.id, LocationType.HOME)
            h.enter(agent.id)
            self.agents.append(agent)
            self.agents_by_id[agent.id] = agent

        # ── Seed initial infections ─────────────────────────────────────
        from simulation.case_table import CaseTable
        seed_agents = self._rng.sample(self.agents, min(epi.initial_seeds, len(self.agents)))
        for seed_agent in seed_agents:
            _prog = self._rng.choices(self.progressions, weights=self._disease_weights)[0]
            _traj = _prog.sample_trajectory(self._rng)
            seed_agent.health_state.infect(_traj)
            if hasattr(_traj, "_record"):
                from simulation.case_table import MimicCaseTable
                seed_agent._case_table = MimicCaseTable(_traj._record, self._rng)
            else:
                seed_agent._case_table = CaseTable(_traj, self._rng, max_days=90)

        # ── Subsystems ──────────────────────────────────────────────────
        self.current_tick:   int  = 0
        self.current_day:    int  = 0
        self.disease_clouds: list[DiseaseCloud] = []
        self.clinic_queues:  list[ClinicQueue]  = [ClinicQueue() for _ in self._hospital_ids]
        self._hosp_queue_map: dict[str, ClinicQueue] = {
            hid: q for hid, q in zip(self._hospital_ids, self.clinic_queues)
        }
        self.clinic_queue = self.clinic_queues[0]
        self.sir_model    = SIRModel(beta=self.BASE_BETA)
        self.event_log:   list[str] = []
        self.paused:      bool      = False

        # Daily log file
        log_dir  = Path(__file__).resolve().parent.parent / "sim_logs"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"sim_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self._log_file = log_path.open("w", encoding="utf-8", buffering=1)
        self._log_file.write(
            f"# WorldEngine — started {datetime.now().isoformat()}\n"
            f"# agents={n_agents}  seed={self._seed}\n\n"
        )

        self.inject_disease_cloud(seed_agents[0].current_location, intensity=0.4)
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
            # Frequency-dependent force of infection (eq. 2):
            #   eps = (β · I_loc/N_loc + ω) / T
            # Dividing by location occupancy makes exposure scale with local
            # prevalence rather than absolute infectious count — prevents
            # explosive spread in small bounded populations where density-
            # dependent β diverges from the intended population-level rate.
            # Only post-incubation agents count as infectious.
            n_present    = len(loc.agents_present)
            n_infectious = sum(
                1 for aid in loc.agents_present
                if self.agents_by_id[aid].is_infectious
            )
            if agent.current_type == LocationType.COMMUTING:
                # Random encounter model: sample a small group from the commute pool
                # rather than using all present — simulates a metro/bus carriage and
                # provides cross-household mixing that bridges household clusters.
                pool = [aid for aid in loc.agents_present if aid != agent.id]
                k    = min(self.COMMUTE_SAMPLE_SIZE, len(pool))
                if k > 0:
                    sample      = self._rng.sample(pool, k)
                    n_inf_samp  = sum(1 for aid in sample
                                     if self.agents_by_id[aid].is_infectious)
                    prevalence  = n_inf_samp / k
                else:
                    prevalence = 0.0
            else:
                prevalence = n_infectious / n_present if n_present > 0 else 0.0

            beta           = self.BASE_BETA * self._beta_scale * self.strategy.transmission_rate(agent.current_type)
            daily_contacts = self.DAILY_CONTACTS.get(agent.current_type, 8) * agent.sociability
            ticks_at_loc   = self._TICKS_AT_LOC.get(agent.current_type, 72)
            # Per-tick contribution normalised so daily total = beta * prevalence
            # * daily_contacts / TICKS_PER_DAY (independent of ticks spent here).
            eps = (beta * prevalence * daily_contacts / ticks_at_loc
                   + loc.ambient_exposure()) / self.TICKS_PER_DAY
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
                ev = agent.build_diagnostic_event()
                # Q26: mask real ICD for incubating patients when reveal_incubating_icd=False
                if not self.reveal_incubating_icd and ev.severity == 0.0 and ev.days_infected > 0:
                    ev.ground_truth = "unknown/mild"
                    ev.gt_disease   = "unknown"
                    ev.gt_severity  = "mild"
                cases.append(ev)
        return cases

    # ── Epidemic state ───────────────────────────────────────────────────────

    @property
    def epidemic_active(self) -> bool:
        """True while any agent is infected or infectious."""
        return self.sir_model.I > 0

    # ── Primary data-generation interface ────────────────────────────────────

    def run_sim_days(self, n: int) -> list[DiagnosticEvent]:
        """
        Advance the simulation by n days and return all DiagnosticEvents
        produced during that period.

        This is the only method FLSilo (or any orchestrator) needs to call.
        It dispatches to _run_static_days when the epidemic config sets
        static_mode=True; otherwise runs the full SIR tick loop.
        """
        if self._config.epidemic.static_mode:
            return self._run_static_days(n)
        return self._run_sir_days(n)

    def _run_sir_days(self, n: int) -> list[DiagnosticEvent]:
        """Run n days of SIR simulation; return events processed by the clinic."""
        start = len(self.clinic_queue.processed)
        for _ in range(n * self.TICKS_PER_DAY):
            self.step_tick()
        events = self.clinic_queue.processed[start:]
        del self.clinic_queue.processed[:start]
        return events

    def _run_static_days(self, n: int) -> list[DiagnosticEvent]:
        """
        Static-mode event generation — no SIR, no agents.

        Produces exactly cases_per_day × n DiagnosticEvents per call.
        infectious_fraction controls the disease/background split.
        """
        from simulation.case_table import CaseTable, HealthyCaseTable
        from simulation.complaints import NON_INFECTIOUS_COMPLAINTS, OTHER_DISEASE_COMPLAINTS
        BACKGROUND_COMPLAINTS = NON_INFECTIOUS_COMPLAINTS + OTHER_DISEASE_COMPLAINTS
        epi   = self._config.epidemic
        total = epi.cases_per_day * n
        events: list[DiagnosticEvent] = []

        for k in range(total):
            personality = self._rng.choice(list(Personality))
            if self._rng.random() < epi.infectious_fraction:
                ev = self._make_static_infectious_event(k, personality)
            else:
                ev = self._make_static_background_event(k, personality, BACKGROUND_COMPLAINTS)
            opening = self._generate_opening_for_event(ev, personality)
            ev.conversation.insert(0, {"role": "patient", "text": opening})
            diag_fn = getattr(self, "_diagnostic_fn", _default_diagnostic)
            ev = diag_fn(ev)
            events.append(ev)

        self.current_day += n
        return events

    def _make_static_infectious_event(self, idx: int, personality: Personality) -> DiagnosticEvent:
        """Build one synthetic infectious DiagnosticEvent for static mode."""
        from simulation.case_table import CaseTable
        epi  = self._config.epidemic
        prog = self._rng.choices(self.progressions, weights=self._disease_weights)[0]
        traj = prog.sample_trajectory(self._rng)
        ct   = CaseTable(traj, self._rng, max_days=30)
        incubation = int(getattr(traj, "incubation_days", 2))
        rise       = int(getattr(traj, "rise_days",       3))
        day  = self._rng.randint(incubation + 1,
                                 max(incubation + 2, incubation + rise + 3))
        sev          = ct._severity_at_day(day)
        disease_name = getattr(traj, "disease_name", "unknown")
        return DiagnosticEvent(
            agent_id      = f"static_{idx}",
            severity      = sev,
            symptoms      = max(0.0, sev * 0.8 + self._rng.gauss(0, 0.08)),
            days_infected = day,
            personality   = personality,
            case_table    = ct,
            ground_truth  = f"{disease_name}/{severity_bucket(sev)}",
            gt_disease    = disease_name,
            gt_severity   = severity_bucket(sev),
            is_background = False,
        )

    def _make_static_background_event(self, idx: int, personality: Personality,
                                      complaints: list) -> DiagnosticEvent:
        """Build one synthetic background (non-infectious) DiagnosticEvent."""
        from simulation.case_table import HealthyCaseTable
        complaint    = self._rng.choice(complaints)
        gt_disease   = complaint.disease
        ground_truth = (gt_disease if gt_disease == "non-infectious"
                        else f"{gt_disease}/discharge")
        return DiagnosticEvent(
            agent_id          = f"bg_{idx}",
            severity          = 0.0,
            symptoms          = self._rng.uniform(0.0, 0.10),
            days_infected     = 0,
            personality       = personality,
            case_table        = HealthyCaseTable(self._rng),
            ground_truth      = ground_truth,
            gt_disease        = gt_disease,
            gt_severity       = "discharge",
            is_background     = True,
            complaint_context = complaint.prompt_context,
        )

    # ── Extension point: diagnostic hook ─────────────────────────────────────

    def register_diagnostic_fn(self, fn) -> None:
        """
        Extension point §3.3 / §3.4.
        fn(DiagnosticEvent) -> DiagnosticEvent with .action and .oracle_label set.
        Defaults to built-in stub when not registered.
        """
        self._diagnostic_fn = fn

    def attach_interaction_logger(self, logger) -> None:
        """Register an InteractionLogger for JSONL audit logging."""
        self._interaction_logger = logger

    def set_disease_strategy(self, strategy: DiseaseStrategy) -> None:
        """Hot-swap the transmission strategy at runtime."""
        self.strategy = strategy
        self.event_log.append(f"Day {self.current_day}: Transmission → {strategy.name}")

    def set_progression_strategy(self, strategy: DiseaseProgressionStrategy) -> None:
        """Hot-swap progression; already-infected agents keep their current trajectory."""
        self.progression = strategy
        self.event_log.append(f"Day {self.current_day}: Progression → {strategy.name}")

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
            dw = self._agent_disease_weights.get(agent.id, self._disease_weights)
            if agent.apply_daily_infection(self.progressions, dw):
                newly_infected.append(agent.name)
            agent.reset_daily_exposure()

        if newly_infected:
            self.event_log.append(
                f"Day {self.current_day}: Infected → {', '.join(newly_infected)}"
            )

        # Progress internal health states (eq. 4/5) + refractory period handling
        for agent in self.agents:
            agent.health_state.step()

            curr = agent.health_state.status
            prev = agent._prev_status

            # Detect RECOVERING → RECOVERED: start refractory countdown
            if prev == HealthStatus.RECOVERING and curr == HealthStatus.RECOVERED:
                agent._refractory_days = REFRACTORY_DAYS
            # Count down refractory; release back to SUSCEPTIBLE when done
            elif curr == HealthStatus.RECOVERED and agent._refractory_days > 0:
                agent._refractory_days -= 1
                if agent._refractory_days == 0:
                    agent.health_state.status = HealthStatus.SUSCEPTIBLE
                    curr = HealthStatus.SUSCEPTIBLE
            agent._prev_status = curr

            if agent.hospitalised and agent.status in (
                HealthStatus.RECOVERING, HealthStatus.RECOVERED
            ):
                agent.hospitalised = False
                self.event_log.append(f"Day {self.current_day}: {agent.name} discharged")

        # SIR observer — do this BEFORE clinic so UI shows updated counts
        self.sir_model.step(self.agents)

        # Collect symptomatic + incubating cases into queue
        cases = self.collect_end_of_day_cases()
        for c in cases:
            ag = self.agents_by_id.get(c.agent_id)
            if ag:
                c.conversation.append({
                    'role':  'patient',
                    'text':  self._generate_opening(ag, event=c),
                })
            q = self._hosp_queue_map.get(
                ag.home_hospital_id if ag else "", self.clinic_queue
            )
            q.enqueue(c)

        # Collect non-infectious background visitors
        for c in self._collect_background_cases():
            ag = self.agents_by_id.get(c.agent_id)
            if ag:
                c.conversation.append({
                    'role':  'patient',
                    'text':  self._generate_opening(ag, event=c),
                })
            q = self._hosp_queue_map.get(
                ag.home_hospital_id if ag else "", self.clinic_queue
            )
            q.enqueue(c)

        # Log file daily header (clinic details written after processing)
        sir = self.sir_model
        self._log_file.write(
            f"=== Day {self.current_day:04d} | S={sir.S:3d} I={sir.I:3d} R={sir.R:3d}\n"
        )
        day_entries = [e for e in self.event_log if f"Day {self.current_day}:" in e]
        for entry in day_entries:
            self._log_file.write(f"  {entry}\n")

        # Keep log bounded
        if len(self.event_log) > 200:
            self.event_log = self.event_log[-200:]

    def _generate_opening_for_event(self, event: DiagnosticEvent,
                                    personality: Personality) -> str:
        """Opening statement for a static-mode event (no live agent)."""
        from simulation.symptom_language import background_opening
        if event.is_background:
            return background_opening(personality, self._rng)
        from simulation.models import InnerState
        inner = InnerState(
            severity      = event.severity,
            symptoms      = event.symptoms,
            days_infected = event.days_infected,
            trend         = "stable",
            fatigue       = 5.0,
            pain          = 4.0,
            mood          = 0.5,
            top_vital     = None,
            disease_name  = event.gt_disease or "influenza",
        )
        return self._config.agents.data_source.opening_statement(inner, event.days_infected, personality)

    def _collect_background_cases(self) -> list[DiagnosticEvent]:
        """
        Non-infectious clinic visitors: susceptible or recovered agents presenting
        with a structured non-infectious complaint (back pain, headache, anxiety,
        hypertension follow-up, or fatigue) rather than an active infection.
        Rate is configurable via background_visit_rate (default 2.5 %/agent/day).
        """
        from simulation.complaints import NON_INFECTIOUS_COMPLAINTS, OTHER_DISEASE_COMPLAINTS
        BACKGROUND_COMPLAINTS = NON_INFECTIOUS_COMPLAINTS + OTHER_DISEASE_COMPLAINTS
        from simulation.case_table import HealthyCaseTable
        cases = []
        for agent in self.agents:
            if agent.status not in (HealthStatus.SUSCEPTIBLE, HealthStatus.RECOVERED):
                continue
            if agent.hospitalised or agent._care_cooldown > 0:
                continue
            if self._rng.random() < self.background_visit_rate:
                complaint = self._rng.choice(BACKGROUND_COMPLAINTS)
                gt_disease = complaint.disease
                ground_truth = (gt_disease if gt_disease == "non-infectious"
                                else f"{gt_disease}/discharge")
                ev = DiagnosticEvent(
                    agent_id          = agent.id,
                    severity          = 0.0,
                    symptoms          = self._rng.uniform(0.0, 0.10),
                    days_infected     = 0,
                    personality       = agent.personality,
                    case_table        = HealthyCaseTable(self._rng),
                    ground_truth      = ground_truth,
                    gt_disease        = gt_disease,
                    gt_severity       = "discharge",
                    is_background     = True,
                    complaint_context = complaint.prompt_context,
                )
                cases.append(ev)
        return cases

    def _generate_opening(self, agent, event: DiagnosticEvent | None = None) -> str:
        """Opening patient complaint for the SIR clinic queue."""
        from simulation.symptom_language import background_opening
        if event is not None and event.complaint_context:
            return background_opening(agent.personality, self._rng)
        return self._config.agents.data_source.opening_statement(
            agent.inner_state, agent.health_state.days_infected, agent.personality
        )

    def _process_clinic_queue(self) -> None:
        """
        Process all hospital clinic queues (slow LLM calls).
        Called separately from _end_of_day so UI can update in between.
        Single-hospital worlds have one queue; shared-world has N.
        """
        diag_fn = getattr(self, "_diagnostic_fn", _default_diagnostic)
        logger  = getattr(self, "_interaction_logger", None)
        all_results: list = []

        for hid, q in self._hosp_queue_map.items():
            if not q.patients:
                continue
            results = q.process_all(diag_fn)
            msgs    = q.apply_outcomes(results, self.agents_by_id, hid)
            self.event_log.extend(msgs)
            all_results.extend(results)

            if logger is not None:
                for ev in results:
                    logger.log(ev, day=self.current_day, seed=self._seed)

        # Write clinic details to log file
        clinic_today = [ev for ev in all_results if ev.oracle_label is not None]
        for ev in clinic_today[-20:]:
            ag = self.agents_by_id.get(ev.agent_id)
            name = ag.name if ag else ev.agent_id
            ptrait = ev.personality.value if ev.personality else "?"
            if ev.is_background:
                tag = "[bg]   "
            elif ev.severity == 0.0 and ev.days_infected > 0:
                tag = "[incub]"
            else:
                tag = "       "
            self._log_file.write(
                f"  clinic | {name:<20} [{ptrait:<7}] {tag} "
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
