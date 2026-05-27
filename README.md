# Federated Simulated World — ncurses TUI

Python implementation of the simulation framework described in:
> Nascimento et al., *"Federated Simulated World Formalization"* (2025)

---

## Quick Start

```bash
python main.py
```

Requires Python 3.9+ and only stdlib (`curses`, `math`, `random`, `collections`, `numpy`).

```bash
pip install numpy
```

---

## Controls

| Key         | Action                                      |
|-------------|---------------------------------------------|
| `SPACE`     | Advance one tick (5 min)                    |
| `R`         | Toggle auto-run mode                        |
| `D`         | Inject disease cloud at random location     |
| `F`         | Cycle agent filter: ALL → S → I → R        |
| `↑ / ↓`    | Scroll event log                            |
| `PgUp/PgDn`| Scroll agent list                           |
| `Q`         | Quit                                        |

---

## Layout

```
┌─────────────────────────┬──────────────────┐
│  WORLD MAP              │  SIR DYNAMICS    │
│  · susc  ✶ infect       │  sparkline chart │
│  ○ recov                ├──────────────────┤
│                         │  AGENTS list     │
├─────────────────────────┼──────────────────┤
│  CLINIC / LLM DOCTOR    │  EVENT LOG       │
│  queue + case outcomes  │                  │
└─────────────────────────┴──────────────────┘
 status bar
```

---

## Architecture (mirrors paper §4)

```
WorldEngine
 ├── Location[]          (WFC-generated spatial graph)
 ├── Agent[]
 │    └── HealthState    (s, σ progression — eq. 4/5)
 ├── DiseaseCloud[]      (ambient viral load ω — eq. 2)
 ├── SIRModel            (population observer — eq. 3)
 └── ClinicQueue
      └── DiagnosticClient  ← extension point
           └── FLClient     ← extension point
```

---

## Extension Points

### 1. Plug in your LLM diagnostic function

```python
from simulation.world import WorldEngine
from simulation.models import DiagnosticEvent, DiagnosticAction

def my_llm_doctor(event: DiagnosticEvent) -> DiagnosticEvent:
    """
    Called once per symptomatic agent per day.
    event.severity  — physiological severity s ∈ [0,1]
    event.symptoms  — manifest symptoms σ ∈ [0,1]
    event.days_infected — days since infection
    
    Must return event with .action and .oracle_label set.
    """
    # Call your local LLM / MedAlpaca API here
    prompt = (
        f"Patient has severity {event.severity:.2f}, "
        f"symptoms {event.symptoms:.2f}, "
        f"infected for {event.days_infected} days. "
        "Choose: home_recovery | hospitalise | resolve"
    )
    response = call_your_llm(prompt)          # ← your code here
    event.action = DiagnosticAction[response.upper()]
    event.oracle_label = response
    return event

world = WorldEngine(num_agents=30)
world.register_diagnostic_fn(my_llm_doctor)
```

### 2. Plug in federated learning (Flower / flwr-llm)

Override `world.run_fl_round()` or subclass `WorldEngine`:

```python
import flwr as fl

class FLWorldEngine(WorldEngine):
    def run_fl_round(self):
        super().run_fl_round()
        # Build local dataset from clinic_queue.processed
        dataset = [(ev.severity, ev.symptoms, ev.oracle_label)
                   for ev in self.clinic_queue.processed]
        # Trigger flwr client update (eq. 6)
        local_update = self.local_model.train(dataset)
        # FedAvg aggregation (eq. 7) happens server-side via flwr
        fl.client.start_numpy_client(server_address="...", client=local_update)
```

### 3. Multiple worlds as FL clients

```python
worlds = [WorldEngine(num_agents=30, seed=k) for k in range(5)]
for world in worlds:
    world.register_diagnostic_fn(my_llm_doctor)
# Run each in its own thread/process; aggregate via FLServer
```

---

## Mathematical Grounding

| Equation | Location in code |
|---|---|
| Mobility policy π(a,t,type) | `models.py::DailySchedule.location_for_tick` |
| Exposure ε_t(a) (eq. 2)     | `world.py::WorldEngine.step_tick`            |
| P(S→I) (eq. 3)              | `models.py::Agent.apply_daily_infection`     |
| [s,σ] progression (eq. 4/5) | `models.py::HealthState.step`               |
| Local update Δθ_k (eq. 6)   | `world.py::WorldEngine.run_fl_round` (stub) |
| FedAvg Θ (eq. 7)            | `world.py::WorldEngine.run_fl_round` (stub) |
