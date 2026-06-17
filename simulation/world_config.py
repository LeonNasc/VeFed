"""
World configuration — pure data objects, no simulation logic.

AgentConfig   describes the population and how agents generate observations.
EpidemicConfig describes SIR dynamics and disease mix.
WorldConfig   composes both; passed to WorldEngine(config, seed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from simulation.data_sources import DataSource
    from simulation.case_summary import CaseSummarizer


@dataclass
class AgentConfig:
    """
    Population size and observation modality.

    data_source determines how infectious agents produce their opening complaint:
      TemplateDataSource    — fast template phrases, no LLM
      PhraseLibraryDataSource — curated phrase banks, higher lexical diversity
      OllamaDataSource      — live local LLM (PatientLLMClient)

    Confusion logic (atypical presentations) lives inside the DataSource, not here.
    """
    num_agents:            int          = 30
    data_source:           "DataSource" = field(
        default_factory=lambda: _default_data_source())
    background_visit_rate: float        = 0.025
    case_summarizer:       Optional["CaseSummarizer"] = None


@dataclass
class EpidemicConfig:
    """SIR dynamics and disease prevalence."""
    progressions:        list[str]             = field(
        default_factory=lambda: ["Influenza"])
    disease_strategy:    str                   = "Influenza"  # transmission strategy key
    disease_weights:     Optional[list[float]] = None   # None → uniform
    beta_scale:          float                 = 1.0
    initial_seeds:       int                   = 3
    contact_rate_sigma:  float                 = 0.0    # 0 = off; >0 = lognormal spread
    # Static mode — bypasses SIR; visits generated at a fixed rate
    static_mode:         bool                  = False
    infectious_fraction: float                 = 0.5
    cases_per_day:       int                   = 20


@dataclass
class WorldConfig:
    """
    Full per-silo simulation specification.

    seed_offset is added to the global seed so each silo in a federation
    draws from an independent RNG stream without manual bookkeeping.
    """
    agents:               AgentConfig   = field(default_factory=AgentConfig)
    epidemic:             EpidemicConfig = field(default_factory=EpidemicConfig)
    seed_offset:          int            = 0
    reveal_incubating_icd: bool          = True


def _default_data_source() -> "DataSource":
    from simulation.data_sources import TemplateDataSource
    return TemplateDataSource()
