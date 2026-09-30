"""In-memory pipeline runner used by the experiments (no database, no production changes)."""

from collections.abc import Sequence
from pathlib import Path

from drone_agent.models import Alert, Entity, Event, Observation
from drone_agent.pipeline import EntityLinker, EventConfig, aggregate_observations
from drone_agent.rules import RuleEngine
from drone_agent.rules.config import SecurityRulesConfig
from drone_agent.scenario import load_scenario
from drone_agent.simulate import run_scenario

SCENARIO_DIR = Path(__file__).resolve().parents[2] / "scenarios"


def run_pipeline(
    observations: Sequence[Observation],
    event_config: EventConfig | None = None,
    rules_config: SecurityRulesConfig | None = None,
    link_entities: bool = True,
) -> tuple[list[Event], list[Entity], list[Alert]]:
    """observations -> events -> (optional) entities -> alerts, exactly as production does.

    `link_entities=False` is the entity-memory ablation: every vehicle event stays independent.
    """
    events = aggregate_observations(observations, event_config)
    if link_entities:
        events, entities = EntityLinker(event_config).link(events)
    else:
        entities = []
    return events, entities, RuleEngine(rules_config).run(events, entities)


def scenario_observations(name: str) -> list[Observation]:
    return run_scenario(load_scenario(SCENARIO_DIR / f"{name}.yaml")).observations
