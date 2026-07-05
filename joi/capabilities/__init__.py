"""Capability registry — THE extension point for giving Joi new skills.

To add a new capability:
  1. Create one file in this package (copy the shape of `weather.py`):
     it must expose a module-level `CAPABILITY = Capability(...)`.
  2. Import it below and add it to REGISTRY (one line).
That's it. The agent picks up its tools and description automatically.

Phase 3+ (multi-agent): REGISTRY is deliberately a list of self-contained
Capability objects so that graduating to LlamaIndex AgentWorkflow is a
contained change — build one FunctionAgent per Capability instead of one
agent with all tools. That change lives entirely in `joi/agent.py`; nothing
in this package or in the capability modules needs to move.
"""
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Capability:
    """One skill Joi has: a name, a one-line description (fed to the agent's
    system prompt so it knows the skill exists), and the tool functions the
    LLM can call. Tool docstrings are what the LLM reads — keep them clear."""

    name: str
    description: str
    tools: list[Callable] = field(default_factory=list)


from joi.capabilities.apps import CAPABILITY as _apps
from joi.capabilities.weather import CAPABILITY as _weather

REGISTRY: list[Capability] = [
    _apps,
    _weather,
    # <- register new capabilities here (one line per capability)
]


def all_tools() -> list[Callable]:
    """Flat list of every registered tool function (for the single agent)."""
    return [tool for cap in REGISTRY for tool in cap.tools]


def describe_all() -> str:
    """One line per capability, injected into the agent's system prompt."""
    return "\n".join(f"- {cap.name}: {cap.description}" for cap in REGISTRY)
