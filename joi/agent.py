"""The agent brain: one LlamaIndex FunctionAgent holding every registered tool.

Single responsibility: turn a user utterance into a spoken-ready reply,
calling capability tools when needed, and stream the reply text out as it
is generated. Owns conversation memory (a reused workflow Context).

Verified against the installed packages (llama-index-llms-ollama 0.10.1,
llama-index-core 0.14.22) on 2026-07-05:
  - Ollama(context_window=8192) DOES reach Ollama's num_ctx
    (`ollama ps` showed CONTEXT 8192, 6.2 GB, 100% GPU).
  - additional_kwargs merges into the per-request `options` dict, so gaming
    mode passes {"num_gpu": 0} through it.
  - keep_alive accepts "30m" (`ollama ps` showed UNTIL ~30 minutes).
  - agent.run() returns a WorkflowHandler; stream_events() yields AgentStream
    (text deltas) and ToolCallResult events; the streamed deltas concatenate
    to exactly the final response text.
"""
from collections.abc import AsyncIterator
from dataclasses import dataclass

from llama_index.core.agent.workflow import AgentStream, FunctionAgent, ToolCallResult
from llama_index.core.workflow import Context
from llama_index.llms.ollama import Ollama

from joi.capabilities import REGISTRY, all_tools, describe_all
from joi.config import Config


@dataclass
class AgentEvent:
    """One streamed event from an agent turn: a text delta and/or a tool note."""

    delta: str = ""
    tool_note: str | None = None  # e.g. "open_app({'app_name': 'firefox'})"


class JoiAgent:
    """Single-agent design (validated as lower-latency than multi-agent
    handoffs). All registered tools live on ONE FunctionAgent.

    Phase 3+ (multi-agent): this constructor is the contained change point —
    instead of one FunctionAgent with all_tools(), build one FunctionAgent per
    Capability in REGISTRY and compose them with
    llama_index.core.agent.workflow.AgentWorkflow. The public surface
    (stream_chat) stays identical.
    """

    def __init__(self, cfg: Config):
        self.llm = Ollama(
            model=cfg.model,
            request_timeout=cfg.request_timeout,
            thinking=cfg.think,                    # False -> ~10x faster turns
            context_window=cfg.num_ctx,            # verified: reaches num_ctx
            keep_alive=cfg.keep_alive,             # model stays loaded between commands
            additional_kwargs=cfg.ollama_extra_options,  # gaming mode: num_gpu=0
        )
        self.agent = FunctionAgent(
            name="joi",
            description="A local voice assistant.",
            llm=self.llm,
            tools=all_tools(),
            system_prompt=(
                f"{cfg.system_prompt}\n"
                "You have tools for these capabilities:\n"
                f"{describe_all()}\n"
                "Use them when the user asks for a matching action; "
                "answer directly otherwise.\n"
                "ALWAYS call the tool for an action, even if a similar "
                "request failed earlier — never assume the result from "
                "memory. App name matching is fuzzy: pass the name exactly "
                "as the user said it."
            ),
        )
        # Reusing one Context across run() calls is what gives multi-turn memory.
        self.ctx = Context(self.agent)
        self.capabilities = REGISTRY

    async def stream_chat(self, user_msg: str) -> AsyncIterator[AgentEvent]:
        """Run one agent turn, yielding text deltas as they are generated.

        Tool calls are surfaced as tool_note events so the UI can print them.
        The concatenated deltas equal the final reply (verified on the
        installed version), so callers need no separate final result.
        """
        handler = self.agent.run(user_msg=user_msg, ctx=self.ctx)
        async for ev in handler.stream_events():
            if isinstance(ev, AgentStream) and ev.delta:
                yield AgentEvent(delta=ev.delta)
            elif isinstance(ev, ToolCallResult):
                yield AgentEvent(tool_note=f"{ev.tool_name}({ev.tool_kwargs})")
        await handler  # surface any exception from the run

    async def warm_up(self) -> None:
        """Preload the model into Ollama so the first real turn is fast."""
        await self.llm.acomplete("hi")

    def to_cpu(self) -> None:
        """Switch all following turns to CPU-only inference (num_gpu=0).
        additional_kwargs is merged into options on every request, so
        mutating it here takes effect on the next LLM call."""
        self.llm.additional_kwargs = {**self.llm.additional_kwargs, "num_gpu": 0}
