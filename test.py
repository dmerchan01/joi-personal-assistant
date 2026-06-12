import asyncio
from ddgs import DDGS
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    AgentInput,
    AgentOutput,
    ToolCall,
    ToolCallResult,
    AgentStream,
)
from llama_index.llms.ollama import Ollama

# Two models — each specialized for a different role
llm_tools = Ollama(model="qwen2.5:7b", request_timeout=240.0)   # better at tool use
llm_reason = Ollama(model="llama3.1:8b", request_timeout=120.0) # better at reasoning

async def search_web(query: str) -> str:
    """Useful for using the web to answer questions about current events or any topic."""
    for attempt in range(3):
        try:
            results = DDGS().text(query, max_results=5)
            if not results:
                return "No results found."
            formatted = []
            for r in results:
                formatted.append(f"Title: {r['title']}\nURL: {r['href']}\nSummary: {r['body']}")
            return "\n\n".join(formatted)
        except Exception as e:
            if attempt == 2:
                return f"Search failed after 3 attempts: {str(e)}"
            await asyncio.sleep(2)

async def record_notes(ctx: Context, notes: str, notes_title: str = "Untitled Notes") -> str:
    """Useful for recording notes on a given topic."""
    current_state = await ctx.store.get("state")
    if "research_notes" not in current_state:
        current_state["research_notes"] = {}
    current_state["research_notes"][notes_title] = notes
    await ctx.store.set("state", current_state)
    return "Notes recorded."

async def write_report(ctx: Context, report_content: str) -> str:
    """Useful for writing a report on a given topic."""
    current_state = await ctx.store.get("state")
    current_state["report_content"] = report_content
    await ctx.store.set("state", current_state)
    return "Report written."

async def review_report(ctx: Context, review: str) -> str:
    """Useful for reviewing a report and providing feedback."""
    current_state = await ctx.store.get("state")
    current_state["review"] = review
    await ctx.store.set("state", current_state)
    return "Report reviewed."

research_agent = FunctionAgent(
    name="ResearchAgent",
    description="Searches the web and records notes.",
    system_prompt=(
        "You are the ResearchAgent. Follow these steps in order:\n"
        "1. Call search_web to find information.\n"
        "2. Call record_notes to save what you found.\n"
        "3. Call handoff to transfer to WriteAgent.\n"
        "Do NOT skip any step."
    ),
    llm=llm_tools,  # qwen — reliable tool use
    tools=[search_web, record_notes],
    can_handoff_to=["WriteAgent"],
)

write_agent = FunctionAgent(
    name="WriteAgent",
    description="Writes a report using the research notes.",
    system_prompt=(
        "You are the WriteAgent. Follow these steps in order:\n"
        "1. Call write_report with the full markdown report as report_content.\n"
        "2. Call handoff to transfer to ReviewAgent.\n"
        "Do NOT output the report as text. You MUST call write_report first."
    ),
    llm=llm_tools,  # qwen — reliable tool use
    tools=[write_report],
    can_handoff_to=["ReviewAgent", "ResearchAgent"],
)

review_agent = FunctionAgent(
    name="ReviewAgent",
    description="Reviews the report and provides feedback.",
    system_prompt=(
        "You are the ReviewAgent. Follow these steps in order:\n"
        "1. Call review_report with your feedback as the review parameter.\n"
        "2. After calling review_report, output your final verdict.\n"
        "Do NOT skip calling review_report."
    ),
    llm=llm_reason,  # llama — better reasoning for feedback
    tools=[review_report],
)

async def main():
    agent_workflow = AgentWorkflow(
        agents=[research_agent, write_agent, review_agent],
        root_agent="ResearchAgent",
        initial_state={
            "research_notes": {},
            "report_content": "Not written yet.",
            "review": "Review required.",
        },
    )

    handler = agent_workflow.run(
        user_msg="Write me a report on the history of the web. Briefly describe the history of the world wide web, including the development of the internet and the development of the web, including 21st century developments."
    )

    current_agent = None
    async for event in handler.stream_events():
        if (
            hasattr(event, "current_agent_name")
            and event.current_agent_name != current_agent
        ):
            current_agent = event.current_agent_name
            print(f"\n{'='*50}")
            print(f"🤖 Agent: {current_agent}")
            print(f"{'='*50}\n")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("📤 Output:", event.response.content)
            if event.tool_calls:
                print("🛠️  Planning to use tools:", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCall):
            print(f"🔨 Calling Tool: {event.tool_name}")
            print(f"   With arguments: {event.tool_kwargs}")
        elif isinstance(event, ToolCallResult):
            print(f"🔧 Tool Result ({event.tool_name}):")
            print(f"   Output: {str(event.tool_output)[:200]}...")

    await asyncio.sleep(0.1)

asyncio.run(main())