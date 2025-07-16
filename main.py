import dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.agents import AgentDirector, AgentInspector, AgentWriter
from src.models.bitnet import BitnetlLLM
from src.states import AgentState
from src.tools.load_prompt import load_prompt
from src.tools.lore_book import get_dragon_lore

_ = dotenv.load_dotenv()

PROMPTS_DIR = "src/prompts"


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:  # type: ignore[reportAttributeAccessIssue]
        return "tools"

    if state["current_agent"] == "director":
        if state["next_agent"] == "writer":
            return "writer"
        elif state["next_agent"] == END:
            return END
        else:
            raise ValueError(f"Unknown next agent: {state['next_agent']}")

    elif state["current_agent"] == "inspector":
        if state["next_agent"] == "writer":
            return "writer"
        elif state["next_agent"] == "director":
            return "director"
        elif state["next_agent"] == "self":
            return "inspector"
        else:
            raise ValueError(f"Unknown next agent: {state['next_agent']}")

    elif state["current_agent"] == "writer":
        if state["next_agent"] == "inspector":
            return "inspector"
        elif state["next_agent"] == "self":
            return "writer"
        else:
            raise ValueError(f"Unknown next agent: {state['next_agent']}")

    else:
        raise ValueError(f"Unknown current agent: {state['current_agent']}")


def build_graph() -> StateGraph:
    """
    Build the state graph for the custom agent.

    Returns:
        StateGraph: The constructed state graph.
    """
    # ================
    # | Load prompts |
    # ================
    director_prompt = load_prompt(f"{PROMPTS_DIR}/director_v2.yaml")
    inspector_prompt = load_prompt(f"{PROMPTS_DIR}/inspector_v2.yaml")
    writer_prompt = load_prompt(f"{PROMPTS_DIR}/writer_v2.yaml")

    bitnet = BitnetlLLM(use_default_model=True)

    inspector_prompt = inspector_prompt.replace("<tool_replace>", "get_dragon_lore")
    writer_prompt = writer_prompt.replace("<tool_replace>", "get_dragon_lore")
    # =================
    # | Set up agents |
    # =================
    director = AgentDirector("Director", director_prompt, model=bitnet, last_n_messages=2)
    inspector = AgentInspector("Inspector", inspector_prompt, model=bitnet, last_n_messages=2)
    writer = AgentWriter("Writer", writer_prompt, model=bitnet, last_n_messages=2)

    # ===============
    # | Build graph |
    # ===============
    graph = StateGraph(AgentState)

    # ================
    # | Define nodes |
    # ================
    graph.add_node("director", director)
    graph.add_node("inspector", inspector)
    graph.add_node("writer", writer)
    graph.add_node("tools", get_dragon_lore)

    # ================
    # | Define edges |
    # ================
    graph.add_edge(START, "director")

    # tools handling
    graph.add_conditional_edges("director", should_continue, {"tools": "tools", "writer": "writer", END: END})
    graph.add_conditional_edges(
        "inspector",
        should_continue,
        {"tools": "tools", "director": "director", "writer": "writer", "inspector": "inspector"},
    )
    graph.add_conditional_edges(
        "writer", should_continue, {"tools": "tools", "inspector": "inspector", "writer": "writer"}
    )
    graph.add_conditional_edges(
        "tools",
        should_continue,
        {
            "tools": "tools",
            "director": "director",
            "inspector": "inspector",
            "writer": "writer",
        },
    )

    checkpointer = MemorySaver()
    compiled_graph = graph.compile(checkpointer=checkpointer)
    return compiled_graph  # type: ignore


def main():
    """
    Main function to run the custom agent graph.
    """
    graph = build_graph()
    mermaid = graph.get_graph().draw_mermaid_png()  # type: ignore[reportAttributeAccessIssue]
    with open("graph.png", "wb") as f:
        f.write(mermaid)
    pass

    for chunk in graph.stream(  # type: ignore[reportAttributeAccessIssue]
        {
            "task": "Write some scenarios about a dragon. Refer to the lore book for details.",
        },
        {"configurable": {"thread_id": "1"}},
    ):
        for event in chunk.values():
            print("\n")
            print(f"Revision: {event['revision']}")
            print(f"Current Agent: {event['current_agent']}")
            print(f"Next Agent: {event['next_agent']}")
            print("Director Can Publish:", event.get("director_can_publish", False))
            print("Inspector Passed:", event.get("inspector_passed", False))
            print("tool_calls:", event.get("tool_calls", []))
            print("\n")
            print("Messages:")
            print(event["messages"][-1].content)
            print("Scenario:")
            print(event.get("writer_scenario", "No scenario yet"))
            print("==========================")


if __name__ == "__main__":
    main()
