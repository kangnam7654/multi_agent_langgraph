import dotenv
import yaml
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.agents import AgentDirector, AgentInspector, AgentWriter
from src.states import AgentState
from src.tools.load_prompt import load_prompt
from src.tools.lore_book import get_dragon_lore

_ = dotenv.load_dotenv()

PROMPTS_DIR = "src/prompts"


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"

    if state["current_agent"] == "director":
        if state.get("can_publish", False):
            return END
        return "writer"
    elif state["current_agent"] == "inspector":
        if state["next_agent"] == "writer":
            return "writer"
        elif state["next_agent"] == "director":
            return "director"
        else:
            raise ValueError(f"Unknown next agent: {state['next_agent']}")
    elif state["current_agent"] == "writer":
        return "inspector"
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
    inspector_prompt = load_prompt(f"{PROMPTS_DIR}/inspector_v1.yaml")
    writer_prompt = load_prompt(f"{PROMPTS_DIR}/writer_v1.yaml")

    # =================
    # | Set up agents |
    # =================
    director = AgentDirector("Director", director_prompt)
    inspector = AgentInspector("Inspector", inspector_prompt)
    writer = AgentWriter("Writer", writer_prompt)
    tool_node = ToolNode([get_dragon_lore])

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
    graph.add_node("tools", tool_node)

    # ================
    # | Define edges |
    # ================
    graph.add_edge(START, "director")

    # tools handling
    graph.add_conditional_edges("director", should_continue, {"tools": "tools", "writer": "writer", END: END})
    graph.add_conditional_edges(
        "inspector", should_continue, {"tools": "tools", "director": "director", "writer": "writer"}
    )
    graph.add_conditional_edges("writer", should_continue, {"tools": "tools", "inspector": "inspector"})
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
    mermaid = graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(mermaid)
    pass

    for chunk in graph.stream(
        {
            "task": "Write some scenario about a dragon.",
        },
        {"configurable": {"thread_id": "1"}},
    ):
        for event in chunk.values():
            print(f"Current Agent: {event['current_agent']}")
            print(f"Next Agent: {event['next_agent']}")
            print(f"Revision: {event['revision']}")
            print("Messages:")
            print(event["messages"][-1].content)
            if event["current_agent"] == "writer":
                print("Scenario:")
                print(event["scenario"])
            print("==========================\n")


if __name__ == "__main__":
    main()
