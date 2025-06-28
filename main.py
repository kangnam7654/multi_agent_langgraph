import dotenv
import yaml
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.custom_agents import AgentDirector, AgentInspector, AgentWriter
from src.custom_states import AgentState
from src.custom_tools.lore_book import get_dragon_lore

_ = dotenv.load_dotenv()

PROMPTS_DIR = "src/prompts"


def load_prompt(path: str) -> str:
    """
    Load a prompt from the specified path.

    Args:
        path (str): The path to the prompt file.

    Returns:
        str: The content of the prompt file.
    """
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)["system"]


def router(state: dict) -> str:
    """
    Router function to determine the next agent based on the current state.

    Args:
        state (dict): The current state of the graph.

    Returns:
        str: The name of the next agent to invoke.
    """
    if state["next_agent"] == "director":
        return "director"
    elif state["next_agent"] == "inspector":
        return "inspector"
    elif state["next_agent"] == "writer":
        return "writer"
    elif state["next_agent"] == END:
        return END
    else:
        raise ValueError(f"Unknown agent: {state['next_agent']}")


def build_graph() -> StateGraph:
    """
    Build the state graph for the custom agent.

    Returns:
        StateGraph: The constructed state graph.
    """
    # Load prompts
    director_prompt = load_prompt(f"{PROMPTS_DIR}/director_v1.yaml")
    inspector_prompt = load_prompt(f"{PROMPTS_DIR}/inspector_v1.yaml")
    writer_prompt = load_prompt(f"{PROMPTS_DIR}/writer_v1.yaml")

    # Set up agents
    director = AgentDirector("Director", director_prompt, tools=[get_dragon_lore])
    inspector = AgentInspector("Inspector", inspector_prompt, tools=[get_dragon_lore])
    writer = AgentWriter("Writer", writer_prompt, tools=[get_dragon_lore])

    # Build graph
    graph = StateGraph(AgentState)

    # Define nodes
    graph.add_node("director", director)
    graph.add_node("inspector", inspector)
    graph.add_node("writer", writer)

    # Define edges
    graph.add_edge(START, "director")
    # graph.add_edge("director", END)
    graph.add_conditional_edges(
        "director",
        router,
        {
            "inspector": "inspector",
            "writer": "writer",
            END: END,
        },
    )
    graph.add_conditional_edges("inspector", router, {"director": "director", "writer": "writer"})
    graph.add_conditional_edges("writer", router, {"inspector": "inspector", "director": "director"})

    checkpointer = MemorySaver()
    compiled_graph = graph.compile(checkpointer=checkpointer)
    return compiled_graph  # type: ignore


def main():
    """
    Main function to run the custom agent graph.
    """
    graph = build_graph()

    for chunk in graph.stream(
        {"task": "드래곤에 대한 간단한 시나리오를 써보세요.", "allow_tools": False},
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
            print("==========================")


if __name__ == "__main__":
    main()
