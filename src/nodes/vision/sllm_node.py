from langchain_core.messages import SystemMessage


class LLMNode:
    def __init__(self, name, llm_model, system_prompt="You are a helpful assistant."):
        self.name = name
        self.llm_model = llm_model
        self.system_prompt = system_prompt

    def __call__(self, state):
        messages = state.get("messages", [])
        if messages:
            response = self.llm_model.invoke([SystemMessage(content=self.system_prompt)] + messages)
            state["messages"].append(response)
        return state
