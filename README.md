# Scenario Writer (WIP)

This project is a multi-agent game scenario writing system built with [LangChain](https://github.com/langchain-ai/langchain) and [LangGraph](https://github.com/langchain-ai/langgraph).  
It aims to automate the collaborative process of creating, inspecting, and directing game scenarios using AI agents.

## Features

- **Multi-agent workflow:** Includes Director, Inspector, and Writer agents, each with distinct roles.
- **Scenario consistency:** Inspector agent checks for logical and canonical consistency with the setting book.
- **Customizable prompts:** System prompts for each agent are configurable via YAML files.
- **Extensible tools:** Agents can use custom tools, such as lore book lookups, to enhance scenario quality.

## Getting Started

> **Note:** This project is a work in progress and not yet production-ready.

### Requirements

- Python 3.11+
- See `pyproject.toml` for dependencies.

### Installation

```bash
git clone https://github.com/yourusername/scenario_writer.git
cd scenario_writer
pip install .
```

### Usage

Run the main script:

```bash
python main.py
```

This will start the scenario writing workflow using the provided prompts and agents.

## Project Structure

- `src/custom_agents/` - Agent implementations (Director, Inspector, Writer)
- `src/custom_tools/` - Custom tools (e.g., lore book)
- `src/custom_states/` - State definitions
- `src/custom_messages/` - Custom message formats
- `src/prompts/` - System prompts for each agent

## Status

This project is under active development.  
Features, APIs, and structure may change frequently.

## License

[MIT](LICENSE)
