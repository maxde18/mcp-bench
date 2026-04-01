# Planning Phase Analysis of LLM Tool-Use Agents

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![MCP Protocol](https://img.shields.io/badge/MCP-Protocol-green)](https://github.com/anthropics/mcp)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository contains the experimental codebase for a thesis investigating the **planning phase of LLM tool-use agents**. Rather than evaluating end-to-end task performance, the focus is on whether and how an agent's planning behaviour — expressed as a tool dependency graph (DAG) — changes across models, prompts, and runs.

The project is built on top of [MCP-Bench](https://arxiv.org/abs/2508.20453), a benchmark framework for evaluating LLM agents across 28 MCP servers. The full execution infrastructure from MCP-Bench is retained under `runtime/` for reference and future use, but the primary experimental work lives in `planning/`, `dag/`, and `analysis/`.

---

## Research Focus

For each task, an agent produces a **tool dependency graph**: a DAG where nodes are tool calls and edges encode data dependencies between them. The thesis collects these DAGs across multiple runs and models and analyses:

- Whether the agent selects the correct tools and orders them correctly
- How much variance exists in plans across repeated runs of the same task
- Whether changes to the model or prompt measurably shift planning behaviour

---

## Repository Structure

```
mcp-bench/
│
│  ── THESIS CORE ────────────────────────────────────────────────────
│
├── planning/                  # Planning-phase agents
│   └── agents/
│       ├── plan_only_executor.py    # Single LLM call → DAG output, no tool execution
│       └── langgraph_executor.py   # Plan-then-execute agent via LangGraph
│
├── dag/                       # Tool dependency graph analysis
│   ├── models.py              # DAG node/edge data structures
│   ├── extractor.py           # Parse agent plan output → DAG
│   ├── comparator.py          # Similarity metrics and behavioural diff between DAGs
│   └── store.py               # Persist and retrieve DAGs by task/run/model
│
├── analysis/                  # Experiment notebooks and visualisations
│
│  ── SHARED INFRASTRUCTURE ──────────────────────────────────────────
│
├── mcp/                       # MCP server infrastructure (planning AND execution)
│   ├── connector.py           # Connects to a single server; discovers and formats tool schemas
│   ├── server_manager.py      # Non-persistent multi-server manager
│   ├── server_manager_persistent.py  # Persistent multi-server manager
│   ├── tool_cache.py          # Tool result caching
│   └── connection_manager.py  # Async context manager: connect servers, expose all_tools
│
├── tasks/                     # Pre-generated benchmark task files
│   ├── mcpbench_tasks_single_runner_format.json    # ~30 single-server tasks
│   ├── mcpbench_tasks_multi_2server_runner_format.json  # ~100 two-server tasks
│   └── mcpbench_tasks_multi_3server_runner_format.json  # ~60 three-server tasks
├── synthesis/                 # Task generation pipeline
├── config/                    # benchmark_config.yaml + config loader
├── llm/                       # LLM provider abstraction (OpenRouter, Azure)
├── utils/                     # MCP server discovery, error handling
│
│  ── SECONDARY / FUTURE WORK ────────────────────────────────────────
│
├── runtime/                   # Full execution infrastructure (from MCP-Bench)
│   ├── agents/                # Baseline multi-round executor
│   └── benchmark/             # LLM-as-judge evaluation pipeline
│
├── mcp_servers/               # 28 MCP server implementations
│
│  ── ENTRY POINTS ───────────────────────────────────────────────────
│
├── run_planning.py            # Primary: planning benchmark + DAG collection
└── run_benchmark.py           # Secondary: full execution benchmark
```

### Restructuring rationale

| Change | Reason |
|--------|--------|
| `planning/` introduced | Isolates the planning-phase agents as the primary research artefact |
| `dag/` introduced | First-class module for DAG extraction, comparison, and storage |
| `analysis/` replaces `network_generation/` | Clearer name; home for experiment notebooks and visualisations |
| `mcp/` promoted to top level | Planning agents require MCP connections for tool **discovery** (not just execution). Placing the MCP module alongside `planning/` reflects that it is shared infrastructure, not execution-only |
| `mcp/connection_manager.py` extracted | `ConnectionManager` was buried in `runtime/benchmark/runner.py`; it is now a named shared component used by both the planning runner and the full benchmark runner |
| `runtime/` contains only execution code | Now holds only the baseline execution agent and the LLM-as-judge pipeline — nothing the planning workflow depends on |
| `run_planning.py` as primary entry point | Renamed from `run_planning_benchmark.py` |

---

## Setup

### 1. Install dependencies

```bash
conda create -n mcpbench python=3.10
conda activate mcpbench
cd mcp_servers
bash ./install.sh
cd ..
```

### 2. Set up API keys

```bash
cat > .env << EOF
export OPENROUTER_API_KEY="your_openrouter_key"
EOF
```

### 3. Configure MCP server API keys

Some servers require external API keys. Set them in `./mcp_servers/api_key`:

| Key | Server | Where to get it |
|-----|--------|-----------------|
| `NPS_API_KEY` | National Parks | [nps.gov](https://www.nps.gov/subjects/developer/get-started.htm) |
| `NASA_API_KEY` | NASA Data | [api.nasa.gov](https://api.nasa.gov/) |
| `HF_TOKEN` | Hugging Face | [huggingface.co](https://huggingface.co/docs/hub/security-tokens) |
| `GOOGLE_MAPS_API_KEY` | Google Maps | [developers.google.com](https://developers.google.com/maps) |
| `NCI_API_KEY` | BioMCP | [clinicaltrialsapi.cancer.gov](https://clinicaltrialsapi.cancer.gov/signin) |

### 4. Verify server connectivity

```bash
python utils/collect_mcp_info.py
# Expected: "28/28 servers connected" and all tools returned
```

---

## Running Planning Experiments

All planning agents are implemented using LangGraph. The primary agent (`PlanOnlyExecutor`) uses a single LangGraph node that makes one LLM call and emits a DAG; the `LangGraphExecutor` uses a multi-node graph (planner → executor → synthesizer) for plan-then-execute experiments.

```bash
source .env

# Single-server tasks (~30 tasks)
python run_planning.py \
    --tasks tasks/mcpbench_tasks_single_runner_format.json \
    --model claude-sonnet-4

# Two-server tasks (~100 tasks)
python run_planning.py \
    --tasks tasks/mcpbench_tasks_multi_2server_runner_format.json \
    --model claude-sonnet-4

# Three-server tasks (~60 tasks)
python run_planning.py \
    --tasks tasks/mcpbench_tasks_multi_3server_runner_format.json \
    --model claude-sonnet-4
```

The planner makes one LLM call per task and returns a tool dependency DAG. No tools are executed. Results are saved under `results/planning/<model>/<timestamp>/`.

### Tool input mode: prompt text vs. native API field

By default, tool schemas are formatted as plain text and embedded in the user prompt. This works with any model but means the tool descriptions consume prompt tokens and are treated as ordinary text.

For fine-tuned tool-calling models, pass `--native-tools` to send the MCP tool schemas via the OpenAI `tools` API field instead:

```bash
python run_planning.py \
    --tasks tasks/mcpbench_tasks_multi_2server_runner_format.json \
    --model your-fine-tuned-model \
    --native-tools
```

| Mode | How tools reach the model | When to use |
|------|--------------------------|-------------|
| Default (prompt text) | Tool names, descriptions, and JSON schemas are embedded as a text block in the user prompt | Any model; baseline experiments |
| `--native-tools` | Tools are passed in the structured `tools` field of the API request; `tool_choice` is set to `"none"` so the model is aware of them but still responds with text | Fine-tuned tool-calling models; experiments isolating the effect of tool input format |

Under the hood, `--native-tools` calls `MCPConnector.format_tools_for_api(all_tools)` to convert the tool dict to the `[{"type": "function", "function": {...}}]` schema the OpenAI API expects. Tool names are sanitised (`ServerName:tool_name` → `ServerName__tool_name`) to satisfy the API's naming constraint; the planning agent is instructed to use the original colon-separated names in its DAG output.

The mode used is recorded in each output file under `experiment_config.native_tools` so results are self-documenting.

### Structured output (OpenRouter models only)

When running against an OpenRouter model, `PlanOnlyExecutor` automatically passes the dependency graph JSON schema to the API via the `response_format` parameter using OpenRouter's `json_schema` structured output format:

```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "dependency_graph",
    "schema": {
      "type": "object",
      "properties": {
        "nodes": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "id":          { "type": "string" },
              "tool":        { "type": "string" },
              "parameters":  { "type": "object" },
              "depends_on":  { "type": "array", "items": { "type": "string" } },
              "description": { "type": "string" }
            },
            "required": ["id", "tool", "parameters", "depends_on", "description"]
          }
        }
      },
      "required": ["nodes"]
    }
  }
}
```

This applies to both the default (prompt-text) and `--native-tools` modes. For Azure and other non-OpenRouter providers the schema is not sent and JSON parsing falls back to the `clean_and_parse_json` heuristic as before.

---

## Running the Full Benchmark (secondary)

```bash
source .env
python run_benchmark.py --list-models
python run_benchmark.py --models gpt-oss-20b

# Scoped to specific task files
python run_benchmark.py --models gpt-oss-20b \
    --tasks-file tasks/mcpbench_tasks_multi_2server_runner_format.json
```

---

## Adding a Planning Agent

All agents — whether planning-only or full executors — share the same interface. The runner injects an LLM provider and a server manager; the agent returns a result dict.

### 1. Create the agent file

Place the file under `planning/agents/` for planning agents or `runtime/agents/` for execution agents.

```python
# planning/agents/your_agent.py
import logging
from typing import Any, Dict, List

import config.config_loader as config_loader
from runtime.mcp_modules.connector import MCPConnector

logger = logging.getLogger(__name__)


class YourAgent:

    def __init__(self, llm_provider, server_manager) -> None:
        self.llm = llm_provider
        self.server_manager = server_manager
        self.all_tools: Dict[str, Any] = server_manager.all_tools

    async def execute(self, task: str) -> Dict[str, Any]:
        tool_descriptions = MCPConnector.format_tools_for_prompt(self.all_tools)
        execution_results: List[Dict[str, Any]] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        # --- your logic here ---

        response, usage = await self.llm.get_completion(
            "You are a helpful agent.",
            f"Task: {task}\n\nTools:\n{tool_descriptions}",
            config_loader.get_planning_tokens(),
            return_usage=True,
        )
        if usage:
            total_prompt_tokens     += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)

        # -----------------------

        total_tokens = total_prompt_tokens + total_completion_tokens
        return {
            "solution":                             response,
            "total_rounds":                         1,
            "execution_results":                    execution_results,
            "planning_json_compliance":             1.0,
            "accumulated_information":              "",
            "accumulated_information_uncompressed": "",
            "total_output_tokens":                  total_completion_tokens,
            "total_prompt_tokens":                  total_prompt_tokens,
            "total_tokens":                         total_tokens,
        }
```

### 2. What the runner provides

**`self.llm` — LLM provider**

```python
response, usage = await self.llm.get_completion(
    system_prompt,       # str
    user_prompt,         # str
    max_tokens,          # int — use config_loader.get_planning_tokens()
    return_usage=True,   # returns (str, dict) instead of str
    temperature=0.7,     # optional, ignored for o-series models
)
# usage keys: prompt_tokens, completion_tokens, total_tokens
```

**`self.all_tools` — available tool schemas**

A dict keyed by `"ServerName:tool_name"`. Each value:

```python
{
    "server":      "BioMCP",
    "description": "Retrieve gene annotation from NCBI...",
    "parameters":  { ... }   # JSON schema
}
```

Format all tools for an LLM prompt:

```python
from runtime.mcp_modules.connector import MCPConnector
tool_descriptions = MCPConnector.format_tools_for_prompt(self.all_tools)
```

**`self.server_manager` — tool execution** (only needed for execution agents)

```python
result = await self.server_manager.call_tool(
    "BioMCP:gene_getter",             # "ServerName:tool_name"
    {"gene_id_or_symbol": "BRAF"}     # parameters dict
)
```

### 3. Required return fields

```python
{
    # Used by the evaluator
    "solution":                             str,    # agent's final answer or plan
    "total_rounds":                         int,    # iterations performed
    "execution_results":                    list,   # tool calls made (see below)
    "planning_json_compliance":             float,  # 0.0–1.0; set 1.0 if not applicable
    "accumulated_information":              str,
    "accumulated_information_uncompressed": str,

    # Token usage
    "total_output_tokens":                  int,
    "total_prompt_tokens":                  int,
    "total_tokens":                         int,
}
```

Each entry in `execution_results`:

```python
{
    "tool_name":   "BioMCP:gene_getter",
    "tool_input":  {"gene_id_or_symbol": "BRAF"},
    "tool_output": { ... },
    "round":       1,
}
```

### 4. Register the agent in `runner.py`

Open `runtime/benchmark/runner.py` and find the executor construction (~line 490). Either swap it directly:

```python
from planning.agents.your_agent import YourAgent
executor = YourAgent(llm_provider, conn_mgr.server_manager)
```

Or add it to the agent registry for CLI-switchable agents:

```python
AGENT_REGISTRY = {
    "baseline":   ("runtime.agents.executor",           "TaskExecutor",      True),
    "langgraph":  ("planning.agents.langgraph_executor", "LangGraphExecutor", False),
    "your_agent": ("planning.agents.your_agent",         "YourAgent",         False),
}
```

Then at the executor construction site:

```python
import importlib

module_path, class_name, needs_concurrent = AGENT_REGISTRY[self.agent]
module = importlib.import_module(module_path)
AgentClass = getattr(module, class_name)

executor = AgentClass(llm_provider, conn_mgr.server_manager) \
    if not needs_concurrent \
    else AgentClass(llm_provider, conn_mgr.server_manager, self.concurrent_summarization)
```

Pass `--agent your_agent` from the CLI and the correct class is loaded at runtime without further code changes.

---

## LLM Configuration

All models are accessed through [OpenRouter](https://openrouter.ai). Set the `OPENROUTER_API_KEY` environment variable and all configured models become available automatically.

To add a model, append an entry to the `openrouter_models` list in `llm/factory.py`:

```python
("your-short-name", "provider/model-id"),   # exact ID from openrouter.ai/models
```

The short name is what you pass to `--model`; the second value is the OpenRouter model ID.

---

## Tool Environment (MCP Servers)

Tasks are drawn from an environment of 28 MCP servers spanning diverse domains. Each task specifies which servers are required plus a set of distraction servers, testing the agent's ability to identify relevant tools amid noise.

| Server | Domain |
|--------|--------|
| BioMCP | Biomedical research, genes, clinical trials |
| Paper Search | Academic paper search |
| Wikipedia | Encyclopaedia content |
| Weather Data | Forecasts and meteorology |
| NASA Data | Space missions, astronomical data |
| Metropolitan Museum | Art collection database |
| Game Trends | Gaming industry statistics |
| Math MCP | Mathematical calculations |
| Time MCP | Date/time utilities (always resident) |
| Medical Calculator | Clinical calculation tools |
| Reddit | Social media discussions |
| Movie Recommender | Film recommendations and metadata |
| National Parks | US National Parks information |
| Google Maps | Geocoding and location services |
| Hugging Face | ML models and datasets |
| OKX Exchange | Cryptocurrency market data |
| DEX Paprika | DeFi analytics |
| Scientific Computing | Advanced maths and data analysis |
| Unit Converter | Measurement conversions |
| OpenAPI Explorer | API specification exploration |
| OSINT Intelligence | Open source intelligence gathering |
| FruityVice | Fruit nutrition data |
| Car Price Evaluator | Vehicle valuation |
| Context7 | Project context and documentation |
| NixOS | Package management tools |
| Huge Icons | Icon search and design resources |
| Call for Papers | Academic conference submissions |
| Bibliomantic | I Ching divination |

Full server source, documentation, and startup commands are in `mcp_servers/`.

---

## Task Format

Each task entry in `tasks/` has the following structure:

```json
{
  "task_id": "paper_search_biomcp_000",
  "task_description": "1) Use BioMCP:gene_getter to... 2) Search...",
  "fuzzy_description": "I'm researching BRAF V600E in melanoma...",
  "dependency_analysis": "Sequential chain: gene_getter → variant_searcher...",
  "distraction_servers": ["Wikipedia", "Weather Data", "Math MCP"]
}
```

- `fuzzy_description` — what the agent sees (natural language)
- `task_description` — step-by-step ground truth used only by the evaluator
- `dependency_analysis` — reference DAG structure used for planning evaluation

---

## Based On

This project is a fork of [MCP-Bench](https://arxiv.org/abs/2508.20453) (Wang et al., 2025), a benchmark for evaluating LLM tool-use via the Model Context Protocol.

```bibtex
@article{wang2025mcpbench,
  title={MCP-Bench: Benchmarking Tool-Using LLM Agents with Complex Real-World Tasks via MCP Servers},
  author={Wang, Zhenting and Chang, Qi and Patel, Hemani and Biju, Shashank and Wu, Cheng-En and Liu, Quan and Ding, Aolin and Rezazadeh, Alireza and Shah, Ankit and Bao, Yujia and Siow, Eugene},
  journal={arXiv preprint arXiv:2508.20453},
  year={2025}
}
```
