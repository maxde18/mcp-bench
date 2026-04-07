# mcp-bench — CLAUDE.md

## Project Overview

Research codebase studying the **planning phase of LLM tool-use agents**. Rather than measuring end-to-end task success, the focus is on whether and how an agent's **tool dependency graph (DAG)** changes across models, prompts, temperatures, and runs. Built on top of MCP-Bench (Wang et al., 2025).

The thesis contribution lives in three directories: `planning/`, `dag/`, and `analysis/`. Everything else is inherited MCP-Bench infrastructure.

---

## Stack

- **Python 3.10+**, conda env `mcpbench`
- **LLM access**: OpenRouter API (all models, no direct provider SDKs)
- **Agent orchestration**: LangGraph + `langchain-openrouter`
- **DAG validation**: Pydantic v2 + custom cycle detection (Kahn's algorithm)
- **MCP protocol**: `mcp[cli]` ≥ 1.9.0
- **Config**: `config/benchmark_config.yaml` with env var overrides
- **Environment**: `OPENROUTER_API_KEY` in `.env` (gitignored)

---

## Key Entry Points

### Primary: `run_planning.py`
Planning-only benchmark — generates DAGs without executing tools.

```bash
# Basic run (single-server tasks)
python run_planning.py \
    --tasks tasks/mcpbench_tasks_single_runner_format.json \
    --model claude-sonnet-4

# Multi-server with repetitions and temperature sweep
python run_planning.py \
    --tasks tasks/mcpbench_tasks_multi_2server_runner_format.json \
    --model qwen-3-32b \
    --repetitions 3 \
    --temperatures 0.0 0.5 1.0 1.5

# Fuzzy prompt variations (variation 0 = original, 1+ = re-synthesized)
python run_planning.py \
    --tasks tasks/mcpbench_tasks_multi_2server_runner_format.json \
    --model claude-sonnet-4 \
    --variations 3

# Pass pre-generated fuzzy prompts directly
python run_planning.py --fuzzy-tasks my_fuzzy_prompts.json --model qwen-3-32b

# Native tools mode (passes tools via OpenAI `tools` field, not prompt text)
python run_planning.py --tasks ... --model <fine-tuned-model> --native-tools

# Exclude distraction servers from the prompt (only task-relevant tools shown)
python run_planning.py --tasks ... --model claude-sonnet-4 --no-distraction-servers
```

Results saved to `results/planning/<model>/<timestamp>/`.

### Secondary: `run_benchmark.py`
Full plan-then-execute benchmark with LLM-as-judge evaluation. Rarely used in current experimental work.

```bash
python run_benchmark.py --models gpt-oss-20b
python run_benchmark.py --models gpt-oss-20b \
    --tasks-file tasks/mcpbench_tasks_multi_2server_runner_format.json
```

---

## Available Models (`--model` argument)

Defined in `llm/factory.py`. Short names map to full OpenRouter IDs:

| Short name | OpenRouter ID |
|---|---|
| `claude-sonnet-4` | `anthropic/claude-sonnet-4` |
| `qwen-3-32b` | `qwen/qwen3-32b` |
| `gpt-5-mini` | `openai/gpt-5-mini` |
| `gpt-oss-20b` / `gpt-oss-120b` | `openai/gpt-oss-*` |
| `deepseek-r1-0528` | `deepseek/deepseek-r1-0528` |
| `gemini-2.5-pro` / `gemini-2.5-flash-lite` | `google/gemini-2.5-*` |
| `minimax-m1` / `minimax-m2.7` | `minimax/minimax-*` |
| `qwq-32b` | `qwen/qwq-32b` |
| `kimi-k2` | `moonshotai/kimi-k2` |

To add a new model: add an entry to the `openrouter_models` list in `llm/factory.py`.

**Special cases:**
- MiniMax models have no native tools API — tools are embedded in the prompt automatically
- OpenAI reasoning models (`o1`, `o3`, `o4`) reject a `temperature` parameter — handled automatically

---

## Architecture

### Planning Pipeline

```
run_planning.py
  └─ PlanOnlyExecutor (planning/agents/plan_only_executor.py)
       ├─ MCPConnector → fetches tool schemas from MCP servers
       ├─ few_shot_examples.py → injects example DAGs into prompt
       ├─ ChatOpenRouter (LangGraph node) → generates structured JSON
       └─ validate_dag() (planning/validation.py) → Pydantic + cycle check
```

### DAG Node Format

```json
{
  "id": "1",
  "tool": "ServerName:tool_function",
  "parameters": {"param": "value", "ref": "{2.result}"},
  "depends_on": ["2"],
  "description": "What this step does",
  "filter": "(optional) post-processing filter",
  "condition": "(optional) conditional execution"
}
```

`{N.field}` placeholders are resolved at execution time using node N's output.

### Agent Interface Contract

All agents return this dict:
```python
{
    "solution": str,
    "total_rounds": int,
    "execution_results": list[dict],
    "planning_json_compliance": float,
    "accumulated_information": str,
    "total_output_tokens": int,
    "total_prompt_tokens": int,
    "total_tokens": int,
}
```

### LLM Access Layers

- `llm/factory.py` — `ModelConfig` + `LLMFactory`: maps short names → OpenRouter IDs, creates providers
- `llm/provider.py` — `LLMProvider`: async wrapper for raw `AsyncOpenAI` completions (used by baseline executor)
- `langchain_openrouter.ChatOpenRouter` — used by `PlanOnlyExecutor` and `LangGraphExecutor` for LangGraph-native tool binding and structured output

### MCP Infrastructure

- `mcp_infra/connector.py` — single-server connection + tool schema discovery
- `mcp_infra/server_manager.py` — non-persistent multi-server manager
- `mcp_infra/server_manager_persistent.py` — stateful persistent connections (used for execution)
- `mcp_infra/tool_cache.py` — persistent hash-keyed cache for tool call results
- `mcp_servers/commands.json` — startup commands for all 28 MCP servers

---

## Configuration

**`config/benchmark_config.yaml`** — all runtime parameters.

Key settings:
- `llm.planning_tokens` — max tokens for planning (default: 12000)
- `execution.max_rounds` — max agent execution rounds (default: 20)
- `benchmark.distraction_server_count` — servers added as noise (default: 10)
- `cache.enabled` — tool result caching on/off

Access in code via `config/config_loader.py` singleton:
```python
import config.config_loader as config_loader
config_loader.get_planning_tokens()
```

Environment variable overrides: `BENCHMARK_<SECTION>_<KEY>=value`

---

## Task Files

Pre-generated benchmark tasks in `tasks/`:

| File | Coverage |
|---|---|
| `mcpbench_tasks_single_runner_format.json` | ~30 single-server tasks |
| `mcpbench_tasks_multi_2server_runner_format.json` | ~100 two-server tasks |
| `mcpbench_tasks_multi_3server_runner_format.json` | ~60 three-server tasks |
| `single_task_3server_runner_format.json` | development/test subset |

Task format:
```json
{
  "task_id": "paper_search_biomcp_000",
  "server_name": "Paper Search+BioMCP",
  "fuzzy_description": "I need to understand...",
  "distraction_servers": ["Wikipedia", "Weather Data"],
  "ground_truth": {
    "task_description": "...",
    "dependency_analysis": "..."
  }
}
```

---

## Analysis

- **Notebook**: `analysis/mcp_bench_graph_analysis.ipynb` — primary DAG comparison analysis
- **Network builder**: `analysis/build_network.py` — NetworkX DAG visualization
- **Collected plans**: `analysis/agent_plans_json/` — JSON outputs for manual review
- **Figures**: `analysis/figures/` — generated plots (consistency, MDS scatter, heatmaps, etc.)

---

## Setup

```bash
conda create -n mcpbench python=3.10
conda activate mcpbench
cd mcp_servers && bash ./install.sh && cd ..

# Create .env with:
export OPENROUTER_API_KEY="your_key"
source .env

# Verify connectivity (optional)
python utils/collect_mcp_info.py  # expected: 28/28 servers connected
```

Per-server API keys go in `mcp_servers/api_key/` (NPS_API_KEY, NASA_API_KEY, HF_TOKEN, etc.).

---

## Key Files to Know

| File | Purpose |
|---|---|
| `run_planning.py` | Primary experiment runner |
| `planning/agents/plan_only_executor.py` | Core planning agent (LangGraph + structured output) |
| `planning/agents/few_shot_examples.py` | In-context DAG examples injected into prompts |
| `planning/validation.py` | Pydantic + cycle-detection DAG validation |
| `llm/factory.py` | Model registry and LLM client creation |
| `llm/provider.py` | Raw async OpenAI-compatible completion wrapper |
| `config/benchmark_config.yaml` | All configurable parameters |
| `mcp_infra/connector.py` | MCP server connection and tool discovery |
| `mcp_servers/commands.json` | Startup commands for all 28 MCP servers |

---

## Documentation

- `README.md` — quick start and overview
- `THESIS_README.md` — deep technical reference (15 sections)
- `AGENT_INTEGRATION.md` — how to implement a new agent
- `planning/agents/PLAN_ONLY_EXECUTOR.md` — PlanOnlyExecutor internals

---

## No Tests

There is no test suite. Validation is embedded in `planning/validation.py` (DAG schema + cycle check). Use `utils/collect_mcp_info.py` to verify MCP server connectivity.

---

## Documentation Updates

When making changes to the codebase, update the relevant documentation to reflect those changes:

- **New or renamed CLI flags** in `run_planning.py` or `run_benchmark.py` → update the usage examples in the **Key Entry Points** section of this file and in `README.md`
- **New or removed models** in `llm/factory.py` → update the **Available Models** table in this file
- **Changes to the DAG node schema** → update the **DAG Node Format** block in this file and the relevant sections in `THESIS_README.md`
- **Changes to the agent interface contract** (return dict keys/types) → update the **Agent Interface Contract** block in this file and `AGENT_INTEGRATION.md`
- **New configuration keys** in `config/benchmark_config.yaml` → update the **Configuration** section of this file
- **New or removed MCP infrastructure files** → update the **MCP Infrastructure** list and the **Key Files to Know** table
- **Changes to `PlanOnlyExecutor`** (prompt structure, LangGraph nodes, structured output schema) → update `planning/agents/PLAN_ONLY_EXECUTOR.md`
- **New task files** added to `tasks/` → update the **Task Files** table in this file
- **Changes to the planning pipeline flow** → update the **Planning Pipeline** diagram in this file and the relevant sections in `THESIS_README.md`

When in doubt: if a section of any doc file describes the thing you just changed, update that section.
