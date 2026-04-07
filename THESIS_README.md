# MCP-Bench: Technical Reference

This document provides a progressive technical explanation of the MCP-Bench codebase — starting from a high-level overview and becoming increasingly detailed. It covers both the original benchmark framework and the extensions added for thesis experimentation.

---

## 1. What Is MCP-Bench?

MCP-Bench is a benchmarking framework that measures how well an LLM agent can use external tools to complete real-world tasks. It does this by connecting an LLM to a set of running software services (called MCP servers) that expose callable tools, then asking the LLM to complete a task using those tools, and finally scoring the result using an LLM-as-judge.

The benchmark is built on the **Model Context Protocol (MCP)** — a standardised protocol that allows any LLM agent to connect to and call tools on any compliant server, regardless of what that server does. This makes the benchmark extensible: adding a new tool domain means adding a new MCP server.

---

## 2. The Benchmark Pipeline (High Level)

At the highest level, a benchmark run is a five-step pipeline:

```
1. TASKS       Pre-generated task files describe what the agent must do
      │
      ▼
2. SERVERS     MCP servers are started and their tool schemas are discovered
      │
      ▼
3. AGENT       The agent receives the task and tool descriptions,
               plans and executes tool calls, and produces a final answer
      │
      ▼
4. EVALUATOR   An LLM judge scores the agent's answer and tool usage
      │
      ▼
5. RESULTS     Scores are aggregated and saved to a JSON file
```

Each of these steps maps directly to a module in the codebase.

---

## 3. Key Concepts

### MCP Servers
A server is an external process (Node.js, Python, or Rust) that exposes a set of callable functions (tools) over a standard protocol. Examples include a biomedical database server, a weather API wrapper, a Wikipedia search server, etc. There are 28 servers in total.

### Tool Schema
When a server starts, it advertises its tools with names, descriptions, and parameter schemas. The agent reads these descriptions to understand what is available before deciding what to call.

### Task Description vs Fuzzy Description
Every task has two forms:
- **task_description**: A precise, numbered step-by-step specification of exactly which tools to call in what order. This is the ground truth.
- **fuzzy_description**: A natural-language version written as a user request, as if a person asked the agent conversationally. This is what the agent actually sees.

The fuzzy description is what is passed to the agent. The detailed description is used by the evaluator as a reference for scoring.

### Dependency Analysis
Each task also includes a `dependency_analysis` field — a prose description of which tool calls depend on each other, which can run in parallel, and what conditional branches exist. This is used as additional context by the evaluator.

### Distraction Servers
When an agent runs a task, it is connected not only to the servers needed for that task but also to several unrelated servers (distractors). This tests whether the agent can identify which tools are relevant and avoid calling irrelevant ones.

---

## 4. Repository Structure

```
mcp-bench/
│
├── run_benchmark.py               # Entry point for the full benchmark
├── run_planning_benchmark.py      # Entry point for planning-only experiments [ADDED]
│
├── tasks/                         # Pre-generated task files (JSON)
│   ├── mcpbench_tasks_single_runner_format.json
│   ├── mcpbench_tasks_multi_2server_runner_format.json
│   └── mcpbench_tasks_multi_3server_runner_format.json
│
├── agent/                         # Agent implementations
│   ├── executor.py                # Baseline: multi-round reactive agent
│   ├── execution_context.py       # Retry state tracker for baseline agent
│   ├── langgraph_executor.py      # LangGraph plan-then-execute agent [ADDED]
│   └── plan_only_executor.py      # Planning-only agent (no tool execution) [ADDED]
│
├── benchmark/                     # Evaluation framework
│   ├── runner.py                  # Orchestrates task loading, agent, evaluation
│   ├── evaluator.py               # LLM-as-judge scoring
│   ├── results_aggregator.py      # Statistical aggregation
│   └── results_formatter.py       # Display and reporting
│
├── llm/                           # LLM provider abstraction
│   ├── factory.py                 # Constructs LLM clients from env vars
│   └── provider.py                # Unified interface for all LLM calls
│
├── mcp_modules/                   # MCP connection management
│   ├── connector.py               # Connects to a single MCP server
│   ├── server_manager_persistent.py  # Manages multiple persistent connections
│   └── tool_cache.py              # Caches tool call results
│
├── synthesis/                     # Task generation pipeline
│   ├── task_synthesis.py          # Core LLM-driven task generator
│   ├── benchmark_generator.py     # Batch generator for all server combinations
│   ├── generate_benchmark_tasks.py  # CLI script
│   └── split_combinations/
│       ├── mcp_2server_combinations.json
│       ├── mcp_3server_combinations.json
│       └── paper_search_biomcp_only.json   # [ADDED]
│
├── config/
│   ├── benchmark_config.yaml      # All configurable parameters
│   └── config_loader.py           # Reads config and exposes helper functions
│
├── utils/
│   ├── collect_mcp_info.py        # Discovers tool schemas from all servers
│   └── local_server_config.py     # Loads server startup commands
│
└── mcp_servers/                   # 28 MCP server implementations
    ├── commands.json              # Startup commands for every server
    └── api_key                    # External API keys for individual servers
```

---

## 5. The Task Files

**Location:** `tasks/`

Three task files ship with the benchmark:
- `mcpbench_tasks_single_runner_format.json` — tasks that use one server
- `mcpbench_tasks_multi_2server_runner_format.json` — tasks requiring two servers
- `mcpbench_tasks_multi_3server_runner_format.json` — tasks requiring three servers

### Task Entry Structure

```json
{
  "task_id": "paper_search_biomcp_000",
  "task_description": "1) Use BioMCP:gene_getter... 2) Search...",
  "fuzzy_description": "I'm researching BRAF V600E in melanoma...",
  "dependency_analysis": "Sequential chain: gene_getter → variant_searcher...",
  "distraction_servers": ["Wikipedia", "Weather Data", "Math MCP", ...]
}
```

The file is structured as:
```json
{
  "generation_info": { ... },
  "server_tasks": [
    {
      "server_name": "Paper Search+BioMCP",
      "servers": ["Paper Search", "BioMCP"],
      "tasks": [ { ...task entry... }, ... ]
    }
  ]
}
```

The `server_name` field (e.g. `"Paper Search+BioMCP"`) is a `+`-joined string telling the runner which MCP servers to start for that task.

---

## 6. The LLM Layer

### `llm/factory.py` — `LLMFactory`

Reads environment variables to discover which API keys are available and builds a dictionary of `ModelConfig` objects — one per model. Supports three provider types:

| Provider type | Client used | Required env vars |
|---|---|---|
| `azure` | `AsyncAzureOpenAI` | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` |
| `openrouter` | `AsyncOpenAI` (OpenRouter base URL) | `OPENROUTER_API_KEY` |
| `openai_compatible` | `AsyncOpenAI` | Model-specific `_API_KEY` and `_BASE_URL` vars |

**Key methods:**
- `LLMFactory.get_model_configs()` — returns the dict of all available models
- `LLMFactory.create_llm_provider(model_config)` — constructs the appropriate async client and wraps it in an `LLMProvider`

### `llm/provider.py` — `LLMProvider`

A unified wrapper around any async OpenAI-compatible client. All LLM calls across the entire codebase go through this single class.

**Key method:**
```python
await provider.get_completion(system_prompt, user_prompt, max_tokens, return_usage=False)
```
Returns either a string (the response) or a `(string, usage_dict)` tuple when `return_usage=True`. Handles retries, token limit errors, and JSON repair internally.

`clean_and_parse_json(raw_string)` — strips markdown fences and parses JSON, used everywhere the LLM is expected to return structured data.

---

## 7. The MCP Layer

### `mcp_modules/connector.py` — `MCPConnector`

Manages the connection to a single MCP server process. Handles both stdio (the server runs as a subprocess) and HTTP transport types. Responsible for starting the process, establishing the MCP session, and discovering available tools.

**Key static method:**
```python
MCPConnector.format_tools_for_prompt(all_tools: dict) -> str
```
Converts the tool schema dictionary into a formatted string suitable for inclusion in an LLM prompt, listing each tool's name, server, description, and parameters.

### `mcp_modules/server_manager_persistent.py` — `PersistentMultiServerManager`

Manages connections to multiple servers simultaneously using persistent sessions (sessions stay open for the lifetime of a task rather than reconnecting per tool call). This preserves server-side state across calls, which matters for stateful servers.

**Key method:**
```python
await manager.call_tool(tool_name: str, parameters: dict) -> Any
```
`tool_name` is in the format `"ServerName:tool_name"` (e.g. `"BioMCP:gene_getter"`). Results are optionally cached via `tool_cache.py`.

`manager.all_tools` — a dict of all discovered tools across all connected servers, keyed by `"ServerName:tool_name"`.

### `mcp_modules/tool_cache.py`

A persistent cache for tool call results. When the same tool is called with the same parameters, the cached result is returned instead of calling the server again. Controlled by `benchmark_config.yaml`.

---

## 8. The Benchmark Runner

### `benchmark/runner.py` — `ConnectionManager`

An async context manager that starts all required MCP server processes for a single task and tears them down cleanly when the task is done.

```python
async with ConnectionManager(server_configs, filter_problematic_tools) as conn_mgr:
    tools = conn_mgr.all_tools        # dict of all discovered tools
    manager = conn_mgr.server_manager # PersistentMultiServerManager instance
```

### `benchmark/runner.py` — `BenchmarkRunner`

The main orchestrator. Its key responsibilities are:

1. **`load_tasks()`** — reads the task JSON file and flattens it into a list of task dicts, handling all three JSON formats (single, multi, combination)

2. **`load_server_configs()`** — reads `mcp_servers/commands.json` via `LocalServerConfigLoader` to get startup commands for every server

3. **`_prepare_server_configs(server_name, servers_info, task_data)`** — resolves which server processes to start for a task:
   - Parses `+`-joined server names into a list of required servers
   - Adds the resident "Time MCP" server (always present)
   - Adds distraction servers (from the task's predefined list or random selection)

4. **`execute_single_task_with_model(task_info, servers_info, model_name, llm_provider)`** — the core execution method:
   - Calls `_prepare_task_execution` to extract the fuzzy description and task metadata
   - Calls `_prepare_server_configs` to build the server list
   - Opens a `ConnectionManager` context
   - Instantiates an agent executor and calls `executor.execute(task_description)`
   - Passes the result to `_evaluate_task_result`
   - Retries up to `max_retries` times on timeout or connection failure

5. **`_run_single_file_benchmark_core()`** — outer loop that iterates over all models and all tasks, calling `execute_single_task_with_model` for each combination

---

## 9. The Baseline Agent

### `agent/executor.py` — `TaskExecutor`

The original agent implementation. It is a **multi-round reactive agent**: it does not generate a complete plan upfront. Instead it plans one round at a time, executes those tool calls, accumulates the results, and decides whether to continue.

**Constructor:**
```python
TaskExecutor(llm_provider, server_manager, concurrent_summarization=False)
```

**`execute(task) -> dict`** — the main entry point. Runs the round loop and returns:

```python
{
    "solution": str,                    # final answer
    "total_rounds": int,                # number of rounds completed
    "execution_results": List[dict],    # every tool call and its output
    "planning_json_compliance": float,  # fraction of planned tools with valid format
    "accumulated_information": str,     # compressed running context
    "accumulated_information_uncompressed": str,
    "total_output_tokens": int,
    "total_prompt_tokens": int,
    "total_tokens": int
}
```

**Round loop internals:**

Each round calls `_plan_next_actions()` which:
1. Builds a prompt containing the task, all tool descriptions, and accumulated information from previous rounds
2. Asks the LLM to return a JSON object: `{ "should_continue": bool, "reasoning": str, "planned_tools": [...] }`
3. Executes all planned tools in parallel via `asyncio.gather` (except tools in the sequential-only list from config)
4. Appends results to `accumulated_information`

The round continues until `should_continue` is `false` or `max_execution_rounds` (from config) is reached. A final `_synthesize_final_solution()` call produces the answer string.

### `agent/execution_context.py` — `ExecutionContext`

A dataclass that tracks retry state within a single planning call. Each call to `_plan_next_actions` creates a fresh `ExecutionContext` which manages:

- **Token reductions** — if the LLM hits a token limit, `apply_token_reduction()` scales down `max_tokens` using configured factors (0.9, 0.8, 0.7)
- **Format fixes** — if the LLM returns malformed JSON, `can_fix_format()` allows up to 5 attempts to correct it
- **Round retries** — up to 3 retries of the entire planning prompt
- **Task retries** — up to 3 full restarts of the planning attempt
- **Compression** — if all else fails, `mark_compressed()` triggers summarisation of `accumulated_information` to reduce prompt size

---

## 10. The Evaluator

### `benchmark/evaluator.py` — `LLMJudge` and `TaskEvaluator`

After the agent returns its result, the evaluator scores it across six sub-dimensions in three groups:

| Group | Sub-dimensions |
|---|---|
| Task Completion | Task Fulfillment, Grounding |
| Tool Usage | Tool Appropriateness, Parameter Accuracy |
| Planning Effectiveness | Dependency Awareness, Parallelism and Efficiency |

Each sub-dimension is scored 1–10 with anchored rubrics (e.g. "9-10: 90-100% of requirements perfectly completed").

The judge prompt includes:
- The fuzzy task description the agent saw
- The concrete task description (as a reference the agent did not see)
- The dependency analysis
- The agent's final answer
- A summary of every tool call made and its output

The evaluator also performs **rule-based schema checking** — it validates that specific required fields appear in the agent's output, giving a binary compliance score independent of the LLM judge.

If `enable_judge_stability` is set in config, the judge runs multiple times and the scores are averaged to reduce variance.

---

## 11. Task Synthesis

The task files that ship with the benchmark were generated by the synthesis pipeline. This is only needed if you want to create new tasks.

### `synthesis/task_synthesis.py` — `TaskSynthesizer`

Uses an LLM to generate tasks for a given set of servers and their tools. For each server combination it:
1. Generates a detailed step-by-step `task_description` specifying exact tool calls
2. Generates a `fuzzy_description` — a conversational version of the same task
3. Generates a `dependency_analysis` describing the tool call graph
4. Evaluates the task for solvability (≥9.0) and utility (≥5.0), discarding tasks that don't meet the thresholds

### `synthesis/benchmark_generator.py` — `BenchmarkTaskGenerator`

Iterates over server combinations and calls `TaskSynthesizer` for each one. The `tasks_per_server` parameter controls how many distinct tasks are generated per combination.

### `synthesis/generate_benchmark_tasks.py`

CLI script wrapping `BenchmarkTaskGenerator`:

```bash
python synthesis/generate_benchmark_tasks.py \
    --mode multi \
    --combinations-file synthesis/split_combinations/mcp_2server_combinations.json \
    --tasks-per-combination 10 \
    --output tasks/my_tasks.json
```

A pre-made combinations file for the Paper Search + BioMCP pair is provided at:
`synthesis/split_combinations/paper_search_biomcp_only.json`

---

## 12. Configuration

### `config/benchmark_config.yaml`

Central configuration file. Key settings:

| Setting | Default | Effect |
|---|---|---|
| `max_execution_rounds` | 20 | Maximum rounds per task in the baseline agent |
| `task_timeout` | 5000s | Hard timeout per task |
| `max_retries` | 3 | Connection/execution retries |
| `planning_tokens` | 12000 | Max tokens for planning LLM calls |
| `summarization_max_tokens` | 10000 | Max tokens for synthesis LLM calls |
| `distraction_servers_count` | 10 | How many distraction servers to add |
| `use_fuzzy_descriptions` | true | Whether to show agents the fuzzy version |
| `enable_cache` | true | Whether to cache tool call results |

### `config/config_loader.py` — `BenchmarkConfig`

A singleton that loads `benchmark_config.yaml` once and exposes named accessor functions used throughout the codebase (e.g. `config_loader.get_planning_tokens()`, `config_loader.get_max_execution_rounds()`). Values can be overridden with environment variables.

---

## 13. Extensions

The following files were added for thesis experimentation. They build on the existing infrastructure without modifying it (except two lines in `runner.py`).

---

### 13.1 `agent/langgraph_executor.py` — `LangGraphExecutor`

A replacement for `TaskExecutor` that uses LangGraph to implement a **plan-then-execute agent**. Unlike the baseline which replans every round, this agent generates a complete dependency graph upfront and then executes it.

**Constructor:**
```python
LangGraphExecutor(llm_provider, server_manager)
```
Takes the same arguments as `TaskExecutor` and exposes the same `execute(task)` method, making it a drop-in replacement in `runner.py`.

**LangGraph pipeline:**

```
planner node → executor node → synthesizer node → END
```

The state passed between nodes is:

```python
class AgentState(TypedDict):
    task: str
    dependency_graph: Dict        # saved plan (output of planner)
    execution_results: List[Dict] # tool calls and outputs
    accumulated_information: str  # concatenated tool outputs
    final_answer: str
    total_rounds: int
    total_output_tokens: int
    total_prompt_tokens: int
    total_tokens: int
```

**Planner node (`_planner_node`)**

Makes a single LLM call with the task and all tool descriptions. The prompt instructs the model to return a JSON dependency graph:

```json
{
  "nodes": [
    {
      "id": "1",
      "tool": "BioMCP:gene_getter",
      "parameters": {"gene_id_or_symbol": "BRAF"},
      "depends_on": [],
      "description": "Get BRAF gene annotation"
    },
    {
      "id": "2",
      "tool": "BioMCP:variant_searcher",
      "parameters": {"gene": "BRAF", "hgvsp": "{1.symbol}"},
      "depends_on": ["1"],
      "description": "Search for V600E variant"
    }
  ]
}
```

The `{node_id.field}` syntax is a placeholder meaning "use the value of `field` from the output of node `node_id`". This is resolved at runtime.

The dependency graph is stored in the agent state and included in the final result dict under the key `"dependency_graph"`.

**Executor node (`_executor_node`)**

Performs a topological traversal of the DAG:
1. Finds all nodes whose `depends_on` list is fully satisfied
2. Dispatches them in parallel via `asyncio.gather`
3. Stores their outputs in `node_outputs`
4. Repeats until all nodes are complete or a cycle is detected

Before each tool call, `_resolve_parameters` replaces `{node_id.field}` placeholders with actual values from `node_outputs`. The `total_rounds` counter increments once per DAG level (i.e. per batch of parallel calls).

**Synthesizer node (`_synthesizer_node`)**

Makes a final LLM call with the accumulated tool outputs to produce a coherent final answer.

**Integration with `runner.py`**

Two lines were changed:
```python
# Added import:
from agent.langgraph_executor import LangGraphExecutor

# Changed constructor call from:
executor = TaskExecutor(llm_provider, conn_mgr.server_manager, self.concurrent_summarization)
# To:
executor = LangGraphExecutor(llm_provider, conn_mgr.server_manager)
```

The rest of the runner, evaluator, and results pipeline is unchanged. The `dependency_graph` key in the result dict is extra data that passes through untouched.

---

### 13.2 `agent/plan_only_executor.py` — `PlanOnlyExecutor`

A minimal agent that only performs the planning phase. No tools are ever called. Used by the planning benchmark runner to isolate and study planning quality independently.

**Constructor:**
```python
PlanOnlyExecutor(llm_provider, all_tools: dict)
```
Takes `all_tools` directly (not `server_manager`) because no tool calls are made — only the tool descriptions are needed.

**`execute(task) -> dict`**

Makes a single LLM call using the same dependency graph prompt as the LangGraph planner. Returns:

```python
{
    "dependency_graph": { "nodes": [...] },
    "tool_descriptions": str,   # what the agent was shown
    "prompt_tokens": int,
    "completion_tokens": int,
    "total_tokens": int
}
```

---

### 13.3 `run_planning_benchmark.py`

A standalone runner for planning-only experiments. It connects to MCP servers to get tool descriptions, runs `PlanOnlyExecutor`, and saves the generated plans without executing any tools.

**Two input modes** (mutually exclusive):

**Mode 1 — `--tasks <file>`**

Loads tasks from an existing benchmark task JSON file. Extracts the fuzzy description from each task and also captures the ground truth (`task_description`, `dependency_analysis`) for later comparison.

**Mode 2 — `--fuzzy-tasks <file>`**

Loads pre-generated fuzzy prompts from a separate JSON file with this format:

```json
{
  "tasks": [
    {
      "task_id": "paper_search_biomcp_000",
      "server_name": "Paper Search+BioMCP",
      "fuzzy_description": "I need to understand...",
      "distraction_servers": ["Wikipedia", "Weather Data"]
    }
  ]
}
```

`distraction_servers` is optional. If omitted, distraction servers are selected randomly.

**Key functions:**

- `load_fuzzy_tasks(file)` — reads a fuzzy tasks JSON and normalises each entry into the internal task shape used by the main loop
- `run(tasks_file, fuzzy_tasks_file, output_file, model_name, task_limit)` — async main function. Reuses `BenchmarkRunner._prepare_server_configs()` and `ConnectionManager` for server resolution and connection, then runs `PlanOnlyExecutor` per task
- `main()` — CLI argument parser, defaults to `claude-sonnet-4` via OpenRouter

**Output format:**

```json
{
  "run_timestamp": "2026-03-18T...",
  "model": "claude-sonnet-4",
  "source": "tasks/mcpbench_tasks_multi_2server_runner_format.json",
  "total_tasks": 10,
  "successful": 9,
  "failed": 1,
  "results": [
    {
      "task_id": "paper_search_biomcp_000",
      "server_name": "Paper Search+BioMCP",
      "status": "success",
      "fuzzy_description": "...",
      "ground_truth": {
        "task_description": "...",
        "dependency_analysis": "..."
      },
      "generated_plan": { "nodes": [...] },
      "token_usage": {
        "prompt_tokens": 4200,
        "completion_tokens": 800,
        "total_tokens": 5000
      }
    }
  ]
}
```

When loaded from a fuzzy tasks file, `ground_truth` is omitted because it is not available.

**Usage:**

```bash
# From the standard task file (includes ground truth in output)
python run_planning_benchmark.py \
    --tasks tasks/mcpbench_tasks_multi_2server_runner_format.json \
    --model claude-sonnet-4

# From a pre-generated fuzzy tasks file
python run_planning_benchmark.py \
    --fuzzy-tasks my_prompts.json \
    --model gemini-2.5-pro

# Quick test with only 5 tasks
python run_planning_benchmark.py \
    --tasks tasks/mcpbench_tasks_multi_2server_runner_format.json \
    --limit 5

# Custom output file
python run_planning_benchmark.py \
    --fuzzy-tasks my_prompts.json \
    --output results/my_plans.json

# Exclude distraction servers (only task-relevant tools shown to the agent)
python run_planning.py \
    --tasks tasks/mcpbench_tasks_multi_2server_runner_format.json \
    --model claude-sonnet-4 \
    --no-distraction-servers
```

**Requires `OPENROUTER_API_KEY` in environment** (or `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` if using an Azure model name).

---

### 13.4 `synthesis/split_combinations/paper_search_biomcp_only.json`

A minimal combinations file containing only the Paper Search + BioMCP pairing. Pass this to the task generation script to generate variations for only this combination without regenerating tasks for all other server pairs.

```bash
python synthesis/generate_benchmark_tasks.py \
    --mode multi \
    --combinations-file synthesis/split_combinations/paper_plan_travel.json \
    --tasks-per-combination 10 \
    --output tasks/paper_search_biomcp_10_variations.json
```

---

## 14. Agent Interface Contract

Any agent executor — whether the baseline `TaskExecutor`, the `LangGraphExecutor`, or any future implementation — must satisfy this interface:

```python
class MyExecutor:
    def __init__(self, llm_provider: LLMProvider, server_manager: PersistentMultiServerManager):
        ...

    async def execute(self, task: str) -> dict:
        # Must return all of:
        return {
            "solution": str,                     # the agent's final answer
            "total_rounds": int,                 # execution iterations
            "execution_results": List[dict],     # tool call log
            "planning_json_compliance": float,   # 0.0–1.0
            "accumulated_information": str,
            "accumulated_information_uncompressed": str,
            "total_output_tokens": int,
            "total_prompt_tokens": int,
            "total_tokens": int,
        }
```

Each entry in `execution_results` should follow:
```python
{
    "tool_name": "ServerName:tool_name",
    "tool_input": { ... },
    "tool_output": { ... },
    "round": int
}
```

To add a new agent approach, create `agent/my_executor.py` implementing this interface. In `runner.py`, import it and replace the constructor call at line ~489. All evaluation and results infrastructure remains unchanged.

---

## 15. End-to-End Data Flow Summary

```
Task file
  └─ fuzzy_description ──────────────────────────────────► Agent
  └─ server_name ────► ConnectionManager ──► all_tools ──► Agent
  └─ distraction_servers ──► ConnectionManager

Agent (TaskExecutor / LangGraphExecutor)
  └─ all_tools ─────────────────────────► Planner (LLM call)
  └─ Planner output ────────────────────► Executor (tool calls via server_manager)
  └─ tool outputs ──────────────────────► Synthesizer (LLM call)
  └─ Returns: { solution, execution_results, dependency_graph, ... }

Evaluator
  └─ task (fuzzy) + task_description (concrete ref) + dependency_analysis
  └─ agent solution + execution_results
  └─ LLM judge ─────────────────────────► scores (1-10 per sub-dimension)
  └─ Rule-based check ──────────────────► schema compliance score

Results aggregator
  └─ Collects all task scores
  └─ Computes per-model averages
  └─ Saves to JSON output file
```
