# PlanOnlyExecutor

`planning/agents/plan_only_executor.py`

## Purpose

`PlanOnlyExecutor` is the core agent used by the planning benchmark. It takes a fuzzy task description and a set of available MCP tool descriptions and asks an LLM to produce a complete execution plan as a directed acyclic graph (DAG). No tools are ever called — the output is the plan itself.

This isolates the *planning phase* from *execution*, making it possible to evaluate how well a model understands tool dependencies and produces structured, executable plans without the noise introduced by actual tool calls.

---

## Architecture

The executor is a plain Python class (no LangGraph graph structure). It wraps a single LLM call and post-processes the result:

```
fuzzy task + tools
       |
       v
  LLM (one call)
       |
       v
  raw JSON string
       |
       v
  clean_and_parse_json()
       |
       v
  validate_dag()
       |
       v
  returned result dict
```

It is instantiated and called by `run_planning.py` inside the task loop:

```python
executor = PlanOnlyExecutor(llm_provider, conn_mgr.all_tools)
plan_result = await executor.execute(fuzzy_desc, temperature=temperature, use_native_tools=native_tools)
```

---

## Constructor

```python
PlanOnlyExecutor(llm_provider, all_tools: Dict[str, Any])
```

| Argument | Type | Description |
|---|---|---|
| `llm_provider` | `LLMProvider` | Configured LLM provider from `llm/provider.py` |
| `all_tools` | `dict` | Tool registry from `ConnectionManager.all_tools`. Keys are `"ServerName:tool_name"` strings. |

No `server_manager` is required because tools are never executed.

---

## `execute()` Parameters

```python
await executor.execute(task, temperature=None, use_native_tools=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `task` | `str` | required | The fuzzy task description shown to the agent |
| `temperature` | `float \| None` | `None` | Sampling temperature (0.0–2.0). `None` uses the model's built-in default. Silently ignored for o-series reasoning models. |
| `use_native_tools` | `bool` | `False` | Controls how tool descriptions are delivered to the model (see Tool Input Modes below) |

---

## Tool Input Modes

### Mode 1 — Prompt text (default, `use_native_tools=False`)

Tool descriptions are formatted as plain text and embedded directly into the user prompt under an `AVAILABLE TOOLS:` section. Each tool is shown with its name, server, description, and JSON input schema.

**When to use:** General-purpose models, any model that does not have a dedicated tool-calling fine-tune.

**System prompt:**
```
You are an expert planning agent that produces tool dependency graphs.
```

**User prompt structure:**
```
You are a planning agent. Given a task and a set of available tools,
produce a complete execution plan as a dependency graph (DAG).

TASK:
{task}

AVAILABLE TOOLS:
Tool: `ServerName:tool_name` (Server: ServerName)
  Description: ...
  Input Schema:
  ```json
  { ... }
  ```
...

Return ONLY valid JSON in this exact schema — no prose, no markdown fences:
{
  "nodes": [
    {
      "id": "1",
      "tool": "ServerName:tool_name",
      "parameters": {"param": "value"},
      "depends_on": [],
      "description": "one-line description of what this step does"
    }
  ]
}

Rules:
- Each node id must be a unique string ("1", "2", ...).
- "depends_on" lists the ids of nodes whose output is needed before this node runs.
- Nodes with no shared dependencies will be executed in parallel.
- When a parameter value depends on a previous node's output, write it as
  "{node_id.field}" (e.g. {"variant_id": "{2.id}"}).
- Only include tools that are actually needed to complete the task.
- The graph must be a valid DAG (no cycles).
```

The tool list is produced by `MCPConnector.format_tools_for_prompt(all_tools)`.

---

### Mode 2 — Native API tools field (`use_native_tools=True`)

Tool descriptions are passed in the structured `tools` field of the OpenAI chat completion request instead of being embedded as text. The model receives tool metadata in a machine-readable format and can leverage any tool-calling fine-tuning it has.

**When to use:** Fine-tuned tool-calling models accessed via OpenRouter or the OpenAI API directly.

**How it works under the hood:**

`MCPConnector.format_tools_for_api(all_tools)` converts each tool into the OpenAI function schema:

```json
{
  "type": "function",
  "function": {
    "name": "ServerName__tool_name",
    "description": "[ServerName] original tool description",
    "parameters": { ... input_schema ... }
  }
}
```

The colon in `ServerName:tool_name` is replaced with `__` (double underscore) because the OpenAI API requires tool names to match `^[a-zA-Z0-9_-]{1,64}$`. The original colon-separated name is preserved in the description so the model knows what to write in DAG output.

`tool_choice` is set to `"none"` — the model is aware of the tools but must respond with plain text (the DAG JSON), not with a tool call. This is correct behaviour for a planning agent.

**System prompt:**
```
You are an expert planning agent that produces tool dependency graphs.
```

**User prompt structure:**
```
You are a planning agent. Given a task and a set of available tools,
produce a complete execution plan as a dependency graph (DAG).

TASK:
{task}

The available tools are provided in the 'tools' field of this request.
Each tool name uses '__' as a separator between server name and tool name
(e.g. 'BioMCP__gene_getter' corresponds to the tool 'BioMCP:gene_getter').

Return ONLY valid JSON in this exact schema — no prose, no markdown fences:
{
  "nodes": [
    {
      "id": "1",
      "tool": "ServerName:tool_name",
      "parameters": {"param": "value"},
      "depends_on": [],
      "description": "one-line description of what this step does"
    }
  ]
}

Rules:
- Use the original colon-separated name (e.g. "BioMCP:gene_getter") in the
  "tool" field of each node — not the double-underscore API name.
- Each node id must be a unique string ("1", "2", ...).
- "depends_on" lists the ids of nodes whose output is needed before this node runs.
- Nodes with no shared dependencies will be executed in parallel.
- When a parameter value depends on a previous node's output, write it as
  "{node_id.field}" (e.g. {"variant_id": "{2.id}"}).
- Only include tools that are actually needed to complete the task.
- The graph must be a valid DAG (no cycles).
```

Note that the prompt deliberately instructs the model to use colon-separated names in the DAG output even though the API names use `__`. This ensures the saved plan uses internally consistent naming.

**Comparison:**

| | Prompt text (`use_native_tools=False`) | Native tools (`use_native_tools=True`) |
|---|---|---|
| Tools delivered via | Prompt text block | `tools` API field |
| LLM call method | `get_completion()` | `get_completion_with_tools()` |
| Tool name format in API | `ServerName:tool_name` | `ServerName__tool_name` |
| Tool name format in DAG output | `ServerName:tool_name` | `ServerName:tool_name` |
| `tool_choice` | n/a | `"none"` (model must respond as text) |
| `tool_descriptions` in output | full formatted text | `"native"` |
| Recommended for | general models | fine-tuned tool-calling models |

---

## JSON Parsing

The raw LLM response is processed by `LLMProvider.clean_and_parse_json()`, which:

1. Strips markdown code fences (` ```json ` or ` ``` `) if present
2. Finds the first `{` or `[` character if the response has leading prose
3. Attempts standard `json.loads()`
4. Falls back to `json_repair.loads()` for malformed but recoverable JSON

If parsing fails entirely, `dependency_graph` is set to:
```json
{ "nodes": [], "parse_error": "...", "raw": "..." }
```
and execution continues to the validation step.

---

## DAG Validation (`planning/validation.py`)

After JSON parsing, `validate_dag(dependency_graph)` runs two layers of checks:

### Layer 1 — Pydantic schema validation

The parsed dict is validated against the `DependencyGraph` Pydantic model:

```python
class DAGNode(BaseModel):
    id: str                          # required
    tool: str                        # required
    parameters: Dict[str, Any] = {}  # optional, defaults to {}
    depends_on: List[str] = []       # optional, defaults to []
    description: str = ""            # optional

class DependencyGraph(BaseModel):
    nodes: List[DAGNode]             # required top-level key
    # field_validator: rejects duplicate node IDs
```

This catches: missing required fields, wrong types, and duplicate node IDs.

If Pydantic validation fails, the graph checks (Layer 2) still run on whatever valid nodes could be extracted from the raw dict, so as much diagnostic information as possible is captured.

### Layer 2 — Graph structural checks

Two checks run on the parsed node list:

**Dangling references:** any node ID listed in a `depends_on` that does not correspond to an existing node.

**Cycle detection (Kahn's algorithm):**
- Build an in-degree count and adjacency list from all `depends_on` edges
- Process nodes with in-degree 0 (no unresolved dependencies) iteratively
- If the number of processed nodes is less than the total node count, at least one cycle exists

### Validation result

`validate_dag()` returns a `DAGValidationResult` dataclass, serialised to dict via `.to_dict()`:

```json
{
  "is_valid":      true,
  "schema_valid":  true,
  "has_cycle":     false,
  "dangling_deps": [],
  "duplicate_ids": [],
  "node_count":    4,
  "errors":        []
}
```

`is_valid` is `true` only when all three conditions hold: schema valid, no cycle, no dangling references.

**Important:** validation failure is non-blocking. The result is recorded and the run proceeds. A failed validation is surfaced in the output file — the run entry will have `"status": "success"` (meaning the LLM call completed) with `"validation.is_valid": false`. This distinction is intentional: it separates API/infrastructure failures from model output quality failures.

---

## Return Value

`execute()` returns:

```python
{
    "dependency_graph": {          # parsed plan (or error stub if parsing failed)
        "nodes": [
            {
                "id": "1",
                "tool": "ServerName:tool_name",
                "parameters": { ... },
                "depends_on": [],
                "description": "..."
            },
            ...
        ]
    },
    "validation": {                # always present, even on parse failure
        "is_valid":      bool,
        "schema_valid":  bool,
        "has_cycle":     bool,
        "dangling_deps": [str, ...],
        "duplicate_ids": [str, ...],
        "node_count":    int,
        "errors":        [str, ...]
    },
    "tool_descriptions": str,      # full formatted tool text, or "native"
    "temperature":       float | None,
    "prompt_tokens":     int,
    "completion_tokens": int,
    "total_tokens":      int,
}
```

---

## How Results Are Saved (`run_planning.py`)

`run_planning.py` calls `execute()` in a nested loop over tasks × variations × temperatures × repetitions. After each variation completes, `save_runs()` writes the accumulated results to disk incrementally (so progress is not lost on crash).

Each saved repetition entry looks like:

```json
{
  "temperature": 0.7,
  "repetition": 0,
  "status": "success",
  "generated_plan": { "nodes": [ ... ] },
  "validation": {
    "is_valid": true,
    "schema_valid": true,
    "has_cycle": false,
    "dangling_deps": [],
    "duplicate_ids": [],
    "node_count": 4,
    "errors": []
  },
  "token_usage": {
    "prompt_tokens": 1842,
    "completion_tokens": 310,
    "total_tokens": 2152
  }
}
```

The `experiment_config` block at the top of each output file records whether `--native-tools` was used:

```json
"experiment_config": {
  "variations": 1,
  "repetitions": 3,
  "temperatures": [0.0, 0.7],
  "total_tasks": 100,
  "total_runs": 600,
  "native_tools": false
}
```

---

## Running the Planning Benchmark

```bash
# Default: prompt-text tool mode
python run_planning.py \
    --tasks tasks/mcpbench_tasks_multi_2server_runner_format.json \
    --model claude-sonnet-4 \
    --repetitions 3 \
    --temperatures 0.0 0.5 1.0

# Native tools mode (for fine-tuned tool-calling models)
python run_planning.py \
    --tasks tasks/mcpbench_tasks_multi_2server_runner_format.json \
    --model gpt-4o \
    --native-tools
```

Output is written to `results/planning/<model>/<timestamp>/` with one file per server count:
- `1server.json` — single-server tasks
- `2server.json` — two-server tasks
- `3server.json` — three-server tasks
