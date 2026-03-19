# Agent Integration Guide

This guide explains how to integrate a custom agent into MCP-Bench. It covers
what the benchmark provides, what your agent must implement, and how to wire
everything together.

---

## The Core Idea

The benchmark runner treats the agent as a black box. It hands the agent two
things and expects one thing back:

```
runner gives:   task description  (string)
                server_manager    (live MCP connections with callable tools)

runner expects: a result dict     (defined below)
```

Everything else — server startup, tool discovery, evaluation, scoring, saving
results — is handled by the benchmark automatically. You only need to implement
what happens between receiving the task and returning the result.

---

## Step 1 — Create Your Agent File

Create a file at `agent/your_agent.py`. The class must have this structure:

```python
class YourAgent:

    def __init__(self, llm_provider, server_manager):
        self.llm = llm_provider
        self.server_manager = server_manager
        self.all_tools = server_manager.all_tools  # dict of all available tools

    async def execute(self, task: str) -> dict:
        # your logic here
        return { ... }  # see required fields below
```

Both arguments are injected by the runner — you never construct them yourself.

---

## What You Have Access to Inside the Agent

### `self.llm` — the LLM provider

```python
response = await self.llm.get_completion(
    system_prompt,      # str
    user_prompt,        # str
    max_tokens,         # int  — use config_loader.get_planning_tokens()
    return_usage=True,  # returns (response_str, usage_dict) instead of just str
    temperature=0.7,    # optional float, ignored for o-series reasoning models
)
```

When `return_usage=True` the return value is a tuple `(response_str, usage_dict)` where
`usage_dict` contains `prompt_tokens`, `completion_tokens`, and `total_tokens`.

### `self.all_tools` — every available tool's schema

A dict keyed by `"ServerName:tool_name"`. Each value contains:

```python
{
    "server":      "BioMCP",
    "description": "Retrieve gene annotation from NCBI...",
    "parameters":  { ... }  # JSON schema of accepted parameters
}
```

To format all tools into a readable string for an LLM prompt:

```python
from mcp_modules.connector import MCPConnector

tool_descriptions = MCPConnector.format_tools_for_prompt(self.all_tools)
```

### `self.server_manager` — tool execution

```python
result = await self.server_manager.call_tool(
    "BioMCP:gene_getter",            # "ServerName:tool_name"
    {"gene_id_or_symbol": "BRAF"}    # parameters dict
)
```

The tool name format is always `"ServerName:tool_name"`, matching the keys in
`self.all_tools`. Results are automatically cached between calls (controlled by
`config/benchmark_config.yaml`).

---

## Step 2 — Implement `execute()`

`execute()` must be `async` and must return a dict with these fields:

```python
async def execute(self, task: str) -> dict:
    return {
        # Required — the evaluator uses all of these
        "solution":                             str,   # your agent's final answer
        "total_rounds":                         int,   # how many iterations you ran
        "execution_results":                    list,  # every tool call made (see below)
        "planning_json_compliance":             float, # 0.0–1.0, set 1.0 if not applicable
        "accumulated_information":              str,   # intermediate context/notes
        "accumulated_information_uncompressed": str,   # same, or uncompressed version

        # Required — for cost and usage tracking
        "total_output_tokens":                  int,
        "total_prompt_tokens":                  int,
        "total_tokens":                         int,
    }
```

### `execution_results` format

Each entry in `execution_results` documents one tool call:

```python
{
    "tool_name":   "BioMCP:gene_getter",           # "ServerName:tool_name"
    "tool_input":  {"gene_id_or_symbol": "BRAF"},  # parameters you passed
    "tool_output": { ... },                         # raw result from the server
    "round":       1,                               # which iteration this was in
}
```

The evaluator reads `execution_results` to score tool appropriateness, parameter
accuracy, and planning efficiency. The richer and more accurate this list is, the
more meaningful your evaluation scores will be.

---

## What the Evaluator Does With Your Output

Once `execute()` returns, the runner passes the result to `TaskEvaluator` which
scores six sub-dimensions (1–10 each) using an LLM judge:

| Dimension | What it looks at in your result |
|---|---|
| Task Fulfillment | `solution` vs task requirements |
| Grounding | whether `solution` claims are backed by `execution_results` |
| Tool Appropriateness | whether tools in `execution_results` were sensible choices |
| Parameter Accuracy | whether `tool_input` values were correct |
| Dependency Awareness | whether tool call order in `execution_results` respects dependencies |
| Parallelism & Efficiency | whether independent tools were called in the same round |

The evaluator also performs a rule-based schema check on `solution` for required
fields, giving a binary compliance score independent of the LLM judge.

---

## Step 3 — Swap the Agent in `runner.py`

Open `benchmark/runner.py` and find the executor construction around line 490:

```python
# before
executor = LangGraphExecutor(llm_provider, conn_mgr.server_manager)
```

Replace the import at the top of the file and the constructor call:

```python
# 1. Change the import (top of file)
from agent.your_agent import YourAgent

# 2. Change the constructor (~line 490)
executor = YourAgent(llm_provider, conn_mgr.server_manager)
```

That is the only change required outside your agent file.

---

## Practical Skeleton

A minimal working agent you can copy and build from:

```python
"""agent/your_agent.py"""
import logging
from typing import Any, Dict, List

import config.config_loader as config_loader
from mcp_modules.connector import MCPConnector

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

        # ----------------------------------------------------------------
        # Your agent logic goes here.
        # The example below makes one planning call and one tool call.
        # ----------------------------------------------------------------

        # Planning call
        response, usage = await self.llm.get_completion(
            "You are a helpful agent.",
            f"Task: {task}\n\nTools:\n{tool_descriptions}",
            config_loader.get_planning_tokens(),
            return_usage=True,
        )
        if usage:
            total_prompt_tokens     += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)

        # Tool call
        result = await self.server_manager.call_tool(
            "BioMCP:gene_getter",
            {"gene_id_or_symbol": "BRAF"},
        )
        execution_results.append({
            "tool_name":   "BioMCP:gene_getter",
            "tool_input":  {"gene_id_or_symbol": "BRAF"},
            "tool_output": result,
            "round":       1,
        })

        # ----------------------------------------------------------------

        total_tokens = total_prompt_tokens + total_completion_tokens
        return {
            "solution":                             response,
            "total_rounds":                         1,
            "execution_results":                    execution_results,
            "planning_json_compliance":             1.0,
            "accumulated_information":              str(result),
            "accumulated_information_uncompressed": str(result),
            "total_output_tokens":                  total_completion_tokens,
            "total_prompt_tokens":                  total_prompt_tokens,
            "total_tokens":                         total_tokens,
        }
```

---

## Running the Benchmark With Your Agent

Once your agent is swapped into `runner.py`:

```bash
# Run against the 2-server tasks
python run_benchmark.py --models claude-sonnet-4 \
    --tasks-file tasks/mcpbench_tasks_multi_2server_runner_format.json

# Quick test with a task limit
python run_benchmark.py --models claude-sonnet-4 \
    --tasks-file tasks/mcpbench_tasks_multi_2server_runner_format.json \
    --task-limit 5
```

Results are saved to a timestamped JSON file and printed as a score table.

---

## Supporting Multiple Agents Without Editing `runner.py` Each Time

If you want to compare several agents without manually editing `runner.py` for
each run, replace the hardcoded executor construction with a registry and a CLI
flag.

**In `benchmark/runner.py`, add a registry near the top:**

```python
AGENT_REGISTRY = {
    "baseline":   ("agent.executor",           "TaskExecutor",      True),
    "langgraph":  ("agent.langgraph_executor", "LangGraphExecutor", False),
    "your_agent": ("agent.your_agent",         "YourAgent",         False),
    #              module path                  class name           needs concurrent_summarization
}
```

**Replace the executor construction (~line 490):**

```python
import importlib

agent_key = self.agent  # set from a CLI arg or config flag
module_path, class_name, needs_concurrent = AGENT_REGISTRY[agent_key]

module    = importlib.import_module(module_path)
AgentClass = getattr(module, class_name)

if needs_concurrent:
    executor = AgentClass(llm_provider, conn_mgr.server_manager, self.concurrent_summarization)
else:
    executor = AgentClass(llm_provider, conn_mgr.server_manager)
```

**Then pass `--agent your_agent` from the CLI** and the correct class is loaded
at runtime without any further code changes.

---

## Summary Checklist

| Requirement | Detail |
|---|---|
| File location | `agent/your_agent.py` |
| Constructor signature | `__init__(self, llm_provider, server_manager)` |
| Entry point | `async def execute(self, task: str) -> dict` |
| Making tool calls | `await self.server_manager.call_tool("Server:tool", params)` |
| Getting tool descriptions | `MCPConnector.format_tools_for_prompt(self.all_tools)` |
| Making LLM calls | `await self.llm.get_completion(system, user, max_tokens, ...)` |
| Required return keys | `solution`, `total_rounds`, `execution_results`, `planning_json_compliance`, `accumulated_information`, `accumulated_information_uncompressed`, token counts |
| `execution_results` entry fields | `tool_name`, `tool_input`, `tool_output`, `round` |
| Change required in runner | 2 lines in `benchmark/runner.py` (~line 490) |
