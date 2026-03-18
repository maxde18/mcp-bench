"""Planning-only executor.

Generates a tool dependency graph from a fuzzy task description and available
tool descriptions. No tools are executed.

This is used by run_planning_benchmark.py to isolate and evaluate the planning
phase independently from execution.
"""

import logging
from typing import Any, Dict

import config.config_loader as config_loader
from mcp_modules.connector import MCPConnector

logger = logging.getLogger(__name__)


class PlanOnlyExecutor:
    """Generates an execution plan (dependency graph) without running any tools.

    Args:
        llm_provider: Existing LLMProvider instance (from llm/provider.py).
        all_tools:    Dict of tool descriptions from PersistentMultiServerManager.all_tools.
                      No server_manager is needed — tools are never called.
    """

    def __init__(self, llm_provider, all_tools: Dict[str, Any]) -> None:
        self.llm = llm_provider
        self.all_tools = all_tools

    async def execute(self, task: str, temperature: float = None) -> Dict[str, Any]:
        """Generate a dependency graph for the task and return it.

        Args:
            task:        The fuzzy task description shown to the agent.
            temperature: Sampling temperature passed to the LLM (0.0–2.0).
                         When None the model default is used. Ignored for
                         reasoning models (o-series) that do not support it.

        Returns:
            {
                "dependency_graph": { "nodes": [...] },
                "tool_descriptions": str,   # what the agent saw
                "temperature":       float | None,
                "prompt_tokens":     int,
                "completion_tokens": int,
                "total_tokens":      int,
            }
        """
        tool_descriptions = MCPConnector.format_tools_for_prompt(self.all_tools)

        prompt = f"""You are a planning agent. Given a task and a set of available tools,
produce a complete execution plan as a dependency graph (DAG).

TASK:
{task}

AVAILABLE TOOLS:
{tool_descriptions}

Return ONLY valid JSON in this exact schema — no prose, no markdown fences:
{{
  "nodes": [
    {{
      "id": "1",
      "tool": "ServerName:tool_name",
      "parameters": {{"param": "value"}},
      "depends_on": [],
      "description": "one-line description of what this step does"
    }}
  ]
}}

Rules:
- Each node id must be a unique string ("1", "2", ...).
- "depends_on" lists the ids of nodes whose output is needed before this node runs.
- Nodes with no shared dependencies will be executed in parallel.
- When a parameter value depends on a previous node's output, write it as
  "{{node_id.field}}" (e.g. "{{"variant_id": "{{2.id}}"}}").
- Only include tools that are actually needed to complete the task.
- The graph must be a valid DAG (no cycles).
"""

        response_data = await self.llm.get_completion(
            "You are an expert planning agent that produces tool dependency graphs.",
            prompt,
            config_loader.get_planning_tokens(),
            return_usage=True,
            temperature=temperature,
        )

        response_str, usage = (
            response_data if isinstance(response_data, tuple) else (response_data, None)
        )

        try:
            dependency_graph = self.llm.clean_and_parse_json(response_str)
        except Exception as e:
            logger.error(f"Failed to parse dependency graph JSON: {e}")
            dependency_graph = {"nodes": [], "parse_error": str(e), "raw": response_str}

        logger.info(
            f"Plan generated: {len(dependency_graph.get('nodes', []))} nodes"
        )

        return {
            "dependency_graph":  dependency_graph,
            "tool_descriptions": tool_descriptions,
            "temperature":       temperature,
            "prompt_tokens":     usage.get("prompt_tokens", 0) if usage else 0,
            "completion_tokens": usage.get("completion_tokens", 0) if usage else 0,
            "total_tokens":      usage.get("total_tokens", 0) if usage else 0,
        }
