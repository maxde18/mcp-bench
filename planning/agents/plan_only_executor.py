"""Planning-only executor.

Generates a tool dependency graph from a fuzzy task description and available
tool descriptions. No tools are executed.

This is used by run_planning_benchmark.py to isolate and evaluate the planning
phase independently from execution.
"""

import logging
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

import config.config_loader as config_loader
from mcp_infra.connector import MCPConnector
from planning.validation import validate_dag

logger = logging.getLogger(__name__)

# OpenRouter structured output schema for the dependency graph response.
# Passed as response_format when the provider is "openrouter".
_DEPENDENCY_GRAPH_RESPONSE_FORMAT = {
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
                            "id":          {"type": "string"},
                            "tool":        {"type": "string"},
                            "parameters":  {"type": "object"},
                            "depends_on":  {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "description": {"type": "string"},
                        },
                        "required": ["id", "tool", "parameters", "depends_on", "description"],
                    },
                },
            },
            "required": ["nodes"],
        },
    },
}


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class PlannerState(TypedDict):
    original_task: str          # never modified — used as base for retry feedback
    task: str                   # current prompt sent to the planner (may include error feedback)
    use_native_tools: bool
    temperature: Optional[float]
    dependency_graph: Dict[str, Any]
    validation: Dict[str, Any]
    tool_descriptions: str
    attempt: int
    max_retries: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class PlanOnlyExecutor:
    """Generates an execution plan (dependency graph) without running any tools.

    Uses a LangGraph StateGraph with two nodes:
      planner   — calls the LLM and parses the DAG JSON.
      validator — checks the DAG for schema errors, cycles, and dangling deps.
    A conditional edge retries the planner (with error feedback injected into
    the prompt) when validation fails and retries remain.

    Args:
        llm_provider: Existing LLMProvider instance (from llm/provider.py).
        all_tools:    Dict of tool descriptions from ConnectionManager.all_tools.
                      No server_manager is needed — tools are never called.
    """

    def __init__(self, llm_provider, all_tools: Dict[str, Any]) -> None:
        self.llm = llm_provider
        self.all_tools = all_tools
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        builder = StateGraph(PlannerState)
        builder.add_node("planner", self._planner_node)
        builder.add_node("validator", self._validator_node)
        builder.set_entry_point("planner")
        builder.add_edge("planner", "validator")
        builder.add_conditional_edges(
            "validator",
            self._should_retry,
            {"retry": "planner", "done": END},
        )
        return builder.compile()

    # ------------------------------------------------------------------
    # Routing function
    # ------------------------------------------------------------------

    def _should_retry(self, state: PlannerState) -> str:
        if not state["validation"]["is_valid"] and state["attempt"] < state["max_retries"]:
            return "retry"
        return "done"

    # ------------------------------------------------------------------
    # Node: planner
    # ------------------------------------------------------------------

    async def _planner_node(self, state: PlannerState) -> PlannerState:
        if state["use_native_tools"]:
            api_tools = MCPConnector.format_tools_for_api(self.all_tools)
            tool_descriptions = "native"

            prompt = f"""You are a planning agent. Given a task and a set of available tools,
produce a complete execution plan as a dependency graph (DAG).

TASK:
{state["task"]}

The available tools are provided in the 'tools' field of this request.
Each tool name uses '__' as a separator between server name and tool name
(e.g. 'BioMCP__gene_getter' corresponds to the tool 'BioMCP:gene_getter').

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
- Use the original colon-separated name (e.g. "BioMCP:gene_getter") in the
  "tool" field of each node — not the double-underscore API name.
- Each node id must be a unique string ("1", "2", ...).
- "depends_on" lists the ids of nodes whose output is needed before this node runs.
- Nodes with no shared dependencies will be executed in parallel.
- When a parameter value depends on a previous node's output, write it as
  "{{node_id.field}}" (e.g. "{{"variant_id": "{{2.id}}"}}").
- Only include tools that are actually needed to complete the task.
- The graph must be a valid DAG (no cycles).
"""

            response_data = await self.llm.get_completion_with_tools(
                "You are an expert planning agent that produces tool dependency graphs.",
                prompt,
                api_tools,
                config_loader.get_planning_tokens(),
                tool_choice="none",
                return_usage=True,
                temperature=state["temperature"],
                response_format=_DEPENDENCY_GRAPH_RESPONSE_FORMAT,
            )
        else:
            tool_descriptions = MCPConnector.format_tools_for_prompt(self.all_tools)

            prompt = f"""You are a planning agent. Given a task and a set of available tools,
produce a complete execution plan as a dependency graph (DAG).

TASK:
{state["task"]}

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
                temperature=state["temperature"],
                response_format=_DEPENDENCY_GRAPH_RESPONSE_FORMAT,
            )

        response_str, usage = (
            response_data if isinstance(response_data, tuple) else (response_data, None)
        )

        try:
            dependency_graph = self.llm.clean_and_parse_json(response_str)
        except Exception as e:
            logger.error(f"Failed to parse dependency graph JSON: {e}")
            dependency_graph = {"nodes": [], "parse_error": str(e), "raw": response_str}

        return {
            **state,
            "dependency_graph":  dependency_graph,
            "tool_descriptions": tool_descriptions,
            "prompt_tokens":     state["prompt_tokens"] + (usage.get("prompt_tokens", 0) if usage else 0),
            "completion_tokens": state["completion_tokens"] + (usage.get("completion_tokens", 0) if usage else 0),
            "total_tokens":      state["total_tokens"] + (usage.get("total_tokens", 0) if usage else 0),
        }

    # ------------------------------------------------------------------
    # Node: validator
    # ------------------------------------------------------------------

    async def _validator_node(self, state: PlannerState) -> PlannerState:
        validation = validate_dag(state["dependency_graph"])

        # Build error-feedback prompt for the next attempt, rooted in the
        # original task so feedback does not accumulate across retries.
        task = state["original_task"]
        if not validation.is_valid and state["attempt"] < state["max_retries"] - 1:
            error_feedback = "\n".join(validation.errors)
            task = (
                f"{state['original_task']}\n\n"
                f"Your previous plan was invalid. Fix these errors and try again:\n"
                f"{error_feedback}"
            )

        logger.info(
            f"Attempt {state['attempt'] + 1}/{state['max_retries']} | "
            f"nodes={len(state['dependency_graph'].get('nodes', []))} | "
            f"valid={validation.is_valid}"
        )

        return {
            **state,
            "validation": validation.to_dict(),
            "task":       task,
            "attempt":    state["attempt"] + 1,
        }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def execute(
        self,
        task: str,
        temperature: Optional[float] = None,
        use_native_tools: bool = False,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Generate a dependency graph for the task and return it.

        Args:
            task:              The fuzzy task description shown to the agent.
            temperature:       Sampling temperature (0.0–2.0).  None uses the
                               model default.  Ignored for o-series models.
            use_native_tools:  When True, MCP tools are passed via the OpenAI
                               ``tools`` field instead of being embedded as text
                               in the prompt.  Recommended for fine-tuned
                               tool-calling models.  Defaults to False.
            max_retries:       Maximum number of planning attempts.  On each
                               failed validation the planner is re-invoked with
                               the validation errors injected into the prompt.
                               The final attempt is always returned regardless
                               of validity.  Defaults to 3.

        Returns:
            {
                "dependency_graph": { "nodes": [...] },
                "validation":        dict,           # DAGValidationResult.to_dict()
                "tool_descriptions": str,            # what the agent saw (text or "native")
                "temperature":       float | None,
                "prompt_tokens":     int,
                "completion_tokens": int,
                "total_tokens":      int,
            }
        """
        initial_state: PlannerState = {
            "original_task":    task,
            "task":             task,
            "use_native_tools": use_native_tools,
            "temperature":      temperature,
            "dependency_graph": {},
            "validation":       {},
            "tool_descriptions": "",
            "attempt":          0,
            "max_retries":      max_retries,
            "prompt_tokens":    0,
            "completion_tokens": 0,
            "total_tokens":     0,
        }

        final_state = await self.graph.ainvoke(initial_state)

        return {
            "dependency_graph":  final_state["dependency_graph"],
            "validation":        final_state["validation"],
            "tool_descriptions": final_state["tool_descriptions"],
            "temperature":       temperature,
            "prompt_tokens":     final_state["prompt_tokens"],
            "completion_tokens": final_state["completion_tokens"],
            "total_tokens":      final_state["total_tokens"],
        }
