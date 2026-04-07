"""Planning-only executor.

Generates a tool dependency graph from a fuzzy task description and available
tool descriptions. No tools are executed.

Uses ``langchain-openrouter``'s ``ChatOpenRouter`` for native LangGraph
integration. Tools are bound via ``.bind_tools()`` and structured output is
enforced via ``.bind(response_format=...)``, removing the need for separate
``get_completion`` / ``get_completion_with_tools`` code paths.
"""

import json
import logging
from typing import Any, Dict, Optional, TypedDict

import json_repair
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import END, StateGraph

import config.config_loader as config_loader
from mcp_infra.connector import MCPConnector
from planning.agents.few_shot_examples import EXAMPLES, build_tool_inventory, format_few_shot_block
from planning.validation import validate_dag

logger = logging.getLogger(__name__)

# MiniMax models on OpenRouter do not expose a native tools API field, so tools
# must be embedded in the prompt using MiniMax's JSON training format.
_MINIMAX_MODEL_SUBSTRINGS = ("minimax/",)

# Reasoning models that reject a custom temperature parameter.
_MODELS_WITHOUT_TEMPERATURE = {
    "openai/o1-preview", "openai/o1-mini", "openai/o4-mini", "openai/o3-mini", "openai/o3",
}

# Singleton system message shared across all invocations.
_SYSTEM_MSG = SystemMessage("You are an expert planning agent that produces tool dependency graphs.")

# OpenRouter structured output schema for the dependency graph response.
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
                            "filter":      {"type": "string"},
                            "condition":   {"type": "string"},
                        },
                        "required": ["id", "tool", "parameters", "depends_on", "description"],
                    },
                },
            },
            "required": ["nodes"],
        },
    },
}


def _build_planner_prompt(task: str, tools_section: str, few_shot_block: str, extra_rules: str = "") -> str:
    """Build the planner prompt.

    Args:
        task:           The task description shown to the agent.
        tools_section:  The block describing available tools (differs between
                        native-tools mode and text-based mode).
        few_shot_block: Pre-built few-shot section (includes tool registry +
                        examples) from format_few_shot_block().
        extra_rules:    Optional additional rule lines prepended to the common
                        rules block (used for the native-tools name convention).
    """
    return f"""You are a planning agent. Given a task and a set of available tools,
produce a complete execution plan as a dependency graph (DAG).

{few_shot_block}

Now produce the plan for the following task.

TASK:
{task}

{tools_section}

Return ONLY valid JSON in this exact schema — no prose, no markdown fences:
{{
  "nodes": [
    {{
      "id": "1",
      "tool": "ServerName:tool_name",
      "parameters": {{"param": "value"}},
      "depends_on": [],
      "description": "one-line description of what this step does",
      "filter": "(optional) post-execution filter expression on this node's output",
      "condition": "(optional) boolean gate — node only executes when this is true"
    }}
  ]
}}

Rules:
{extra_rules}- Each node id must be a unique string ("1", "2", ...).
- "depends_on" lists the ids of nodes whose output is needed before this node runs.
- Nodes with no shared dependencies will be executed in parallel.
- When a parameter value depends on a previous node's output, write it as
  "{{node_id.field}}" (e.g. "{{"variant_id": "{{2.id}}"}}").
- Only include tools that are actually needed to complete the task.
- The graph must be a valid DAG (no cycles).
- "filter" is optional. Use it when the task requires keeping only results that
  match a post-execution predicate (e.g. "license == 'apache-2.0' AND weight_size < 2000000000").
- "condition" is optional. Use it as a boolean gate that determines whether a node
  executes at all based on a prior node's output. For conditional branching, create
  two separate nodes with complementary conditions rather than combining tools in
  one node.
"""


def _clean_and_parse_json(raw: str) -> Any:
    """Strip markdown fences and parse JSON, falling back to json_repair."""
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        parts = raw.split("```")
        if len(parts) >= 2:
            raw = parts[1].strip()

    raw = raw.strip()
    if not raw.startswith("{") and not raw.startswith("["):
        first_brace = raw.find("{")
        first_bracket = raw.find("[")
        if first_brace == -1 and first_bracket == -1:
            raise ValueError(f"No JSON object or array found in response: {raw[:200]}")
        start = min(x for x in (first_brace, first_bracket) if x != -1)
        raw = raw[start:]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json_repair.loads(raw)


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class PlannerState(TypedDict):
    original_task: str          # never modified — used as base for retry feedback
    task: str                   # current prompt sent to the planner (may include error feedback)
    use_native_tools: bool
    use_structured_output: bool
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
      planner   — calls the LLM via ChatOpenRouter and parses the DAG JSON.
      validator — checks the DAG for schema errors, cycles, and dangling deps.
    A conditional edge retries the planner (with error feedback injected into
    the prompt) when validation fails and retries remain.

    The ChatOpenRouter model is configured per-call using LangChain's ``.bind()``
    and ``.bind_tools()`` methods, collapsing what was previously three separate
    code paths (text mode, native-tools mode, MiniMax mode) into a single
    ``model.ainvoke(messages)`` call.

    Args:
        model:     ChatOpenRouter instance from LLMFactory.create_chat_model().
        all_tools: Dict of tool descriptions from ConnectionManager.all_tools.
                   No server_manager is needed — tools are never called.
    """

    def __init__(self, model: ChatOpenRouter, all_tools: Dict[str, Any]) -> None:
        self.base_model = model
        self.all_tools = all_tools
        self.graph = self._build_graph()

    def _is_minimax_model(self) -> bool:
        """Return True when the LLM is a MiniMax model accessed via OpenRouter."""
        return any(sub in self.base_model.model_name for sub in _MINIMAX_MODEL_SUBSTRINGS)

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
        # Start from the base model and layer configuration for this run.
        model = self.base_model

        # Apply temperature — skipped for reasoning models that reject it.
        if (
            state["temperature"] is not None
            and model.model_name not in _MODELS_WITHOUT_TEMPERATURE
        ):
            model = model.bind(temperature=state["temperature"])

        # Build the tool inventory (used in the few-shot block for all modes).
        api_tools = MCPConnector.format_tools_for_api(self.all_tools)
        inventory = build_tool_inventory(api_tools)
        few_shot_block = format_few_shot_block(EXAMPLES, tool_inventory=inventory)

        # Build the tools section and configure model tool/output binding.
        if state["use_native_tools"]:
            # Pass tools via the OpenRouter `tools` API field so fine-tuned
            # tool-calling models receive them in their expected format.
            tools_section = ""
            extra_rules = (
                '- Tool names in the "tool" field must use colon-separated form\n'
                '  (e.g. "BioMCP:gene_getter"), not the double-underscore form\n'
                '  (e.g. "BioMCP__gene_getter") used in the function definitions.\n'
            )
            # bind_tools() sends tools in the structured API field.
            # tool_choice="none" makes the model text-respond (output the DAG)
            # rather than actually calling any tool.
            model = model.bind_tools(api_tools, tool_choice="none")
            if state["use_structured_output"]:
                model = model.bind(response_format=_DEPENDENCY_GRAPH_RESPONSE_FORMAT)
            tool_descriptions = "native"

        elif self._is_minimax_model():
            # MiniMax models were trained on a specific JSON tool format.
            # OpenRouter does not expose a native tools field for these models.
            minimax_tools_json = MCPConnector.format_tools_for_minimax_prompt(self.all_tools)
            tools_section = (
                "The available tools are listed below as a JSON array.\n"
                "Each tool name uses '__' as a separator between server name and tool name\n"
                "(e.g. 'BioMCP__gene_getter' corresponds to 'BioMCP:gene_getter').\n\n"
                f"AVAILABLE TOOLS:\n```json\n{minimax_tools_json}\n```"
            )
            extra_rules = (
                '- Use the original colon-separated name (e.g. "BioMCP:gene_getter") in the\n'
                '  "tool" field of each node — not the double-underscore name.\n'
            )
            model = model.bind(response_format=_DEPENDENCY_GRAPH_RESPONSE_FORMAT)
            tool_descriptions = "minimax_json"

        else:
            # Default: embed tool descriptions as text in the prompt.
            tool_descriptions = MCPConnector.format_tools_for_prompt(self.all_tools)
            tools_section = f"AVAILABLE TOOLS:\n{tool_descriptions}"
            extra_rules = ""
            model = model.bind(response_format=_DEPENDENCY_GRAPH_RESPONSE_FORMAT)

        prompt = _build_planner_prompt(state["task"], tools_section, few_shot_block, extra_rules)
        messages = [_SYSTEM_MSG, HumanMessage(prompt)]

        logger.info(
            f"Calling {self.base_model.model_name} "
            f"(attempt {state['attempt'] + 1}/{state['max_retries']}, "
            f"native_tools={state['use_native_tools']})"
        )
        response = await model.ainvoke(messages)

        # Usage metadata uses LangChain's standard keys (input_tokens / output_tokens).
        usage = response.usage_metadata or {}

        try:
            dependency_graph = _clean_and_parse_json(response.content)
        except Exception as e:
            logger.error(f"Failed to parse dependency graph JSON: {e}")
            dependency_graph = {"nodes": [], "parse_error": str(e), "raw": response.content}

        return {
            **state,
            "dependency_graph":  dependency_graph,
            "tool_descriptions": tool_descriptions,
            "prompt_tokens":     state["prompt_tokens"]     + usage.get("input_tokens", 0),
            "completion_tokens": state["completion_tokens"] + usage.get("output_tokens", 0),
            "total_tokens":      state["total_tokens"]      + usage.get("total_tokens", 0),
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
        use_structured_output: bool = False,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Generate a dependency graph for the task and return it.

        Args:
            task:                   The fuzzy task description shown to the agent.
            temperature:            Sampling temperature (0.0–2.0).  None uses the
                                    model default.  Ignored for reasoning models.
            use_native_tools:       When True, MCP tools are passed via the OpenRouter
                                    ``tools`` API field using ``bind_tools()``.
                                    Recommended for fine-tuned tool-calling models.
                                    Defaults to False (prompt-injection mode).
            use_structured_output:  When True, ``response_format`` (json_schema) is
                                    enforced via ``bind(response_format=...)``.  Only
                                    applies when ``use_native_tools`` is also True.
                                    Defaults to False.
            max_retries:            Maximum number of planning attempts.  On each
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
            "original_task":         task,
            "task":                  task,
            "use_native_tools":      use_native_tools,
            "use_structured_output": use_structured_output,
            "temperature":           temperature,
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
