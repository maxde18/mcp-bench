"""Decompose-then-assemble planning executor.

Three-phase planning pipeline:
  1. Node Selection       — LLM selects the relevant tool functions from the full
                            tool inventory given the task description.
  2. Dependency Analysis  — LLM analyses every ordered pair of selected nodes and
                            determines whether one node's output is a direct,
                            immediate upstream input to the other.  This step also
                            names the specific output field and input parameter that
                            flow between the two nodes.
  3. DAG Assembly         — LLM constructs the full DAG JSON from the pre-selected
                            nodeset and the pre-computed dependency matrix.

Validation and retry logic (applied to phase 3 only) mirror PlanOnlyExecutor.
"""

import json
import logging
from typing import Any, Dict, List, Optional, TypedDict

import json_repair
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import END, StateGraph

from mcp_infra.connector import MCPConnector
from planning.agents.few_shot_examples import EXAMPLES, build_tool_inventory, format_few_shot_block
from planning.validation import validate_dag

logger = logging.getLogger(__name__)

_MINIMAX_MODEL_SUBSTRINGS = ("minimax/",)

_MODELS_WITHOUT_TEMPERATURE = {
    "openai/o1-preview", "openai/o1-mini", "openai/o4-mini", "openai/o3-mini", "openai/o3",
}

_SYSTEM_MSG = SystemMessage("You are an expert planning agent that produces tool dependency graphs.")

# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------

_NODE_SELECTION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "node_selection",
        "schema": {
            "type": "object",
            "properties": {
                "selected_nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool":      {"type": "string"},
                            "rationale": {"type": "string"},
                        },
                        "required": ["tool", "rationale"],
                    },
                },
            },
            "required": ["selected_nodes"],
        },
    },
}

_DEPENDENCY_ANALYSIS_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "dependency_analysis",
        "schema": {
            "type": "object",
            "properties": {
                "dependencies": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "producer":     {"type": "string"},
                            "consumer":     {"type": "string"},
                            "output_field": {"type": "string"},
                            "input_param":  {"type": "string"},
                            "reasoning":    {"type": "string"},
                        },
                        "required": ["producer", "consumer", "output_field", "input_param", "reasoning"],
                    },
                },
            },
            "required": ["dependencies"],
        },
    },
}

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
                            "depends_on":  {"type": "array", "items": {"type": "string"}},
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
            raise ValueError(f"No JSON found in response: {raw[:200]}")
        start = min(x for x in (first_brace, first_bracket) if x != -1)
        raw = raw[start:]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json_repair.loads(raw)


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class DecomposeState(TypedDict):
    original_task: str
    temperature: Optional[float]
    top_k: Optional[int]
    top_p: Optional[float]
    # Phase outputs
    selected_nodes: List[Dict[str, str]]    # [{tool, rationale}, ...]
    dependencies: List[Dict[str, str]]      # [{producer, consumer, output_field, input_param, reasoning}, ...]
    dependency_graph: Dict[str, Any]
    validation: Dict[str, Any]
    # Retry tracking
    attempt: int
    max_retries: int
    # Token accounting
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class DecomposeAssembleExecutor:
    """Three-phase planning agent: select nodes → analyse pairwise deps → assemble DAG.

    Uses a LangGraph StateGraph with four nodes:
      node_selector       — phase 1: choose relevant tools from the inventory.
      dependency_analyzer — phase 2: pairwise immediate-upstream-producer reasoning.
      dag_assembler       — phase 3: build full DAG from nodeset + dependency matrix.
      validator           — validate schema, cycles, and dangling references.

    Retry logic applies only to the dag_assembler node.  On failure the
    validator injects the error list back into state; the assembler reads it
    on its next invocation.

    Args:
        model:     ChatOpenRouter instance from LLMFactory.create_chat_model().
        all_tools: Dict of tool descriptions from ConnectionManager.all_tools.
    """

    def __init__(self, model: ChatOpenRouter, all_tools: Dict[str, Any]) -> None:
        self.base_model = model
        self.all_tools = all_tools
        self.graph = self._build_graph()

    def _is_minimax_model(self) -> bool:
        return any(sub in self.base_model.model_name for sub in _MINIMAX_MODEL_SUBSTRINGS)

    def _bind_sampling(self, model: ChatOpenRouter, state: DecomposeState) -> ChatOpenRouter:
        """Apply temperature / top_p / top_k, skipping temperature for reasoning models."""
        if (
            state["temperature"] is not None
            and model.model_name not in _MODELS_WITHOUT_TEMPERATURE
        ):
            model = model.bind(temperature=state["temperature"])
        if state["top_p"] is not None:
            model = model.bind(top_p=state["top_p"])
        if state["top_k"] is not None:
            # The openrouter SDK's Chat.send_async() does not accept top_k directly
            # (it is only supported in the /responses endpoint).  Passing it causes
            # a TypeError.  Log a warning and skip rather than crashing the run.
            logger.warning(
                "top_k is not supported by langchain-openrouter's Chat.send_async() "
                "and will be ignored.  Use top_p for nucleus sampling instead."
            )
        return model

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        builder = StateGraph(DecomposeState)
        builder.add_node("node_selector",       self._node_selector_node)
        builder.add_node("dependency_analyzer", self._dependency_analyzer_node)
        builder.add_node("dag_assembler",       self._dag_assembler_node)
        builder.add_node("validator",           self._validator_node)
        builder.set_entry_point("node_selector")
        builder.add_edge("node_selector",       "dependency_analyzer")
        builder.add_edge("dependency_analyzer", "dag_assembler")
        builder.add_edge("dag_assembler",       "validator")
        builder.add_conditional_edges(
            "validator",
            self._should_retry,
            {"retry": "dag_assembler", "done": END},
        )
        return builder.compile()

    def _should_retry(self, state: DecomposeState) -> str:
        if not state["validation"]["is_valid"] and state["attempt"] < state["max_retries"]:
            return "retry"
        return "done"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _nodeset_summary(self, selected_nodes: List[Dict[str, str]]) -> str:
        """Detailed description of each selected node for downstream prompts."""
        lines = []
        for entry in selected_nodes:
            tool = entry["tool"]
            rationale = entry.get("rationale", "")
            tool_info = self.all_tools.get(tool, {})
            desc = tool_info.get("description", "")
            schema = tool_info.get("inputSchema", {})
            props = schema.get("properties", {})
            required = schema.get("required", [])
            param_lines = []
            for pname, pschema in props.items():
                req_marker = " (required)" if pname in required else ""
                ptype = pschema.get("type", "any")
                pdesc = pschema.get("description", "")
                param_lines.append(f"      {pname} [{ptype}]{req_marker}: {pdesc}")
            params_text = "\n".join(param_lines) if param_lines else "      (none)"
            lines.append(
                f"  - {tool}\n"
                f"    Description: {desc}\n"
                f"    Parameters:\n{params_text}\n"
                f"    Selection rationale: {rationale}"
            )
        return "\n".join(lines)

    def _deps_summary(self, dependencies: List[Dict[str, str]]) -> str:
        """Human-readable dependency matrix for the assembler prompt."""
        if not dependencies:
            return "  (no inter-node dependencies — all nodes can run in parallel)"
        lines = []
        for d in dependencies:
            lines.append(
                f"  {d['producer']}  →  {d['consumer']}\n"
                f"    output_field = {d['output_field']}   input_param = {d['input_param']}\n"
                f"    reasoning: {d['reasoning']}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Phase 1: Node selector
    # ------------------------------------------------------------------

    async def _node_selector_node(self, state: DecomposeState) -> DecomposeState:
        model = self._bind_sampling(self.base_model, state)
        model = model.bind(response_format=_NODE_SELECTION_RESPONSE_FORMAT)

        # For MiniMax models use their JSON tool format; otherwise plain text.
        if self._is_minimax_model():
            minimax_tools_json = MCPConnector.format_tools_for_minimax_prompt(self.all_tools)
            tools_section = (
                "The available tools are listed below as a JSON array.\n"
                "Each tool name uses '__' as a separator between server and tool name\n"
                "(e.g. 'BioMCP__gene_getter' corresponds to 'BioMCP:gene_getter').\n\n"
                f"AVAILABLE TOOLS:\n```json\n{minimax_tools_json}\n```\n\n"
                "Use the colon-separated form (e.g. 'BioMCP:gene_getter') in your output."
            )
        else:
            tool_descriptions = MCPConnector.format_tools_for_prompt(self.all_tools)
            tools_section = f"AVAILABLE TOOLS:\n{tool_descriptions}"

        prompt = f"""You are a planning agent. Identify ALL tool functions needed to complete
the following task.

TASK:
{state['original_task']}

{tools_section}

Instructions:
- Select every tool that is directly necessary to complete the task end-to-end.
- Do NOT include tools that are redundant or only tangentially related.
- Use the exact colon-separated tool name as listed (e.g. "ServerName:tool_name").
- For each selected tool provide a one-line rationale explaining why it is needed.

Return ONLY valid JSON:
{{
  "selected_nodes": [
    {{"tool": "ServerName:tool_name", "rationale": "why this tool is needed"}}
  ]
}}"""

        logger.info(f"[Phase 1] Selecting nodes for: {state['original_task'][:80]}...")
        response = await model.ainvoke([_SYSTEM_MSG, HumanMessage(prompt)])
        usage = response.usage_metadata or {}

        try:
            parsed = _clean_and_parse_json(response.content)
            selected_nodes = parsed.get("selected_nodes", [])
        except Exception as e:
            logger.error(f"[Phase 1] Failed to parse node selection: {e}")
            selected_nodes = []

        logger.info(f"[Phase 1] Selected {len(selected_nodes)} node(s): {[n['tool'] for n in selected_nodes]}")

        return {
            **state,
            "selected_nodes":    selected_nodes,
            "prompt_tokens":     state["prompt_tokens"]     + usage.get("input_tokens", 0),
            "completion_tokens": state["completion_tokens"] + usage.get("output_tokens", 0),
            "total_tokens":      state["total_tokens"]      + usage.get("total_tokens", 0),
        }

    # ------------------------------------------------------------------
    # Phase 2: Dependency analyser
    # ------------------------------------------------------------------

    async def _dependency_analyzer_node(self, state: DecomposeState) -> DecomposeState:
        model = self._bind_sampling(self.base_model, state)
        model = model.bind(response_format=_DEPENDENCY_ANALYSIS_RESPONSE_FORMAT)

        nodeset_text = self._nodeset_summary(state["selected_nodes"])

        prompt = f"""You are a planning agent analysing data-flow dependencies between tool calls.

TASK:
{state['original_task']}

SELECTED TOOLS (nodeset):
{nodeset_text}

Your job: for every ordered pair (A, B) in the nodeset, determine whether A is an
IMMEDIATE upstream producer of a required input to B.

A is an immediate upstream producer of B when ALL of the following hold:
  1. B requires a specific input value to run.
  2. A's output is the direct, authoritative source of that value — not a constant,
     a user-supplied value, or something derivable without calling A first.
  3. No other node in the nodeset provides the same value.

For each such dependency record:
  producer     — the tool that generates the output
  consumer     — the tool that consumes it
  output_field — the specific field/key on the producer's output used by the consumer
  input_param  — the parameter name on the consumer that receives this value
  reasoning    — one sentence: why A is the immediate upstream producer of this
                 input for B, and why no other node in the set could provide it

If no immediate upstream dependency exists between a pair, omit it entirely.

Return ONLY valid JSON:
{{
  "dependencies": [
    {{
      "producer":     "ServerName:tool_a",
      "consumer":     "ServerName:tool_b",
      "output_field": "result_field",
      "input_param":  "param_name",
      "reasoning":    "tool_b needs X which is only produced by tool_a"
    }}
  ]
}}"""

        logger.info(f"[Phase 2] Dependency analysis over {len(state['selected_nodes'])} node(s)")
        response = await model.ainvoke([_SYSTEM_MSG, HumanMessage(prompt)])
        usage = response.usage_metadata or {}

        try:
            parsed = _clean_and_parse_json(response.content)
            dependencies = parsed.get("dependencies", [])
        except Exception as e:
            logger.error(f"[Phase 2] Failed to parse dependency analysis: {e}")
            dependencies = []

        logger.info(f"[Phase 2] Identified {len(dependencies)} dependency edge(s)")

        return {
            **state,
            "dependencies":      dependencies,
            "prompt_tokens":     state["prompt_tokens"]     + usage.get("input_tokens", 0),
            "completion_tokens": state["completion_tokens"] + usage.get("output_tokens", 0),
            "total_tokens":      state["total_tokens"]      + usage.get("total_tokens", 0),
        }

    # ------------------------------------------------------------------
    # Phase 3: DAG assembler
    # ------------------------------------------------------------------

    async def _dag_assembler_node(self, state: DecomposeState) -> DecomposeState:
        model = self._bind_sampling(self.base_model, state)
        model = model.bind(response_format=_DEPENDENCY_GRAPH_RESPONSE_FORMAT)

        # Build few-shot block for structural examples.
        api_tools = MCPConnector.format_tools_for_api(self.all_tools)
        inventory = build_tool_inventory(api_tools)
        few_shot_block = format_few_shot_block(EXAMPLES, tool_inventory=inventory)

        nodeset_text = self._nodeset_summary(state["selected_nodes"])
        deps_text = self._deps_summary(state["dependencies"])

        # Inject validation error feedback on retry attempts.
        error_feedback = ""
        if state["attempt"] > 0 and not state["validation"].get("is_valid", True):
            errors = "\n".join(state["validation"].get("errors", []))
            error_feedback = (
                f"\n\nYour previous DAG was invalid. Fix these errors before responding:\n{errors}\n"
            )

        prompt = f"""You are a planning agent. Assemble a complete execution DAG from
the pre-analysed nodeset and dependency information provided below.

{few_shot_block}

TASK:
{state['original_task']}
{error_feedback}
PRE-SELECTED NODES:
{nodeset_text}

PRE-COMPUTED DEPENDENCIES (producer → consumer):
{deps_text}

Assembly rules:
- Assign each node a unique integer string id ("1", "2", ...).
- Set depends_on to the list of ids whose outputs this node directly consumes
  (derived from the PRE-COMPUTED DEPENDENCIES above).
- For parameter values that come from another node's output, use the reference
  syntax "{{node_id.output_field}}" (e.g. "{{"gene_id": "{{2.id}}"}}").
- Static or user-supplied parameter values may be filled in directly.
- Nodes with no dependency between them will execute in parallel.
- "filter" and "condition" are optional — only include them when the task requires it.
- The result must be a valid DAG: no cycles, no dangling depends_on references.
- Include ALL pre-selected nodes; do not drop, merge, or add tools.

Return ONLY valid JSON — no prose, no markdown fences:
{{
  "nodes": [
    {{
      "id": "1",
      "tool": "ServerName:tool_name",
      "parameters": {{}},
      "depends_on": [],
      "description": "one-line description of what this step does"
    }}
  ]
}}"""

        logger.info(
            f"[Phase 3] DAG assembly "
            f"(attempt {state['attempt'] + 1}/{state['max_retries']})"
        )
        response = await model.ainvoke([_SYSTEM_MSG, HumanMessage(prompt)])
        usage = response.usage_metadata or {}

        try:
            dependency_graph = _clean_and_parse_json(response.content)
        except Exception as e:
            logger.error(f"[Phase 3] Failed to parse DAG JSON: {e}")
            dependency_graph = {"nodes": [], "parse_error": str(e), "raw": response.content}

        return {
            **state,
            "dependency_graph":  dependency_graph,
            "prompt_tokens":     state["prompt_tokens"]     + usage.get("input_tokens", 0),
            "completion_tokens": state["completion_tokens"] + usage.get("output_tokens", 0),
            "total_tokens":      state["total_tokens"]      + usage.get("total_tokens", 0),
        }

    # ------------------------------------------------------------------
    # Validator
    # ------------------------------------------------------------------

    async def _validator_node(self, state: DecomposeState) -> DecomposeState:
        validation = validate_dag(state["dependency_graph"])

        logger.info(
            f"Attempt {state['attempt'] + 1}/{state['max_retries']} | "
            f"nodes={len(state['dependency_graph'].get('nodes', []))} | "
            f"valid={validation.is_valid}"
        )

        return {
            **state,
            "validation": validation.to_dict(),
            "attempt":    state["attempt"] + 1,
        }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def execute(
        self,
        task: str,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        use_native_tools: bool = False,
        use_structured_output: bool = False,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Generate a dependency graph via decompose-then-assemble planning.

        Three sequential LLM calls per run:
          1. Node selection  — identify which tools are needed.
          2. Dependency analysis — pairwise immediate-upstream-producer reasoning.
          3. DAG assembly    — build the full DAG JSON from the above.

        Retry logic (up to max_retries) applies to the DAG assembly phase only.
        On each failed validation the assembler is re-invoked with the error list
        appended to the prompt.

        Args:
            task:               Fuzzy task description shown to the agent.
            temperature:        Sampling temperature.  None uses the model default.
                                Ignored for OpenAI reasoning models.
            top_k:              Top-K sampling (OpenRouter native param).
            top_p:              Top-P nucleus sampling (0.0–1.0).
            use_native_tools:   Accepted for interface compatibility; not used by
                                this executor (tools are always embedded in the prompt).
            use_structured_output: Accepted for interface compatibility; structured
                                output is always enforced in this executor.
            max_retries:        Maximum DAG assembly attempts.  Defaults to 3.

        Returns:
            {
                "dependency_graph":  {"nodes": [...]},
                "validation":        dict,
                "selected_nodes":    list[dict],   # phase 1 output
                "dependencies":      list[dict],   # phase 2 output
                "tool_descriptions": "decompose_assemble",
                "temperature":       float | None,
                "top_k":             int | None,
                "top_p":             float | None,
                "prompt_tokens":     int,
                "completion_tokens": int,
                "total_tokens":      int,
            }
        """
        initial_state: DecomposeState = {
            "original_task":    task,
            "temperature":      temperature,
            "top_k":            top_k,
            "top_p":            top_p,
            "selected_nodes":   [],
            "dependencies":     [],
            "dependency_graph": {},
            "validation":       {"is_valid": True},   # placeholder before first validator run
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
            "selected_nodes":    final_state["selected_nodes"],
            "dependencies":      final_state["dependencies"],
            "tool_descriptions": "decompose_assemble",
            "temperature":       temperature,
            "top_k":             top_k,
            "top_p":             top_p,
            "prompt_tokens":     final_state["prompt_tokens"],
            "completion_tokens": final_state["completion_tokens"],
            "total_tokens":      final_state["total_tokens"],
        }
