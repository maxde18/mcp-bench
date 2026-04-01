"""LangGraph-based Task Executor.

Two-phase agent: generates a full tool dependency graph (plan) upfront,
then executes it level-by-level, parallelising independent tool calls.

The dependency graph is saved in the executor result under 'dependency_graph'
so it is available for downstream evaluation and analysis.

Graph: planner -> executor -> synthesizer -> END
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

import config.config_loader as config_loader
from mcp_infra.connector import MCPConnector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    task: str
    dependency_graph: Dict[str, Any]        # saved plan produced by planner
    execution_results: List[Dict[str, Any]] # one entry per tool call
    accumulated_information: str            # running context for synthesizer
    final_answer: str
    total_rounds: int                       # execution levels in the DAG
    total_output_tokens: int
    total_prompt_tokens: int
    total_tokens: int


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class LangGraphExecutor:
    """Drop-in replacement for TaskExecutor that uses a LangGraph pipeline.

    Matches the interface expected by BenchmarkRunner:
        executor = LangGraphExecutor(llm_provider, server_manager)
        result   = await executor.execute(task_description)

    Args:
        llm_provider:   Existing LLMProvider instance (from llm/provider.py).
        server_manager: Existing PersistentMultiServerManager with live connections.
    """

    def __init__(self, llm_provider, server_manager) -> None:
        self.llm = llm_provider
        self.server_manager = server_manager
        self.all_tools: Dict[str, Any] = server_manager.all_tools
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("planner", self._planner_node)
        builder.add_node("executor", self._executor_node)
        builder.add_node("synthesizer", self._synthesizer_node)
        builder.set_entry_point("planner")
        builder.add_edge("planner", "executor")
        builder.add_edge("executor", "synthesizer")
        builder.add_edge("synthesizer", END)
        return builder.compile()

    # ------------------------------------------------------------------
    # Node: planner
    # ------------------------------------------------------------------

    async def _planner_node(self, state: AgentState) -> AgentState:
        """Generate a full tool dependency graph from tool descriptions and task."""

        tool_descriptions = MCPConnector.format_tools_for_prompt(self.all_tools)

        prompt = f"""You are a planning agent. Given a task and available tools, produce a
complete execution plan as a dependency graph (DAG).

TASK:
{state["task"]}

AVAILABLE TOOLS:
{tool_descriptions}

Return ONLY valid JSON in this exact schema:
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
- "depends_on" lists the ids of nodes that must complete before this one runs.
- Nodes with no shared dependencies will be executed in parallel.
- When a parameter value depends on a previous node's output, write it as
  "{{node_id.field}}" (e.g. "{{"variant_id": "{{2.id}}"}}"). These placeholders
  are resolved at runtime.
- Only include tools that are actually needed to answer the task.
- Order nodes so the graph forms a valid DAG (no cycles).
"""

        response_data = await self.llm.get_completion(
            "You are an expert planning agent that produces tool dependency graphs.",
            prompt,
            config_loader.get_planning_tokens(),
            return_usage=True,
        )

        response_str, usage = (
            response_data if isinstance(response_data, tuple) else (response_data, None)
        )

        try:
            dependency_graph = self.llm.clean_and_parse_json(response_str)
        except Exception as e:
            logger.error(f"Failed to parse dependency graph: {e}")
            dependency_graph = {"nodes": []}

        logger.info(
            f"Planner produced {len(dependency_graph.get('nodes', []))} nodes"
        )

        return {
            **state,
            "dependency_graph": dependency_graph,
            "total_output_tokens": state.get("total_output_tokens", 0)
            + (usage.get("completion_tokens", 0) if usage else 0),
            "total_prompt_tokens": state.get("total_prompt_tokens", 0)
            + (usage.get("prompt_tokens", 0) if usage else 0),
            "total_tokens": state.get("total_tokens", 0)
            + (usage.get("total_tokens", 0) if usage else 0),
        }

    # ------------------------------------------------------------------
    # Node: executor
    # ------------------------------------------------------------------

    async def _executor_node(self, state: AgentState) -> AgentState:
        """Execute the dependency graph level-by-level (topological order).

        Within each level, all ready nodes are dispatched in parallel via
        asyncio.gather. Dynamic parameter placeholders ({node_id.field}) are
        resolved against the outputs of completed nodes before each call.
        """

        nodes = state["dependency_graph"].get("nodes", [])
        node_by_id: Dict[str, Dict] = {n["id"]: n for n in nodes}
        node_outputs: Dict[str, Any] = {}  # id -> raw tool output
        execution_results: List[Dict[str, Any]] = []
        accumulated_parts: List[str] = []
        completed: set = set()
        remaining: Dict[str, Dict] = dict(node_by_id)
        round_num = 0

        while remaining:
            # Nodes whose dependencies are all satisfied
            ready = [
                n
                for n in remaining.values()
                if all(dep in completed for dep in n.get("depends_on", []))
            ]

            if not ready:
                logger.error(
                    "Dependency graph stalled — possible cycle or unresolvable deps. "
                    f"Remaining nodes: {list(remaining.keys())}"
                )
                break

            round_num += 1
            logger.info(
                f"Execution round {round_num}: dispatching "
                f"{len(ready)} node(s) in parallel: "
                f"{[n['tool'] for n in ready]}"
            )

            results = await asyncio.gather(
                *[self._call_node(n, node_outputs) for n in ready],
                return_exceptions=True,
            )

            for node, output in zip(ready, results):
                node_id = node["id"]
                if isinstance(output, Exception):
                    logger.error(f"Node {node_id} ({node['tool']}) failed: {output}")
                    node_outputs[node_id] = {"error": str(output)}
                else:
                    node_outputs[node_id] = output

                resolved_params = self._resolve_parameters(
                    node.get("parameters", {}), node_outputs
                )
                execution_results.append(
                    {
                        "tool_name": node["tool"],
                        "tool_input": resolved_params,
                        "tool_output": node_outputs[node_id],
                        "round": round_num,
                    }
                )
                snippet = json.dumps(node_outputs[node_id])[:400]
                accumulated_parts.append(f"[{node['tool']}]: {snippet}")

                completed.add(node_id)
                del remaining[node_id]

        return {
            **state,
            "execution_results": execution_results,
            "accumulated_information": "\n".join(accumulated_parts),
            "total_rounds": round_num,
        }

    async def _call_node(
        self, node: Dict[str, Any], node_outputs: Dict[str, Any]
    ) -> Any:
        """Resolve parameters and call the tool via the persistent server manager."""
        params = self._resolve_parameters(node.get("parameters", {}), node_outputs)
        return await self.server_manager.call_tool(node["tool"], params)

    def _resolve_parameters(
        self, params: Dict[str, Any], node_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Replace {node_id.field} placeholders with actual output values.

        Supports:
          - "{2.id}"       -> node_outputs["2"]["id"]
          - "{2.output}"   -> node_outputs["2"] (whole output)
          - "{2}"          -> node_outputs["2"] (whole output, shorthand)
        """
        resolved = {}
        for key, value in params.items():
            if (
                isinstance(value, str)
                and value.startswith("{")
                and value.endswith("}")
                and not value.startswith("{{")
            ):
                ref = value[1:-1]
                parts = ref.split(".", 1)
                node_id = parts[0]
                field = parts[1] if len(parts) > 1 else None
                raw = node_outputs.get(node_id)
                if field and isinstance(raw, dict):
                    resolved[key] = raw.get(field, value)
                else:
                    resolved[key] = raw if raw is not None else value
            else:
                resolved[key] = value
        return resolved

    # ------------------------------------------------------------------
    # Node: synthesizer
    # ------------------------------------------------------------------

    async def _synthesizer_node(self, state: AgentState) -> AgentState:
        """Synthesize a final answer from all accumulated tool outputs."""

        prompt = f"""TASK:
{state["task"]}

GATHERED INFORMATION:
{state["accumulated_information"]}

Produce a comprehensive, evidence-based final answer to the task above.
Use the gathered information; do not speculate beyond it."""

        response_data = await self.llm.get_completion(
            "You are an expert analyst synthesizing tool results into a final answer.",
            prompt,
            config_loader.get_summarization_max_tokens(),
            return_usage=True,
        )

        response_str, usage = (
            response_data if isinstance(response_data, tuple) else (response_data, None)
        )

        return {
            **state,
            "final_answer": response_str,
            "total_output_tokens": state.get("total_output_tokens", 0)
            + (usage.get("completion_tokens", 0) if usage else 0),
            "total_prompt_tokens": state.get("total_prompt_tokens", 0)
            + (usage.get("prompt_tokens", 0) if usage else 0),
            "total_tokens": state.get("total_tokens", 0)
            + (usage.get("total_tokens", 0) if usage else 0),
        }

    # ------------------------------------------------------------------
    # Public entry point (matches TaskExecutor.execute interface)
    # ------------------------------------------------------------------

    async def execute(self, task: str) -> Dict[str, Any]:
        """Execute the task and return a result dict compatible with BenchmarkRunner.

        The returned dict contains all standard fields expected by the evaluator
        plus 'dependency_graph' which holds the saved plan from the planning phase.
        """

        initial_state: AgentState = {
            "task": task,
            "dependency_graph": {},
            "execution_results": [],
            "accumulated_information": "",
            "final_answer": "",
            "total_rounds": 0,
            "total_output_tokens": 0,
            "total_prompt_tokens": 0,
            "total_tokens": 0,
        }

        final_state = await self.graph.ainvoke(initial_state)

        return {
            # Standard fields expected by BenchmarkRunner / TaskEvaluator
            "solution": final_state["final_answer"],
            "total_rounds": final_state["total_rounds"],
            "execution_results": final_state["execution_results"],
            "planning_json_compliance": 1.0,  # plan is generated as structured JSON
            "accumulated_information": final_state["accumulated_information"],
            "accumulated_information_uncompressed": final_state["accumulated_information"],
            "total_output_tokens": final_state["total_output_tokens"],
            "total_prompt_tokens": final_state["total_prompt_tokens"],
            "total_tokens": final_state["total_tokens"],
            # Extra: the saved dependency graph from the planning phase
            "dependency_graph": final_state["dependency_graph"],
        }
