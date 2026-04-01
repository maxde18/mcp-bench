"""DAG validation for planning agent output.

Two-layer validation:
  1. Pydantic schema — checks required fields, types, and duplicate node IDs.
  2. Graph checks    — detects dangling depends_on references and cycles
                       (via Kahn's topological sort algorithm).

Usage:
    from planning.validation import validate_dag

    result = validate_dag(dependency_graph)
    # result.is_valid  -> bool
    # result.errors    -> list of human-readable error strings
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List

from pydantic import BaseModel, field_validator, model_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class DAGNode(BaseModel):
    id: str
    tool: str
    parameters: Dict[str, Any] = {}
    depends_on: List[str] = []
    description: str = ""


class DependencyGraph(BaseModel):
    nodes: List[DAGNode]

    @field_validator("nodes")
    @classmethod
    def unique_node_ids(cls, nodes: List[DAGNode]) -> List[DAGNode]:
        seen: set = set()
        duplicates: List[str] = []
        for node in nodes:
            if node.id in seen:
                duplicates.append(node.id)
            seen.add(node.id)
        if duplicates:
            raise ValueError(f"Duplicate node IDs: {duplicates}")
        return nodes


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

@dataclass
class DAGValidationResult:
    is_valid: bool
    schema_valid: bool
    has_cycle: bool
    dangling_deps: List[str]   # depends_on IDs that reference non-existent nodes
    duplicate_ids: List[str]   # node IDs that appear more than once
    node_count: int
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid":      self.is_valid,
            "schema_valid":  self.schema_valid,
            "has_cycle":     self.has_cycle,
            "dangling_deps": self.dangling_deps,
            "duplicate_ids": self.duplicate_ids,
            "node_count":    self.node_count,
            "errors":        self.errors,
        }


# ---------------------------------------------------------------------------
# Public validation function
# ---------------------------------------------------------------------------

def validate_dag(graph: Dict[str, Any]) -> DAGValidationResult:
    """Validate a dependency graph dict produced by the planning agent.

    Args:
        graph: The parsed ``dependency_graph`` dict, expected to have a
               ``"nodes"`` list.

    Returns:
        A :class:`DAGValidationResult` describing all issues found.
        ``is_valid`` is True only when schema is valid, no cycles exist,
        and no dangling dependency references are present.
    """
    errors: List[str] = []
    duplicate_ids: List[str] = []

    # ------------------------------------------------------------------
    # Layer 1: Pydantic schema validation
    # ------------------------------------------------------------------
    try:
        validated = DependencyGraph.model_validate(graph)
        schema_valid = True
        nodes = validated.nodes
    except Exception as e:
        schema_valid = False
        errors.append(f"Schema error: {e}")

        # Extract duplicate IDs reported by the validator if possible
        err_str = str(e)
        if "Duplicate node IDs" in err_str:
            try:
                # Pull the list out of the error message for the result field
                start = err_str.index("[")
                end = err_str.rindex("]") + 1
                import ast
                duplicate_ids = ast.literal_eval(err_str[start:end])
            except Exception:
                pass

        # Fall back to raw node list for graph checks below
        raw_nodes = graph.get("nodes", [])
        if not isinstance(raw_nodes, list):
            return DAGValidationResult(
                is_valid=False,
                schema_valid=False,
                has_cycle=False,
                dangling_deps=[],
                duplicate_ids=duplicate_ids,
                node_count=0,
                errors=errors,
            )
        # Build minimal objects so graph checks can still run
        nodes = []
        seen_ids: set = set()
        for n in raw_nodes:
            if isinstance(n, dict) and "id" in n and "tool" in n:
                if n["id"] in seen_ids:
                    if n["id"] not in duplicate_ids:
                        duplicate_ids.append(n["id"])
                else:
                    seen_ids.add(n["id"])
                    nodes.append(
                        DAGNode(
                            id=n["id"],
                            tool=n["tool"],
                            parameters=n.get("parameters", {}),
                            depends_on=n.get("depends_on", []),
                            description=n.get("description", ""),
                        )
                    )

    # ------------------------------------------------------------------
    # Layer 2: Graph structural checks
    # ------------------------------------------------------------------
    node_ids = {n.id for n in nodes}

    # 2a. Dangling references
    dangling: List[str] = []
    for node in nodes:
        for dep in node.depends_on:
            if dep not in node_ids and dep not in dangling:
                dangling.append(dep)
    if dangling:
        errors.append(f"depends_on references non-existent node IDs: {dangling}")

    # 2b. Cycle detection via Kahn's algorithm
    in_degree: Dict[str, int] = {n.id: 0 for n in nodes}
    adjacency: Dict[str, List[str]] = {n.id: [] for n in nodes}
    for node in nodes:
        for dep in node.depends_on:
            if dep in adjacency:           # skip dangling refs
                adjacency[dep].append(node.id)
                in_degree[node.id] += 1

    queue: deque = deque(nid for nid, deg in in_degree.items() if deg == 0)
    visited = 0
    while queue:
        nid = queue.popleft()
        visited += 1
        for neighbor in adjacency[nid]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    has_cycle = visited != len(nodes)
    if has_cycle:
        errors.append(
            f"Cycle detected: only {visited}/{len(nodes)} nodes are reachable "
            "in topological order."
        )

    is_valid = schema_valid and not has_cycle and not dangling

    result = DAGValidationResult(
        is_valid=is_valid,
        schema_valid=schema_valid,
        has_cycle=has_cycle,
        dangling_deps=dangling,
        duplicate_ids=duplicate_ids,
        node_count=len(nodes),
        errors=errors,
    )

    if is_valid:
        logger.info(f"DAG validation passed: {len(nodes)} nodes, no issues.")
    else:
        logger.warning(f"DAG validation failed: {errors}")

    return result
