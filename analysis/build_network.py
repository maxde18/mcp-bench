"""
Build NetworkX DAGs from planning_results JSON files.

Each successful run produces a record with:
  - prompt metadata (task_id, fuzzy_description, source, model, temperature, repetition)
  - ground_truth_graph: DiGraph built from ground_truth.dependency_analysis.tool_chains
  - llm_graph:          DiGraph built from generated_plan.nodes / depends_on

Records are returned as a flat list so callers can groupby any field, e.g.:
  - same prompt, different runs  → group by task_id + repetition
  - different prompts, same GT   → group by source
"""

import json
import re
from pathlib import Path

import networkx as nx


# ---------------------------------------------------------------------------
# Ground truth graph
# ---------------------------------------------------------------------------

def _parse_chain(chain: str) -> list[list[str]]:
    """
    Parse a chain string into steps, each step being a list of tool names.
    '&' within a step = parallel tools (fork). Server prefix is propagated
    to siblings that lack one.

    "A → B → Server:foo & bar"  →  [["A"], ["B"], ["Server:foo", "Server:bar"]]
    """
    steps = []
    for part in chain.split("→"):
        tools = [t.strip() for t in part.split("&")]
        # Strip parenthetical annotations like "(with LA as fixed origin)"
        tools = [re.sub(r"\s*\(.*?\)", "", t).strip() for t in tools]
        tools = [t for t in tools if t]

        # Propagate server prefix to siblings that lack one
        if tools:
            prefix = None
            for t in tools:
                if ":" in t:
                    prefix = t.split(":")[0] + ":"
                    break
            if prefix:
                tools = [t if ":" in t else prefix + t for t in tools]

        if tools:
            steps.append(tools)
    return steps


def build_ground_truth_graph(task: dict) -> nx.DiGraph:
    """Build a DiGraph from ground_truth.dependency_analysis.tool_chains."""
    G = nx.DiGraph()
    dep = task.get("ground_truth", {}).get("dependency_analysis", {})
    if not isinstance(dep, dict):
        return G
    for chain in dep.get("tool_chains", []):
        steps = _parse_chain(chain)
        for i in range(len(steps) - 1):
            for parent in steps[i]:
                G.add_node(parent)
                for child in steps[i + 1]:
                    G.add_node(child)
                    G.add_edge(parent, child)
    return G


# ---------------------------------------------------------------------------
# LLM generated graph
# ---------------------------------------------------------------------------

def build_llm_graph(generated_plan: dict) -> nx.DiGraph:
    """
    Build a DiGraph from a generated_plan dict.

    Nodes use the 'tool' field as the node name.
    Edges follow depends_on: for each dependency id, add edge dependency→node.
    """
    G = nx.DiGraph()
    nodes = generated_plan.get("nodes", [])

    # Map node id → tool name
    id_to_tool = {n["id"]: n["tool"] for n in nodes}

    for node in nodes:
        tool = node["tool"]
        G.add_node(tool)
        for dep_id in node.get("depends_on", []):
            parent = id_to_tool.get(dep_id)
            if parent:
                G.add_node(parent)
                G.add_edge(parent, tool)

    return G


# ---------------------------------------------------------------------------
# Record extraction
# ---------------------------------------------------------------------------

def extract_records(planning_results: dict) -> list[dict]:
    """
    Return one record per successful run in a planning_results dict.

    Each record:
      task_id           str
      variation_id      int   ← distinguishes fuzz variants of the same task
      fuzzy_description str   ← use as prompt key for groupby
      source            str   ← originating task file
      model             str
      temperature       float | None
      repetition        int
      ground_truth_graph nx.DiGraph
      llm_graph          nx.DiGraph
    """
    records = []
    source = planning_results.get("source", "")
    model = planning_results.get("model", "")

    for task in planning_results.get("runs", []):
        task_id = task.get("task_id", "")
        variation_id = task.get("variation_id", 0)
        fuzzy_description = task.get("fuzzy_description", "")
        gt_graph = build_ground_truth_graph(task)

        for run in task.get("repetitions", []):
            if run.get("status") != "success":
                continue
            generated_plan = run.get("generated_plan")
            if not generated_plan:
                continue

            records.append({
                "task_id": task_id,
                "variation_id": variation_id,
                "fuzzy_description": fuzzy_description,
                "source": source,
                "model": model,
                "temperature": run.get("temperature"),
                "repetition": run.get("repetition"),
                "ground_truth_graph": gt_graph,
                "llm_graph": build_llm_graph(generated_plan),
            })

    return records


def load_records(json_path: str | Path) -> list[dict]:
    """Load a planning_results JSON and return its extracted records."""
    with open(json_path) as f:
        data = json.load(f)
    return extract_records(data)


def load_all_records(directory: str | Path = None) -> list[dict]:
    """Load all *.json files in a directory."""
    directory = Path(directory or Path(__file__).parent / "agent_plans_json")
    records = []
    for path in sorted(directory.glob("*.json")):
        records.extend(load_records(path))
    return records


# ---------------------------------------------------------------------------
# CLI preview
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    records = load_all_records()
    print(f"Total successful runs: {len(records)}\n")

    for r in records:
        gt = r["ground_truth_graph"]
        llm = r["llm_graph"]
        print(f"task_id    : {r['task_id']}")
        print(f"source     : {r['source']}")
        print(f"model      : {r['model']}")
        print(f"temperature: {r['temperature']}  repetition: {r['repetition']}")
        print(f"GT  nodes={sorted(gt.nodes)}  edges={sorted(gt.edges)}")
        print(f"LLM nodes={sorted(llm.nodes)}  edges={sorted(llm.edges)}")
        print()
