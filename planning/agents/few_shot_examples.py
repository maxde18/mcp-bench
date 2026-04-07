"""
Few-shot examples for the DAG planning agent prompt.
Each example is a (task_description, expected_dag) tuple.

Schema fields:
  - id, tool, parameters, depends_on, description  (required)
  - filter    (optional) – post-execution filter expression on the node's output
  - condition (optional) – boolean gate; node is only executed when this evaluates to true

DESIGN PRINCIPLES (addressing tool-name hallucination):
  1. Examples use PLACEHOLDER tool names (e.g. "{{SERVER_A}}__action") that are
     explicitly called out as placeholders. The formatting function injects the
     real tool inventory separately, making the constraint "use ONLY these tools"
     unambiguous.
  2. Fan-out patterns (N parallel calls on list results) are shown with 2-3
     representative nodes + a comment, not exhaustively unrolled. This cuts
     token cost ~60% and avoids teaching the model to always emit 10 copies.
  3. Each example is annotated with the DAG *pattern* it demonstrates, so the
     model learns structure (chain, fan-out, conditional) rather than memorising
     surface tool names.
"""

import json
from typing import Optional


# ─────────────────────────────────────────────────────────────────────
#  EXAMPLES – using abstract placeholder names
# ─────────────────────────────────────────────────────────────────────

EXAMPLES = [
    # ── Example 1 ─────────────────────────────────────────────────────
    # PATTERN: linear chain → fan-out after geocode → independent branches
    # Demonstrates: sequential dependencies, parallel siblings, static params
    {
        "pattern": "chain → fan-out → independent branches",
        "task": (
            "Plan a weekend visit to a national park in California for hiking. "
            "Identify parks, get the first park's address, geocode it, then in "
            "parallel: reverse-geocode to find the nearest city and fetch weather, "
            "search for nearby restaurants, and calculate driving distance from LA."
        ),
        "dag": {
            "nodes": [
                {
                    "id": "1",
                    "tool": "ParkService__find_parks",
                    "parameters": {"stateCode": "CA", "activities": "hiking", "limit": 5},
                    "depends_on": [],
                    "description": "Find up to 5 California parks offering hiking"
                },
                {
                    "id": "2",
                    "tool": "ParkService__get_park_details",
                    "parameters": {"parkCode": "{{1.parks[0].parkCode}}"},
                    "depends_on": ["1"],
                    "description": "Get address/details for the first park"
                },
                {
                    "id": "3",
                    "tool": "ParkService__get_alerts",
                    "parameters": {"parkCode": "{{1.parks[0].parkCode}}"},
                    "depends_on": ["1"],
                    "description": "Check active alerts for the selected park"
                },
                {
                    "id": "4",
                    "tool": "Maps__geocode",
                    "parameters": {"address": "{{2.address}}"},
                    "depends_on": ["2"],
                    "description": "Convert park address to coordinates"
                },
                {
                    "id": "5",
                    "tool": "Maps__reverse_geocode",
                    "parameters": {"latitude": "{{4.lat}}", "longitude": "{{4.lng}}"},
                    "depends_on": ["4"],
                    "description": "Find nearest city from park coordinates"
                },
                {
                    "id": "6",
                    "tool": "Weather__get_current",
                    "parameters": {"city": "{{5.city}}"},
                    "depends_on": ["5"],
                    "description": "Current weather for the nearest city"
                },
                {
                    "id": "7",
                    "tool": "Weather__get_forecast",
                    "parameters": {"city": "{{5.city}}", "days": 3},
                    "depends_on": ["5"],
                    "description": "3-day forecast for the nearest city"
                },
                {
                    "id": "8",
                    "tool": "Maps__search_nearby",
                    "parameters": {
                        "latitude": "{{4.lat}}",
                        "longitude": "{{4.lng}}",
                        "keyword": "restaurant",
                        "radius": 1000,
                        "minRating": 4,
                        "openNow": True,
                    },
                    "depends_on": ["4"],
                    "description": "4-star+ open restaurants within 1 km of the park"
                },
                {
                    "id": "9",
                    "tool": "Maps__distance_matrix",
                    "parameters": {
                        "origins": "Los Angeles, CA",
                        "destinations": "{{4.lat}},{{4.lng}}",
                        "mode": "driving",
                    },
                    "depends_on": ["4"],
                    "description": "Driving distance/duration from LA to the park"
                },
            ]
        },
    },

    # ── Example 2 ─────────────────────────────────────────────────────
    # PATTERN: two independent roots → bounded fan-out with filter →
    #          convergence → conditional fallback → independent branch
    # Demonstrates: parallel roots, fan-out (compact), filter, condition
    {
        "pattern": "parallel roots → filtered fan-out → convergence → conditional fallback",
        "task": (
            "An NLP researcher wants to find a lightweight open-license NER model, "
            "pair it with the most popular NER dataset, cross-validate via paper "
            "search, and find a demo Space.\n"
            "1. Search models (query='ner', limit=10), get info for each, filter "
            "   to license='apache-2.0' and size < 2 GB.\n"
            "2. Search datasets (query='ner', limit=5), get info for each.\n"
            "3. Search papers for the dataset with most examples; if < 3 results, "
            "   fall back to a second paper source.\n"
            "4. Independently search for a demo Space."
        ),
        "dag": {
            "nodes": [
                # ── Two independent search roots ──
                {
                    "id": "1",
                    "tool": "ModelHub__search_models",
                    "parameters": {"query": "ner", "tags": "token-classification", "limit": 10},
                    "depends_on": [],
                    "description": "Search for up to 10 NER models"
                },
                {
                    "id": "2",
                    "tool": "ModelHub__search_datasets",
                    "parameters": {"query": "ner", "tags": "token-classification", "limit": 5},
                    "depends_on": [],
                    "description": "Search for up to 5 NER datasets"
                },

                # ── Fan-out: get model info (showing 2 of N; repeat pattern
                #    for results[2] … results[9] with ids 3c … 3j) ──
                {
                    "id": "3a",
                    "tool": "ModelHub__get_model_info",
                    "parameters": {"model_id": "{{1.results[0].model_id}}"},
                    "depends_on": ["1"],
                    "filter": "license == 'apache-2.0' AND weight_size < 2000000000",
                    "description": "Get info for 1st model; keep if apache-2.0 and < 2 GB"
                },
                {
                    "id": "3b",
                    "tool": "ModelHub__get_model_info",
                    "parameters": {"model_id": "{{1.results[1].model_id}}"},
                    "depends_on": ["1"],
                    "filter": "license == 'apache-2.0' AND weight_size < 2000000000",
                    "description": "Get info for 2nd model; keep if apache-2.0 and < 2 GB"
                },
                # ... nodes 3c-3j follow the same pattern for results[2]-[9]

                # ── Fan-out: get dataset info (showing 2 of N) ──
                {
                    "id": "4a",
                    "tool": "ModelHub__get_dataset_info",
                    "parameters": {"dataset_id": "{{2.results[0].dataset_id}}"},
                    "depends_on": ["2"],
                    "description": "Get info for 1st dataset; record total examples"
                },
                {
                    "id": "4b",
                    "tool": "ModelHub__get_dataset_info",
                    "parameters": {"dataset_id": "{{2.results[1].dataset_id}}"},
                    "depends_on": ["2"],
                    "description": "Get info for 2nd dataset; record total examples"
                },
                # ... nodes 4c-4e follow the same pattern for results[2]-[4]

                # ── Convergence: paper search on best dataset ──
                {
                    "id": "5",
                    "tool": "Papers__search",
                    "parameters": {
                        "query": "{{SELECTED_DATASET}} named entity recognition",
                        "max_results": 5,
                    },
                    "depends_on": ["4a", "4b"],  # in practice: ["4a".."4e"]
                    "description": "Search papers for the dataset with the most examples"
                },
                {
                    "id": "6a",
                    "tool": "Papers__read",
                    "parameters": {"paper_id": "{{5.results[0].paper_id}}"},
                    "depends_on": ["5"],
                    "description": "Read 1st paper; count mentions of SELECTED_DATASET"
                },
                {
                    "id": "6b",
                    "tool": "Papers__read",
                    "parameters": {"paper_id": "{{5.results[1].paper_id}}"},
                    "depends_on": ["5"],
                    "description": "Read 2nd paper; count mentions of SELECTED_DATASET"
                },
                # ... nodes 6c-6e follow the same pattern

                # ── Conditional fallback ──
                {
                    "id": "7",
                    "tool": "Papers__search_alt_source",
                    "parameters": {
                        "query": "{{SELECTED_DATASET}} named entity recognition",
                        "max_results": 5,
                    },
                    "depends_on": ["6a", "6b"],  # in practice: ["6a".."6e"]
                    "condition": "count_mentions(6a, 6b, ...) < 3",
                    "description": "Fallback paper search if fewer than 3 papers mention the dataset"
                },

                # ── Independent branch: demo Space ──
                {
                    "id": "8",
                    "tool": "ModelHub__search_spaces",
                    "parameters": {"query": "ner-evaluation", "sdk": "gradio", "limit": 3},
                    "depends_on": [],
                    "description": "Search for interactive NER evaluation Spaces"
                },
                {
                    "id": "9",
                    "tool": "ModelHub__get_space_info",
                    "parameters": {"space_id": "{{8.results[0].space_id}}"},
                    "depends_on": ["8"],
                    "description": "Get details for the top NER evaluation Space"
                },
            ]
        },
    },
]


# ─────────────────────────────────────────────────────────────────────
#  FORMATTING – injects tool inventory + examples into the prompt
# ─────────────────────────────────────────────────────────────────────

def build_tool_inventory(tools: list[dict]) -> str:
    """
    Build a concise tool inventory string from the OpenAI-format tools list.
    Each entry: "ServerName__tool_name – description"

    This is injected into the prompt so the model has an explicit,
    authoritative list of tool names to choose from.
    """
    lines = []
    for t in tools:
        func = t.get("function", t)
        name = func["name"]
        desc = func.get("description", "").split("\n")[0].strip()
        # Truncate long descriptions to keep inventory scannable
        if len(desc) > 120:
            desc = desc[:117] + "..."
        lines.append(f"  - {name}: {desc}")
    return "\n".join(lines)


def format_few_shot_block(
    examples: list[dict],
    tool_inventory: Optional[str] = None,
) -> str:
    """
    Returns the few-shot section ready for insertion into the system prompt.

    If tool_inventory is provided (recommended), it prepends a TOOL REGISTRY
    block with an explicit grounding constraint.
    """
    sections = []

    # ── Tool grounding constraint ──
    if tool_inventory:
        sections.append(
            "=== AVAILABLE TOOL REGISTRY ===\n"
            "You MUST use ONLY tool names from this registry in your plan.\n"
            "Do NOT invent, rename, or guess tool names. If no suitable tool "
            "exists for a step, state that explicitly in the node description "
            "and set tool to \"UNAVAILABLE\".\n\n"
            f"{tool_inventory}\n"
        )

    # ── Few-shot examples ──
    sections.append(
        "=== FEW-SHOT EXAMPLES ===\n"
        "The examples below illustrate correct DAG structure and patterns.\n"
        "IMPORTANT: The tool names in these examples are PLACEHOLDERS that "
        "demonstrate naming conventions and DAG structure. In your actual plan, "
        "you MUST substitute real tool names from the AVAILABLE TOOL REGISTRY above.\n"
        "Fan-out sections marked '... nodes Xc-Xn follow the same pattern' "
        "should be fully expanded in your output.\n"
    )

    for i, ex in enumerate(examples, 1):
        pattern = ex.get("pattern", "general")
        sections.append(f"--- EXAMPLE {i} (pattern: {pattern}) ---")
        sections.append(f"TASK:\n{ex['task']}\n")
        sections.append(f"PLAN:\n{json.dumps(ex['dag'], indent=2)}\n")

    return "\n".join(sections)


# ─────────────────────────────────────────────────────────────────────
#  CONVENIENCE: build the full prompt section from a tools list
# ─────────────────────────────────────────────────────────────────────

def build_prompt_section(
    tools: list[dict],
    examples: Optional[list[dict]] = None,
) -> str:
    """
    One-call helper: builds tool inventory + few-shot block.

    Usage in your agent setup:
        from few_shot_examples import EXAMPLES, build_prompt_section
        section = build_prompt_section(
            tools=request_payload["tools"],
            examples=EXAMPLES,
        )
        system_prompt = f"...preamble...\n\n{section}\n\n...rest of prompt..."
    """
    if examples is None:
        examples = EXAMPLES

    inventory = build_tool_inventory(tools)
    return format_few_shot_block(examples, tool_inventory=inventory)


# ─────────────────────────────────────────────────────────────────────
#  Quick preview
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Demo with some fake tools to show the output format
    demo_tools = [
        {"function": {"name": "Metropolitan Museum__search-museum-objects",
                       "description": "Search for objects in the Met Museum collection"}},
        {"function": {"name": "Metropolitan Museum__get-museum-object",
                       "description": "Get a museum object by its ID"}},
        {"function": {"name": "Huge Icons__search_icons",
                       "description": "Search for icons by name or tags"}},
    ]
    print(build_prompt_section(demo_tools, EXAMPLES))
