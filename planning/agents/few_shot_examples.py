"""
Few-shot examples for the DAG planning agent prompt.
Each example is a (task_description, expected_dag) tuple.

Schema fields:
  - id, tool, parameters, depends_on, description  (required)
  - filter    (optional) – post-execution filter expression on the node's output
  - condition (optional) – boolean gate; node is only executed when this evaluates to true
"""

import json

EXAMPLES = [
    # ── Example 1: National Parks + Google Maps + Weather ──────────────
    #    Pattern: linear chain with fan-out after geocode.
    #    No filters or conditions needed.
    {
        "task": (
            "Plan a weekend visit to a national park in California for hiking by evaluating "
            "weather conditions, park alerts, and nearby amenities. First, identify up to 5 "
            "national parks in California offering hiking opportunities using National Parks:findParks "
            "(params: stateCode=\"CA\", activities=\"hiking\", limit=5). Select the first park listed "
            "and retrieve its address via National Parks:getParkDetails. Convert the park's address "
            "into coordinates using Google Maps:maps_geocode. Use these coordinates to determine the "
            "nearest city via Google Maps:maps_reverse_geocode. Fetch current weather and 3-day "
            "forecast for that city using Weather Data tools. Check for active alerts or closures in "
            "the selected park via National Parks:getAlerts. Search for 4-star+ currently open dining "
            "options within 1 km of the park using Google Maps:search_nearby (radius=1000, minRating=4, "
            "openNow=true). Finally, calculate travel distance and duration from Los Angeles to the "
            "park using Google Maps:maps_distance_matrix (mode=driving) to plan arrival time."
        ),
        "dag": {
            "nodes": [
                {
                    "id": "1",
                    "tool": "National Parks:findParks",
                    "parameters": {"stateCode": "CA", "activities": "hiking", "limit": 5},
                    "depends_on": [],
                    "description": "Find up to 5 national parks in California that offer hiking"
                },
                {
                    "id": "2",
                    "tool": "National Parks:getParkDetails",
                    "parameters": {"parkCode": "{{1.parks[0].parkCode}}"},
                    "depends_on": ["1"],
                    "description": "Get address and details for the first returned park"
                },
                {
                    "id": "3",
                    "tool": "National Parks:getAlerts",
                    "parameters": {"parkCode": "{{1.parks[0].parkCode}}"},
                    "depends_on": ["1"],
                    "description": "Check active alerts or closures for the selected park"
                },
                {
                    "id": "4",
                    "tool": "Google Maps:maps_geocode",
                    "parameters": {"address": "{{2.address}}"},
                    "depends_on": ["2"],
                    "description": "Convert the park address into lat/lng coordinates"
                },
                {
                    "id": "5",
                    "tool": "Google Maps:maps_reverse_geocode",
                    "parameters": {"latitude": "{{4.latitude}}", "longitude": "{{4.longitude}}"},
                    "depends_on": ["4"],
                    "description": "Determine the nearest city from the park coordinates"
                },
                {
                    "id": "6",
                    "tool": "Weather Data:get_current_weather",
                    "parameters": {"city": "{{5.city}}"},
                    "depends_on": ["5"],
                    "description": "Fetch current weather conditions for the nearest city"
                },
                {
                    "id": "7",
                    "tool": "Weather Data:get_weather_forecast",
                    "parameters": {"city": "{{5.city}}", "days": 3},
                    "depends_on": ["5"],
                    "description": "Fetch a 3-day weather forecast for the nearest city"
                },
                {
                    "id": "8",
                    "tool": "Google Maps:search_nearby",
                    "parameters": {
                        "latitude": "{{4.latitude}}",
                        "longitude": "{{4.longitude}}",
                        "keyword": "restaurant",
                        "radius": 1000,
                        "minRating": 4,
                        "openNow": True
                    },
                    "depends_on": ["4"],
                    "description": "Search for 4-star+ open restaurants within 1 km of the park"
                },
                {
                    "id": "9",
                    "tool": "Google Maps:maps_distance_matrix",
                    "parameters": {
                        "origins": "Los Angeles, CA",
                        "destinations": "{{4.latitude}},{{4.longitude}}",
                        "mode": "driving"
                    },
                    "depends_on": ["4"],
                    "description": "Calculate driving distance and duration from LA to the park"
                }
            ]
        }
    },
    # ── Example 2: Hugging Face NER model + dataset + paper search ────
    #    Pattern: two independent search roots → parallel fan-out with
    #    post-filter → convergence into cross-server paper search →
    #    conditional PubMed fallback.
    {
        "task": (
            "An NLP researcher needs to select a lightweight, open-license transformer for named "
            "entity recognition (NER) and pair it with a well-established NER dataset, then confirm "
            "the dataset's popularity in publications and find an interactive Space for benchmarking.\n"
            "1. Search Hugging Face for up to 10 NER models (query=\"ner\", tags=\"token-classification\", limit=10).\n"
            "2. For each model, get its info and filter to license=\"apache-2.0\" and weight_size < 2 GB.\n"
            "3. Search for up to 5 NER datasets (query=\"ner\", tags=\"token-classification\", limit=5).\n"
            "4. For each dataset, get its info and record total number of examples across splits.\n"
            "5. Select the dataset with the most examples (SELECTED_DATASET).\n"
            "6. Cross-validate popularity: search arXiv for \"SELECTED_DATASET named entity recognition\" "
            "(max_results=5), read each paper and count mentions. If fewer than 3 arXiv papers mention "
            "it, also search PubMed.\n"
            "7. Search Hugging Face Spaces for \"ner-evaluation\" (sdk=\"gradio\", limit=3) and get details "
            "for the top result.\n"
            "8. Compile a final JSON report."
        ),
        "dag": {
            "nodes": [
                # ── Step 1-2: search models + get info with filter ──
                {
                    "id": "1",
                    "tool": "Hugging Face:search-models",
                    "parameters": {"query": "ner", "tags": "token-classification", "limit": 10},
                    "depends_on": [],
                    "description": "Search for up to 10 NER token-classification models on Hugging Face"
                },
                {
                    "id": "2",
                    "tool": "Hugging Face:search-datasets",
                    "parameters": {"query": "ner", "tags": "token-classification", "limit": 5},
                    "depends_on": [],
                    "description": "Search for up to 5 NER token-classification datasets on Hugging Face"
                },
                {
                    "id": "3a",
                    "tool": "Hugging Face:get-model-info",
                    "parameters": {"model_id": "{{1.results[0].model_id}}"},
                    "depends_on": ["1"],
                    "filter": "license == 'apache-2.0' AND weight_size < 2000000000",
                    "description": "Get info for the 1st model; keep only if apache-2.0 and < 2 GB"
                },
                {
                    "id": "3b",
                    "tool": "Hugging Face:get-model-info",
                    "parameters": {"model_id": "{{1.results[1].model_id}}"},
                    "depends_on": ["1"],
                    "filter": "license == 'apache-2.0' AND weight_size < 2000000000",
                    "description": "Get info for the 2nd model; keep only if apache-2.0 and < 2 GB"
                },
                {
                    "id": "3c",
                    "tool": "Hugging Face:get-model-info",
                    "parameters": {"model_id": "{{1.results[2].model_id}}"},
                    "depends_on": ["1"],
                    "filter": "license == 'apache-2.0' AND weight_size < 2000000000",
                    "description": "Get info for the 3rd model; keep only if apache-2.0 and < 2 GB"
                },
                {
                    "id": "3d",
                    "tool": "Hugging Face:get-model-info",
                    "parameters": {"model_id": "{{1.results[3].model_id}}"},
                    "depends_on": ["1"],
                    "filter": "license == 'apache-2.0' AND weight_size < 2000000000",
                    "description": "Get info for the 4th model; keep only if apache-2.0 and < 2 GB"
                },
                {
                    "id": "3e",
                    "tool": "Hugging Face:get-model-info",
                    "parameters": {"model_id": "{{1.results[4].model_id}}"},
                    "depends_on": ["1"],
                    "filter": "license == 'apache-2.0' AND weight_size < 2000000000",
                    "description": "Get info for the 5th model; keep only if apache-2.0 and < 2 GB"
                },
                {
                    "id": "3f",
                    "tool": "Hugging Face:get-model-info",
                    "parameters": {"model_id": "{{1.results[5].model_id}}"},
                    "depends_on": ["1"],
                    "filter": "license == 'apache-2.0' AND weight_size < 2000000000",
                    "description": "Get info for the 6th model; keep only if apache-2.0 and < 2 GB"
                },
                {
                    "id": "3g",
                    "tool": "Hugging Face:get-model-info",
                    "parameters": {"model_id": "{{1.results[6].model_id}}"},
                    "depends_on": ["1"],
                    "filter": "license == 'apache-2.0' AND weight_size < 2000000000",
                    "description": "Get info for the 7th model; keep only if apache-2.0 and < 2 GB"
                },
                {
                    "id": "3h",
                    "tool": "Hugging Face:get-model-info",
                    "parameters": {"model_id": "{{1.results[7].model_id}}"},
                    "depends_on": ["1"],
                    "filter": "license == 'apache-2.0' AND weight_size < 2000000000",
                    "description": "Get info for the 8th model; keep only if apache-2.0 and < 2 GB"
                },
                {
                    "id": "3i",
                    "tool": "Hugging Face:get-model-info",
                    "parameters": {"model_id": "{{1.results[8].model_id}}"},
                    "depends_on": ["1"],
                    "filter": "license == 'apache-2.0' AND weight_size < 2000000000",
                    "description": "Get info for the 9th model; keep only if apache-2.0 and < 2 GB"
                },
                {
                    "id": "3j",
                    "tool": "Hugging Face:get-model-info",
                    "parameters": {"model_id": "{{1.results[9].model_id}}"},
                    "depends_on": ["1"],
                    "filter": "license == 'apache-2.0' AND weight_size < 2000000000",
                    "description": "Get info for the 10th model; keep only if apache-2.0 and < 2 GB"
                },

                # ── Step 3-4: get dataset info (parallel) ──
                {
                    "id": "4a",
                    "tool": "Hugging Face:get-dataset-info",
                    "parameters": {"dataset_id": "{{2.results[0].dataset_id}}"},
                    "depends_on": ["2"],
                    "description": "Get info for the 1st dataset; record total examples across splits"
                },
                {
                    "id": "4b",
                    "tool": "Hugging Face:get-dataset-info",
                    "parameters": {"dataset_id": "{{2.results[1].dataset_id}}"},
                    "depends_on": ["2"],
                    "description": "Get info for the 2nd dataset; record total examples across splits"
                },
                {
                    "id": "4c",
                    "tool": "Hugging Face:get-dataset-info",
                    "parameters": {"dataset_id": "{{2.results[2].dataset_id}}"},
                    "depends_on": ["2"],
                    "description": "Get info for the 3rd dataset; record total examples across splits"
                },
                {
                    "id": "4d",
                    "tool": "Hugging Face:get-dataset-info",
                    "parameters": {"dataset_id": "{{2.results[3].dataset_id}}"},
                    "depends_on": ["2"],
                    "description": "Get info for the 4th dataset; record total examples across splits"
                },
                {
                    "id": "4e",
                    "tool": "Hugging Face:get-dataset-info",
                    "parameters": {"dataset_id": "{{2.results[4].dataset_id}}"},
                    "depends_on": ["2"],
                    "description": "Get info for the 5th dataset; record total examples across splits"
                },

                # ── Step 6: cross-validate via arXiv ──
                {
                    "id": "5",
                    "tool": "Paper Search:search_arxiv",
                    "parameters": {"query": "{{SELECTED_DATASET}} named entity recognition", "max_results": 5},
                    "depends_on": ["4a", "4b", "4c", "4d", "4e"],
                    "description": "Search arXiv for papers mentioning the dataset with the most examples"
                },
                {
                    "id": "6a",
                    "tool": "Paper Search:read_arxiv_paper",
                    "parameters": {"paper_id": "{{5.results[0].paper_id}}"},
                    "depends_on": ["5"],
                    "description": "Read the 1st arXiv paper and check for mentions of SELECTED_DATASET"
                },
                {
                    "id": "6b",
                    "tool": "Paper Search:read_arxiv_paper",
                    "parameters": {"paper_id": "{{5.results[1].paper_id}}"},
                    "depends_on": ["5"],
                    "description": "Read the 2nd arXiv paper and check for mentions of SELECTED_DATASET"
                },
                {
                    "id": "6c",
                    "tool": "Paper Search:read_arxiv_paper",
                    "parameters": {"paper_id": "{{5.results[2].paper_id}}"},
                    "depends_on": ["5"],
                    "description": "Read the 3rd arXiv paper and check for mentions of SELECTED_DATASET"
                },
                {
                    "id": "6d",
                    "tool": "Paper Search:read_arxiv_paper",
                    "parameters": {"paper_id": "{{5.results[3].paper_id}}"},
                    "depends_on": ["5"],
                    "description": "Read the 4th arXiv paper and check for mentions of SELECTED_DATASET"
                },
                {
                    "id": "6e",
                    "tool": "Paper Search:read_arxiv_paper",
                    "parameters": {"paper_id": "{{5.results[4].paper_id}}"},
                    "depends_on": ["5"],
                    "description": "Read the 5th arXiv paper and check for mentions of SELECTED_DATASET"
                },

                # ── Step 6c-d: conditional PubMed fallback ──
                {
                    "id": "7",
                    "tool": "Paper Search:search_pubmed",
                    "parameters": {"query": "{{SELECTED_DATASET}} named entity recognition", "max_results": 5},
                    "depends_on": ["6a", "6b", "6c", "6d", "6e"],
                    "condition": "count_mentions({{6a}}, {{6b}}, {{6c}}, {{6d}}, {{6e}}) < 3",
                    "description": "Search PubMed if fewer than 3 arXiv papers mention SELECTED_DATASET"
                },

                # ── Step 7-8: find interactive Space (independent branch) ──
                {
                    "id": "8",
                    "tool": "Hugging Face:search-spaces",
                    "parameters": {"query": "ner-evaluation", "sdk": "gradio", "limit": 3},
                    "depends_on": [],
                    "description": "Search for interactive Gradio-based NER evaluation Spaces on Hugging Face"
                },
                {
                    "id": "9",
                    "tool": "Hugging Face:get-space-info",
                    "parameters": {"space_id": "{{8.results[0].space_id}}"},
                    "depends_on": ["8"],
                    "description": "Get URL and description for the top NER evaluation Space"
                }
            ]
        }
    },
]


def format_few_shot_block(examples: list[dict]) -> str:
    """
    Returns a string ready to be inserted into the prompt,
    containing all few-shot examples.
    """
    lines = [
        "Here are examples of correct execution plans:\n"
    ]
    for i, ex in enumerate(examples, 1):
        lines.append(f"--- EXAMPLE {i} ---")
        lines.append(f"TASK:\n{ex['task']}\n")
        lines.append(f"EXPECTED OUTPUT:\n{json.dumps(ex['dag'], indent=2)}\n")
    return "\n".join(lines)


# ── Quick preview ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print(format_few_shot_block(EXAMPLES))
