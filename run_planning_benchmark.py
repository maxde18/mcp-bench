#!/usr/bin/env python3
"""Planning-only benchmark runner.

Connects to the required MCP servers to collect tool descriptions, passes a
fuzzy task description to PlanOnlyExecutor, and saves the generated dependency
graphs. No tools are executed.

Supports multiple repetitions per task and temperature sweeps, producing an
output structure suitable for DAG stability analysis and anomaly detection.

--- Input modes (mutually exclusive) ---

  1. Task file (default)
     Loads fuzzy descriptions from an existing benchmark task JSON.

       python run_planning_benchmark.py \\
           --tasks tasks/mcpbench_tasks_multi_2server_runner_format.json

  2. Fuzzy tasks file  (--fuzzy-tasks)
     Loads pre-generated fuzzy prompts from a separate JSON file.
     Required fields per entry: task_id, server_name, fuzzy_description.
     Optional field:            distraction_servers (list of server name strings).

       python run_planning_benchmark.py --fuzzy-tasks my_fuzzy_prompts.json

     Expected format:
       {
         "tasks": [
           {
             "task_id": "paper_search_biomcp_000",
             "server_name": "Paper Search+BioMCP",
             "fuzzy_description": "I need to understand...",
             "distraction_servers": ["Wikipedia", "Weather Data"]
           }
         ]
       }

--- Repetitions and temperature sweep ---

  Run 5 repetitions at default temperature:
    python run_planning_benchmark.py --tasks ... --repetitions 5

  Sweep across temperatures (3 repetitions each):
    python run_planning_benchmark.py --tasks ... --repetitions 3 --temperatures 0.0 0.5 1.0 1.5

--- Output structure ---

  Results are grouped by task. Each task contains a flat list of "runs"
  where each run records its temperature, repetition index, and generated plan:

  {
    "experiment_config": { "repetitions": 3, "temperatures": [0.5, 1.0] },
    "results": [
      {
        "task_id": "...",
        "fuzzy_description": "...",
        "ground_truth": { ... },         # present when loaded from --tasks
        "runs": [
          {
            "temperature": 0.5,
            "repetition": 0,
            "generated_plan": { "nodes": [...] },
            "token_usage": { ... },
            "status": "success"
          },
          ...
        ]
      }
    ]
  }
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.plan_only_executor import PlanOnlyExecutor
from benchmark.runner import BenchmarkRunner, ConnectionManager
from llm.factory import LLMFactory
import config.config_loader as config_loader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_fuzzy_tasks(fuzzy_tasks_file: str) -> List[Dict[str, Any]]:
    """Load pre-generated fuzzy tasks from a JSON file.

    Normalises each entry into the internal task shape used by the main loop.
    """
    with open(fuzzy_tasks_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw = data.get("tasks", data) if isinstance(data, dict) else data

    tasks = []
    for entry in raw:
        tasks.append(
            {
                "task_id":           entry["task_id"],
                "server_name":       entry["server_name"],
                "fuzzy_description": entry["fuzzy_description"],
                "task_data": {
                    "distraction_servers": entry.get("distraction_servers", [])
                },
                "ground_truth": None,
            }
        )
    return tasks


async def run(
    tasks_file: Optional[str],
    fuzzy_tasks_file: Optional[str],
    output_file: str,
    model_name: str,
    temperatures: List[Optional[float]],
    repetitions: int,
    task_limit: Optional[int],
) -> None:
    # ------------------------------------------------------------------
    # LLM provider
    # ------------------------------------------------------------------
    model_configs = LLMFactory.get_model_configs()
    if model_name not in model_configs:
        available = list(model_configs.keys())
        raise ValueError(
            f"Model '{model_name}' not found. Check your env vars. "
            f"Available: {available}"
        )
    llm_provider = await LLMFactory.create_llm_provider(model_configs[model_name])

    # ------------------------------------------------------------------
    # BenchmarkRunner — used for server config resolution helpers only
    # ------------------------------------------------------------------
    runner = BenchmarkRunner(
        tasks_file=tasks_file or "tasks/mcpbench_tasks_multi_2server_runner_format.json",
        use_fuzzy_descriptions=True,
        enable_distraction_servers=True,
    )
    servers_info = await runner.load_server_configs()
    runner.commands_config = await runner.load_commands_config()

    # ------------------------------------------------------------------
    # Load and normalise tasks
    # ------------------------------------------------------------------
    if fuzzy_tasks_file:
        tasks = load_fuzzy_tasks(fuzzy_tasks_file)
        source_label = fuzzy_tasks_file
        logger.info(f"Loaded {len(tasks)} tasks from fuzzy tasks file: {fuzzy_tasks_file}")
    else:
        raw_tasks = await runner.load_tasks()
        tasks = []
        for task_info in raw_tasks:
            task_exec_info = await runner._prepare_task_execution(task_info)
            task_data = task_exec_info["task_data"]
            tasks.append(
                {
                    "task_id":           task_exec_info["task_id"],
                    "server_name":       task_exec_info["server_name"],
                    "fuzzy_description": task_exec_info["task_description"],
                    "task_data":         task_data,
                    "ground_truth": {
                        "task_description":   task_data.get("task_description", ""),
                        "dependency_analysis": task_data.get("dependency_analysis", ""),
                    },
                }
            )
        source_label = tasks_file
        logger.info(f"Loaded {len(tasks)} tasks from task file: {tasks_file}")

    if task_limit:
        tasks = tasks[:task_limit]

    total_runs = len(tasks) * len(temperatures) * repetitions
    logger.info(
        f"Experiment: {len(tasks)} tasks × {len(temperatures)} temperature(s) "
        f"× {repetitions} repetition(s) = {total_runs} total LLM calls"
    )

    # ------------------------------------------------------------------
    # Main loop: for each task, resolve servers once, then run all
    # (temperature × repetition) combinations inside the same connection.
    # ------------------------------------------------------------------
    results: List[Dict[str, Any]] = []
    completed = 0

    for task in tasks:
        task_id          = task["task_id"]
        server_name      = task["server_name"]
        fuzzy_description = task["fuzzy_description"]
        task_data        = task.get("task_data", {})
        ground_truth     = task.get("ground_truth")

        server_config_result = await runner._prepare_server_configs(
            server_name, servers_info, task_data
        )
        if server_config_result["status"] == "failed":
            logger.error(f"  Server config failed for {task_id}: {server_config_result['error']}")
            results.append(
                {
                    "task_id":     task_id,
                    "server_name": server_name,
                    "status":      "failed",
                    "error":       server_config_result["error"],
                    "runs":        [],
                }
            )
            continue

        task_entry: Dict[str, Any] = {
            "task_id":          task_id,
            "server_name":      server_name,
            "fuzzy_description": fuzzy_description,
            "runs":             [],
        }
        if ground_truth:
            task_entry["ground_truth"] = ground_truth

        # Open one connection per task and reuse it for all runs
        try:
            async with ConnectionManager(
                server_config_result["all_server_configs"],
                runner.filter_problematic_tools,
            ) as conn_mgr:
                if not conn_mgr.all_tools:
                    raise RuntimeError("No tools discovered from any server")

                executor = PlanOnlyExecutor(llm_provider, conn_mgr.all_tools)

                for temperature in temperatures:
                    for rep in range(repetitions):
                        completed += 1
                        temp_label = f"T={temperature}" if temperature is not None else "T=default"
                        logger.info(
                            f"[{completed}/{total_runs}] {task_id} | {temp_label} | rep {rep + 1}/{repetitions}"
                        )

                        try:
                            plan_result = await executor.execute(
                                fuzzy_description, temperature=temperature
                            )
                            task_entry["runs"].append(
                                {
                                    "temperature":    temperature,
                                    "repetition":     rep,
                                    "status":         "success",
                                    "generated_plan": plan_result["dependency_graph"],
                                    "token_usage": {
                                        "prompt_tokens":     plan_result["prompt_tokens"],
                                        "completion_tokens": plan_result["completion_tokens"],
                                        "total_tokens":      plan_result["total_tokens"],
                                    },
                                }
                            )
                            logger.info(
                                f"  -> {len(plan_result['dependency_graph'].get('nodes', []))} nodes, "
                                f"{plan_result['total_tokens']} tokens"
                            )
                        except Exception as e:
                            logger.error(f"  Run failed: {e}")
                            task_entry["runs"].append(
                                {
                                    "temperature": temperature,
                                    "repetition":  rep,
                                    "status":      "failed",
                                    "error":       str(e),
                                }
                            )

        except Exception as e:
            logger.error(f"Connection failed for {task_id}: {e}")
            task_entry["runs"].append(
                {"temperature": None, "repetition": 0, "status": "failed", "error": str(e)}
            )

        results.append(task_entry)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    successful_runs = sum(
        1 for t in results for r in t.get("runs", []) if r.get("status") == "success"
    )
    output = {
        "run_timestamp": datetime.now().isoformat(),
        "model":         model_name,
        "source":        source_label,
        "experiment_config": {
            "repetitions":  repetitions,
            "temperatures": temperatures,
            "total_tasks":  len(tasks),
            "total_runs":   total_runs,
        },
        "summary": {
            "successful_runs": successful_runs,
            "failed_runs":     total_runs - successful_runs,
        },
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Done. {successful_runs}/{total_runs} runs saved to {output_file}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Planning-only benchmark with repetition and temperature sweep support.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--tasks",
        default=None,
        help="Benchmark task JSON file to extract fuzzy descriptions from.",
    )
    source.add_argument(
        "--fuzzy-tasks",
        default=None,
        help="Pre-generated fuzzy tasks JSON file.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file (default: planning_results_<timestamp>.json)",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4",
        help="Model name (must be registered in LLMFactory, default: claude-sonnet-4).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of times to repeat each task per temperature (default: 1).",
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Temperature values to sweep over (e.g. --temperatures 0.0 0.5 1.0). "
            "Defaults to the model's built-in default (no temperature set). "
            "Ignored for reasoning models (o-series)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tasks (useful for quick tests).",
    )

    args = parser.parse_args()

    if not args.tasks and not args.fuzzy_tasks:
        args.tasks = "tasks/mcpbench_tasks_multi_2server_runner_format.json"

    # None in the temperatures list means "use model default (no param set)"
    temperatures: List[Optional[float]] = args.temperatures if args.temperatures else [None]

    output_file = args.output or (
        f"planning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    asyncio.run(
        run(
            args.tasks,
            args.fuzzy_tasks,
            output_file,
            args.model,
            temperatures,
            args.repetitions,
            args.limit,
        )
    )


if __name__ == "__main__":
    main()
