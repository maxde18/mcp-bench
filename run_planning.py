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

--- Fuzzy variations ---

  Generate multiple fuzzy phrasings of the same underlying task:
    python run_planning_benchmark.py --tasks ... --variations 3

  Variation 0 is always the original fuzzy_description from the task file.
  Variations 1+ are newly synthesised by re-running fuzzy generation on the
  same task_description. Requires ground_truth.task_description to be present
  (automatically available when using --tasks mode).

--- Output structure ---

  Results are a flat list of "runs", one entry per (task × variation).
  Each entry contains a "repetitions" list for the (temperature × repetition)
  sweep. The output can be fed directly back into --fuzzy-tasks for further
  runs, with all context (distraction_servers, ground_truth) preserved.

  {
    "experiment_config": { "variations": 3, "repetitions": 3, "temperatures": [0.5, 1.0] },
    "runs": [
      {
        "task_id": "...",
        "variation_id": 0,
        "fuzzy_description": "...",
        "distraction_servers": [...],
        "ground_truth": { ... },
        "repetitions": [
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

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
# Strip carriage returns from all env vars in case .env was saved with Windows line endings
for _k, _v in os.environ.items():
    if '\r' in _v:
        os.environ[_k] = _v.replace('\r', '')

from planning.agents.plan_only_executor import PlanOnlyExecutor
from runtime.benchmark.runner import BenchmarkRunner
from mcp_infra.connection_manager import ConnectionManager
from llm.factory import LLMFactory
from synthesis.task_synthesis import TaskSynthesizer
import config.config_loader as config_loader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_fuzzy_tasks(fuzzy_tasks_file: str) -> List[Dict[str, Any]]:
    """Load pre-generated fuzzy tasks from a JSON file.

    Accepts both the --fuzzy-tasks flat format ({ "tasks": [...] }) and
    planning results output ({ "runs": [...] }), making planning outputs
    directly re-usable as inputs.
    """
    with open(fuzzy_tasks_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Accept "tasks" (fuzzy-tasks format) or "runs" (planning results format)
    if isinstance(data, dict):
        raw = data.get("tasks", data.get("runs", data))
    else:
        raw = data

    tasks = []
    for entry in raw:
        tasks.append(
            {
                "task_id":           entry["task_id"],
                "variation_id":      entry.get("variation_id", 0),
                "server_name":       entry["server_name"],
                "fuzzy_description": entry["fuzzy_description"],
                "task_data": {
                    "distraction_servers": entry.get("distraction_servers", []),
                    "task_description":    (entry.get("ground_truth") or {}).get("task_description", ""),
                    "dependency_analysis": (entry.get("ground_truth") or {}).get("dependency_analysis", ""),
                },
                "ground_truth": entry.get("ground_truth"),
            }
        )
    return tasks


async def run(
    tasks_file: Optional[str],
    fuzzy_tasks_file: Optional[str],
    output_dir: str,
    model_name: str,
    temperatures: List[Optional[float]],
    repetitions: int,
    task_limit: Optional[int],
    variations: int,
    native_tools: bool = False,
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

    total_runs = len(tasks) * variations * len(temperatures) * repetitions
    logger.info(
        f"Experiment: {len(tasks)} tasks × {variations} variation(s) "
        f"× {len(temperatures)} temperature(s) × {repetitions} repetition(s) "
        f"= {total_runs} total LLM calls"
    )

    # ------------------------------------------------------------------
    # Main loop: for each task, resolve servers once, generate fuzzy
    # variations, then run all (variation × temperature × repetition)
    # combinations inside the same connection.
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    run_timestamp = datetime.now().isoformat()

    def save_runs(runs: List[Dict[str, Any]]) -> None:
        """Write current runs to disk, grouped by server count."""
        by_server_count: Dict[int, List] = {}
        for entry in runs:
            n = len(entry.get("server_name", "").split("+"))
            by_server_count.setdefault(n, []).append(entry)

        for n_servers, server_runs in by_server_count.items():
            successful = sum(
                1 for e in server_runs for r in e.get("repetitions", []) if r.get("status") == "success"
            )
            output = {
                "run_timestamp": run_timestamp,
                "model":         model_name,
                "source":        source_label,
                "experiment_config": {
                    "variations":    variations,
                    "repetitions":   repetitions,
                    "temperatures":  temperatures,
                    "total_tasks":   len(tasks),
                    "total_runs":    total_runs,
                    "native_tools":  native_tools,
                },
                "summary": {
                    "successful_runs": successful,
                    "failed_runs":     sum(len(e.get("repetitions", [])) for e in server_runs) - successful,
                },
                "runs": server_runs,
            }
            out_path = os.path.join(output_dir, f"{n_servers}server.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

    runs: List[Dict[str, Any]] = []
    completed = 0
    synthesizer = TaskSynthesizer(llm_provider)

    for task in tasks:
        task_id           = task["task_id"]
        base_variation_id = task.get("variation_id", 0)
        server_name       = task["server_name"]
        fuzzy_description = task["fuzzy_description"]
        task_data         = task.get("task_data", {})
        ground_truth      = task.get("ground_truth")

        server_config_result = await runner._prepare_server_configs(
            server_name, servers_info, task_data
        )
        if server_config_result["status"] == "failed":
            logger.error(f"  Server config failed for {task_id}: {server_config_result['error']}")
            runs.append(
                {
                    "task_id":     task_id,
                    "variation_id": base_variation_id,
                    "server_name": server_name,
                    "status":      "failed",
                    "error":       server_config_result["error"],
                    "repetitions": [],
                }
            )
            continue

        # Open one connection per task and reuse it for all variations/runs
        try:
            async with ConnectionManager(
                server_config_result["all_server_configs"],
                runner.filter_problematic_tools,
            ) as conn_mgr:
                if not conn_mgr.all_tools:
                    raise RuntimeError("No tools discovered from any server")

                executor = PlanOnlyExecutor(llm_provider, conn_mgr.all_tools)

                # Build list of (variation_id, fuzzy_description) to run.
                # Variation 0 is always the original fuzzy description.
                # Additional variations are generated here using TaskSynthesizer.
                fuzzy_variations: List[tuple] = [(base_variation_id, fuzzy_description)]

                if variations > 1:
                    source_task_description = (ground_truth or {}).get("task_description", "")
                    if not source_task_description:
                        logger.warning(
                            f"  Cannot generate additional variations for {task_id}: "
                            f"no task_description in ground_truth. Only variation 0 will run."
                        )
                    else:
                        for i in range(1, variations):
                            logger.info(f"  Generating fuzzy variation {i}/{variations - 1} for {task_id}")
                            new_fuzzy = await synthesizer.generate_fuzzy_version(
                                source_task_description,
                                conn_mgr.all_tools,
                                server_name,
                            )
                            if new_fuzzy:
                                fuzzy_variations.append((base_variation_id + i, new_fuzzy))
                            else:
                                logger.warning(f"  Failed to generate fuzzy variation {i} for {task_id}, skipping")

                for (variation_id, fuzzy_desc) in fuzzy_variations:
                    variation_entry: Dict[str, Any] = {
                        "task_id":            task_id,
                        "variation_id":       variation_id,
                        "server_name":        server_name,
                        "fuzzy_description":  fuzzy_desc,
                        "distraction_servers": task_data.get("distraction_servers", []),
                        "repetitions":        [],
                    }
                    if ground_truth:
                        variation_entry["ground_truth"] = ground_truth

                    for temperature in temperatures:
                        for rep in range(repetitions):
                            completed += 1
                            temp_label = f"T={temperature}" if temperature is not None else "T=default"
                            logger.info(
                                f"[{completed}/{total_runs}] {task_id} | "
                                f"var {variation_id} | {temp_label} | rep {rep + 1}/{repetitions}"
                            )

                            try:
                                plan_result = await executor.execute(
                                    fuzzy_desc,
                                    temperature=temperature,
                                    use_native_tools=native_tools,
                                )
                                variation_entry["repetitions"].append(
                                    {
                                        "temperature":    temperature,
                                        "repetition":     rep,
                                        "status":         "success",
                                        "generated_plan": plan_result["dependency_graph"],
                                        "validation":     plan_result["validation"],
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
                                variation_entry["repetitions"].append(
                                    {
                                        "temperature": temperature,
                                        "repetition":  rep,
                                        "status":      "failed",
                                        "error":       str(e),
                                    }
                                )

                    runs.append(variation_entry)
                    save_runs(runs)

        except Exception as e:
            logger.error(f"Connection failed for {task_id}: {e}")
            runs.append(
                {
                    "task_id":      task_id,
                    "variation_id": base_variation_id,
                    "server_name":  server_name,
                    "fuzzy_description": fuzzy_description,
                    "distraction_servers": task_data.get("distraction_servers", []),
                    "ground_truth": ground_truth,
                    "repetitions":  [{"temperature": None, "repetition": 0, "status": "failed", "error": str(e)}],
                }
            )
            save_runs(runs)

    # ------------------------------------------------------------------
    # Final save (also done incrementally after each variation)
    # ------------------------------------------------------------------
    save_runs(runs)
    total_successful = sum(
        1 for e in runs for r in e.get("repetitions", []) if r.get("status") == "success"
    )
    logger.info(f"Done. {total_successful}/{total_runs} runs saved to {output_dir}/")


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
        "--output-dir",
        default=None,
        help="Output directory (default: planning_results_<timestamp>/). Files inside are named <n>server.json.",
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
        "--variations",
        type=int,
        default=1,
        help=(
            "Number of fuzzy description variations to generate per task (default: 1). "
            "Variation 0 is always the original fuzzy description. "
            "Additional variations are generated by re-running fuzzy synthesis on the "
            "same task_description. Requires ground_truth.task_description to be present."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tasks (useful for quick tests).",
    )
    parser.add_argument(
        "--native-tools",
        action="store_true",
        default=False,
        help=(
            "Pass MCP tools via the OpenAI 'tools' API field instead of embedding "
            "them as text in the prompt.  Recommended for fine-tuned tool-calling "
            "models.  Defaults to False (prompt-injection mode)."
        ),
    )

    args = parser.parse_args()

    if not args.tasks and not args.fuzzy_tasks:
        args.tasks = "tasks/mcpbench_tasks_multi_2server_runner_format.json"

    # None in the temperatures list means "use model default (no param set)"
    temperatures: List[Optional[float]] = args.temperatures if args.temperatures else [None]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join("results", "planning", args.model, timestamp)

    asyncio.run(
        run(
            args.tasks,
            args.fuzzy_tasks,
            output_dir,
            args.model,
            temperatures,
            args.repetitions,
            args.limit,
            args.variations,
            native_tools=args.native_tools,
        )
    )


if __name__ == "__main__":
    main()
