#!/usr/bin/env python3
"""Run one entry from the consolidated ZINC-full experiment matrix."""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import yaml


def deep_merge(base, override):
    if isinstance(base, dict) and isinstance(override, dict):
        result = copy.deepcopy(base)
        for key, value in override.items():
            result[key] = deep_merge(result[key], value) if key in result else copy.deepcopy(value)
        return result
    return copy.deepcopy(override)


def load_matrix(path):
    with path.open("r", encoding="utf-8") as handle:
        matrix = yaml.safe_load(handle)
    if not isinstance(matrix, dict):
        raise ValueError("The experiment file must contain a mapping.")
    seeds, variants = matrix.get("seeds"), matrix.get("variants")
    if not isinstance(seeds, list) or not isinstance(variants, list):
        raise ValueError("The experiment file must define list-valued seeds and variants.")
    actual = len(seeds) * len(variants)
    expected = matrix.get("expected_runs")
    if expected is not None and int(expected) != actual:
        raise ValueError(f"Expected {expected} runs, but the matrix expands to {actual}.")
    return matrix


def expanded_runs(matrix):
    common = matrix.get("common") or {}
    runs = []
    for variant in matrix["variants"]:
        name = variant["name"]
        for seed in matrix["seeds"]:
            config = deep_merge(common, variant.get("config") or {})
            data = config.setdefault("data", {})
            data["seed"] = int(seed)
            data.setdefault("dataset_params", {})["dataset_seed"] = int(seed)
            run_name = f"{name}_s{seed}"
            config.update({
                "experiment_name": run_name,
                "db_collection": run_name,
                "run_no": len(runs),
                "base_run_name": name,
                "manuscript_label": variant.get("manuscript_label"),
                "variant_name": name,
            })
            runs.append(config)
    return runs


def choose_index(args, runs):
    if args.run is not None:
        matches = [i for i, config in enumerate(runs) if config["experiment_name"] == args.run]
        if len(matches) != 1:
            raise ValueError(f"Unknown or ambiguous run name: {args.run}")
        return matches[0]
    raw_index = args.index if args.index is not None else os.environ.get("SLURM_ARRAY_TASK_ID")
    if raw_index is None:
        raise ValueError("Supply --index, --run, or SLURM_ARRAY_TASK_ID.")
    index = int(raw_index)
    if index < 0 or index >= len(runs):
        raise ValueError(f"Run index {index} is outside 0..{len(runs) - 1}.")
    return index


def execute(config):
    # Delay heavy imports so --list works without importing PyTorch.
    from main_experiment import ExperimentWrapper

    required = ("project_name", "trainer_params", "data", "model", "optimization")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Run config is missing required keys: {missing}")
    experiment = ExperimentWrapper(init_all=False)
    data_config = copy.deepcopy(config["data"])
    experiment.init_dataset(_config=data_config, **data_config)
    experiment.init_model(**copy.deepcopy(config["model"]))
    experiment.init_optimizer(**copy.deepcopy(config["optimization"]))
    runtime = config.get("runtime") or {}
    return experiment.train(
        trainer_params=copy.deepcopy(config["trainer_params"]),
        project_name=config["project_name"],
        _config=config,
        notes=str(config.get("notes") or ""),
        ckpt_path=runtime.get("ckpt_path"),
        use_wandb=bool(runtime.get("use_wandb", False)),
        use_profiler=bool(runtime.get("use_profiler", False)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("experiment/ZINC_full_all.yaml"))
    parser.add_argument("--index", type=int)
    parser.add_argument("--run")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    matrix = load_matrix(args.config)
    runs = expanded_runs(matrix)
    if args.list:
        for index, config in enumerate(runs):
            print(f"{index:02d}\t{config['experiment_name']}\t{config.get('manuscript_label', '')}")
        return 0
    index = choose_index(args, runs)
    config = runs[index]
    print(f"Running matrix entry {index}/{len(runs) - 1}: {config['experiment_name']}", flush=True)
    if args.print_config:
        print(json.dumps(config, indent=2, sort_keys=True, default=str))
    execute(config)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
