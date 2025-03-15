import gzip
import json
import logging
import os
import sys
from typing import Dict, Iterable, List, Optional

import fire

ROOT_DIR = "results"


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, "rt") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def prepare_filtered_evaluations(
    evaluations_path: str,
    bug_subset_list_path: str,
    output_dir: str,
):
    evaluations_file_name = os.path.basename(evaluations_path)
    output_file = os.path.join(output_dir, evaluations_file_name)
    # Recursively create the output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(bug_subset_list_path) as f:
        bug_subset_list = f.read().splitlines()
    bug_subset = set(bug_subset_list)
    evaluations = stream_jsonl(evaluations_path)
    with open(output_file, "w") as out_file:
        for e in evaluations:
            if e["identifier"] not in bug_subset:
                out_file.write(json.dumps(e) + "\n")


def run_generate_statistics(
    benchmark: str,
    evaluations_path: str,
):
    # Run the script `export_results.py` with `../elle-elle-aime` as root dir on the `evaluation_path` file to generate the statistics
    evaluations_path = os.path.abspath(evaluations_path)
    os.system(
        f"cd '../elle-elle-aime' && python export_results.py {benchmark} {evaluations_path}",
    )


def get_folders_from_path(path: os.PathLike) -> List[str]:
    # Get list of strings from folders inside path, ignore files
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]


def get_files_from_path(path: os.PathLike) -> List[str]:
    # Get list of strings from files inside path, ignore folders
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def get_models_folders() -> Iterable[str]:
    # Get list of strings from folders inside root_dir, ignore files
    return get_folders_from_path(ROOT_DIR)


def get_statistics_file_abspaths(
    root_dir: os.PathLike, model: str, benchmark: str
) -> os.PathLike | None:
    # Skip if benchmark folder does not exist
    if not os.path.exists(os.path.join(root_dir, model, benchmark)):
        return None
    statistics_files = [
        file
        for file in get_files_from_path(os.path.join(root_dir, model, benchmark))
        if file.startswith("statistics_") and file.endswith(".json")
    ]
    statistics_file = statistics_files[0] if statistics_files else None
    if not statistics_file:
        return None
    path = os.path.join(root_dir, model, benchmark, statistics_file)
    return os.path.abspath(path)


def entry_point():
    out_dir_root = "has-sources-results"
    model_folders = get_models_folders()
    for model_folder in model_folders:
        logging.info(f"Processing model {model_folder}")
        # For each benchmark folder in the model folder
        model_folder_path = os.path.join(ROOT_DIR, model_folder)
        benchmarks = get_folders_from_path(model_folder_path)
        for benchmark in benchmarks:
            benchmark_folder_path = os.path.join(model_folder_path, benchmark)
            # Get .jsonl file from the benchmark folder that starts with `evaluation_`
            evaluation_files = [
                file
                for file in get_files_from_path(benchmark_folder_path)
                if file.startswith("evaluation_")
            ]
            evaluation_file = evaluation_files[0] if evaluation_files else None
            if evaluation_file:
                evaluation_file_path = os.path.join(
                    benchmark_folder_path, evaluation_file
                )
                # Read the file and filter the evaluations
                out_dir = os.path.join(out_dir_root, model_folder, benchmark)
                if not os.path.exists(os.path.join(out_dir, evaluation_file)):
                    prepare_filtered_evaluations(
                        evaluation_file_path, f"{benchmark}-has-sources.txt", out_dir
                    )
                statistics_files = [
                    file
                    for file in get_files_from_path(out_dir)
                    if file.startswith("statistics_")
                ]
                statistics_file = statistics_files[0] if statistics_files else None
                if not statistics_file:
                    # Generate statistics
                    run_generate_statistics(
                        benchmark,
                        os.path.abspath(os.path.join(out_dir, evaluation_file)),
                    )

    # Generate table with statistics for all models
    models = list(get_models_folders())
    defect4j_results = []
    defect4j_has_sources_results = []
    gitbugjava_results = []
    gitbugjava_has_sources_results = []

    has_sources_root_dir = "has-sources-results"
    for model in models:
        path = get_statistics_file_abspaths(ROOT_DIR, model, "defects4j")
        if not path:
            defect4j_results.append(None)
            continue
        with open(path, "r") as f:
            data = json.load(f)
            defect4j_results.append(data["plausible@1"])
    for model in models:
        path = get_statistics_file_abspaths(has_sources_root_dir, model, "defects4j")
        if not path:
            defect4j_has_sources_results.append(None)
            continue
        with open(path, "r") as f:
            data = json.load(f)
            defect4j_has_sources_results.append(data["plausible@1"])

    for model in models:
        path = get_statistics_file_abspaths(ROOT_DIR, model, "gitbugjava")
        if not path:
            gitbugjava_results.append(None)
            continue
        with open(path, "r") as f:
            data = json.load(f)
            gitbugjava_results.append(data["plausible@1"])
    for model in models:
        path = get_statistics_file_abspaths(has_sources_root_dir, model, "gitbugjava")
        if not path:
            gitbugjava_has_sources_results.append(None)
            continue
        with open(path, "r") as f:
            data = json.load(f)
            gitbugjava_has_sources_results.append(data["plausible@1"])

    # Generate table with statistics for all models
    import pandas as pd

    df = pd.DataFrame(
        {
            "Model": models,
            "Plausible@1 Defects4J": defect4j_results,
            "Plausible@1 Defects4J (No Gemini Sources)": defect4j_has_sources_results,
            "Plausible@1 GitBugJava": gitbugjava_results,
            "Plausible@1 GitBugJava (No Gemini Sources)": gitbugjava_has_sources_results,
        }
    )
    df.to_csv("results.csv", index=False)


def main():
    logging.getLogger().setLevel(logging.INFO)
    fire.Fire(entry_point)


if __name__ == "__main__":
    sys.exit(main())
