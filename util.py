import gzip
import json
import os
import sys
from os import PathLike, path
from paramiko.client import SSHClient, AutoAddPolicy
from typing import List, Tuple, TypeAlias, Generator, Iterable
from huggingface_hub import list_datasets
from datasets import load_dataset, load_dataset_builder
from itertools import islice
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from functools import wraps, reduce
import shutil
from dataclasses import dataclass


def load_jsonl(path, open=open) -> List[dict]:
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def levenshtein_distance(s1: List[int], s2: List[int]) -> int:
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for index2, char2 in enumerate(s2):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(s1):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(
                    1
                    + min((distances[index1], distances[index1 + 1], new_distances[-1]))
                )
        distances = new_distances

    return distances[-1]


def truncate(sample: str) -> str:
    return (
        sample.strip().split("\n\n\n")[0]
        if "\n\n\n" in sample
        else sample.strip().split("```")[0]
    )


def tokenize_code(
    sample: str, tokenizer: PreTrainedTokenizerBase, length: int
) -> List[int]:
    return tokenizer.encode(sample)[:length] if length else tokenizer.encode(sample)


def get_edit_distance_distribution_star(
    samples: List[List[int]],
    gready_sample: List[int],
):
    num = []
    max_length = len(gready_sample)
    for sample in samples:
        num.append(levenshtein_distance(gready_sample, sample))
        max_length = max(max_length, len(sample))
    return num, max_length


def calculate_ratio(numbers: List[int], threshold: float) -> float:
    count = sum(1 for num in numbers if num <= threshold)
    total = len(numbers)
    ratio = count / total if total > 0 else 0
    return ratio


# Spefcific implementations


def gpto_get_samples(item: dict) -> List[str] | None:
    def unwrap(sample: str) -> str:
        unwrapped = sample.strip("```java\n").strip("```").strip()
        truncated = truncate(unwrapped)
        return truncated

    if not item["generation"]:
        return None
    samples = (c["message"]["content"] for c in item["generation"]["choices"])
    cleaned_samples = map(unwrap, samples)
    # replace "\n" with line break
    return list(cleaned_samples)


def gpto_get_samples_greedy(item: dict) -> str | None:
    samples = gpto_get_samples(item)
    return samples[0] if samples else None


def get_samples(item: dict) -> List[str] | None:
    return item["generation"]


def get_samples_greedy(item: dict) -> str | None:
    samples = get_samples(item)
    return samples[0] if samples else None


# SSH into the server "alvis1.c3se.chalmers.se", called "alvis1" in ssh config file.
# Then run "ls" to make sure the ssh connection works properly.
# The user is "adamhenr" and the password is "yhE5s3WH4r7s#%".

ALVIS_BASE_DIR = os.path.join("mimer", "elle-elle-aime")
LOCAL_BASE_DIR = os.path.join("..", "elle-elle-aime")
DATA_DIR = "data"

data_dir_path_map = {
    "GitBugJava": "gitbug-java",
    "Defects4J": "defects4j",
    "HumanEvalJava": "humaneval-java",
}


def ssh_alvis(commands: Iterable[str], base_path=ALVIS_BASE_DIR) -> str:
    with SSHClient() as ssh:
        ssh.set_missing_host_key_policy(AutoAddPolicy())
        ssh.connect(
            "alvis1.c3se.chalmers.se", username="adamhenr", password="yhE5s3WH4r7s#%"
        )
        cmd = " && ".join((f"cd {base_path}", *commands))
        stdin, stdout, stderr = ssh.exec_command(cmd)
        stdout_str = stdout.read().decode()
    return stdout_str


def write_alvis(
    path: str, data: bytes, replace=False, base_path=ALVIS_BASE_DIR
) -> bool:
    remote_path = os.path.join(base_path, path)
    with SSHClient() as ssh:
        ssh.set_missing_host_key_policy(AutoAddPolicy())
        ssh.connect(
            "alvis1.c3se.chalmers.se", username="adamhenr", password="yhE5s3WH4r7s#%"
        )
        sftp = ssh.open_sftp()
        file_alrad_exists = os.path.basename(remote_path) in sftp.listdir(
            os.path.dirname(remote_path)
        )
        if not replace and file_alrad_exists:
            raise ValueError(f"File {remote_path} already exists.")
        with sftp.open(remote_path, "w") as f:
            f.write(data)
        return file_alrad_exists


def read_alvis(path: str, base_path=ALVIS_BASE_DIR) -> bytes:
    with SSHClient() as ssh:
        ssh.set_missing_host_key_policy(AutoAddPolicy())
        ssh.connect(
            "alvis1.c3se.chalmers.se", username="adamhenr", password="yhE5s3WH4r7s#%"
        )
        sftp = ssh.open_sftp()
        remote_path = os.path.join(base_path, path)
        with sftp.open(remote_path, "r") as f:
            return f.read()


@dataclass
class JobFilesInfo:
    DATA_DATASET_DIR: str
    DATA_DATASET_GREEDY_DIR: str
    DATA_DATASET_MULTIPLE_DIR: str
    # Filenames
    samples_file: str
    candidates_greedy_file: str
    candidates_multiple_file: str
    # Paths
    samples_data_dir_path: str
    candidates_greedy_data_dir_path: str
    candidates_multiple_data_dir_path: str
    samples_alvis_path: str
    candidates_multiple_alvis_path: str
    candidates_greedy_alvis_path: str
    # Exists?
    samples_exists: bool
    candidates_greedy_exists: bool
    candidates_multiple_exists: bool
    samples_exists_on_alvis: bool
    candidates_greedy_exists_on_alvis: bool
    candidates_multiple_exists_on_alvis: bool


def get_jobfiles_info(
    dataset: str,
    method: str,
    patch_strategy: str,
    candidate_model: str,
    temperature: str = "1.0",
) -> JobFilesInfo:
    # Spec for running the generation and candidate generation

    DATASET_DIR = os.path.join("data", data_dir_path_map[dataset])
    GENERATED_DATA_DIR = os.path.join(DATASET_DIR, candidate_model)
    DATA_DATASET_GREEDY_DIR = os.path.join(GENERATED_DATA_DIR, "greedy")
    DATA_DATASET_MULTIPLE_DIR = os.path.join(GENERATED_DATA_DIR, "multiple")

    samples_file = f"samples_{dataset}_{method}_.jsonl"

    # Candidates
    candidate_greedy_kwargs = [
        ("model_name", candidate_model.replace("/", ":")),
        ("generation_strategy", "beam_search"),
        ("num_return_sequences", 1),
    ]
    candidate_greedy_kwargs_str = "_".join(
        f"{k}={v}" for k, v in candidate_greedy_kwargs
    )
    candidates_greedy_file = f"candidates_{dataset}_{method}_{patch_strategy}_{candidate_greedy_kwargs_str}.jsonl"
    candidate_multiple_kwargs = [
        ("model_name", candidate_model.replace("/", ":")),
        ("temperature", temperature),
        ("generation_strategy", "sampling"),
        ("num_return_sequences", 10),
    ]
    candidate_multiple_kwargs_str = "_".join(
        f"{k}={v}" for k, v in candidate_multiple_kwargs
    )
    candidates_multiple_file = f"candidates_{dataset}_{method}_{patch_strategy}_{candidate_multiple_kwargs_str}.jsonl"

    # Set up folders
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(GENERATED_DATA_DIR, exist_ok=True)
    os.makedirs(DATA_DATASET_GREEDY_DIR, exist_ok=True)
    os.makedirs(DATA_DATASET_MULTIPLE_DIR, exist_ok=True)
    # Data dir files
    samples_data_dir_path = os.path.join(DATASET_DIR, samples_file)
    samples_exists = os.path.exists(samples_data_dir_path)
    candidates_greedy_data_dir_path = os.path.join(
        DATA_DATASET_GREEDY_DIR, candidates_greedy_file
    )
    candidates_greedy_exists = os.path.exists(candidates_greedy_data_dir_path)
    candidates_multiple_data_dir_path = os.path.join(
        DATA_DATASET_MULTIPLE_DIR, candidates_multiple_file
    )
    candidates_multiple_exists = os.path.exists(candidates_multiple_data_dir_path)
    # Alvis files
    samples_alvis_path = os.path.join(ALVIS_BASE_DIR, samples_file)
    samples_exists_on_alvis = bool(ssh_alvis([f"ls {samples_file}.gz"]))
    candidates_greedy_alvis_path = os.path.join(ALVIS_BASE_DIR, candidates_greedy_file)
    candidates_greedy_exists_on_alvis = bool(
        ssh_alvis([f"ls {candidates_greedy_file}"])
    )
    candidates_multiple_alvis_path = os.path.join(
        ALVIS_BASE_DIR, candidates_multiple_file
    )
    candidates_multiple_exists_on_alvis = bool(
        ssh_alvis([f"ls {candidates_multiple_file}"])
    )

    return JobFilesInfo(
        DATA_DATASET_DIR=DATASET_DIR,
        DATA_DATASET_GREEDY_DIR=DATA_DATASET_GREEDY_DIR,
        DATA_DATASET_MULTIPLE_DIR=DATA_DATASET_MULTIPLE_DIR,
        samples_file=samples_file,
        candidates_greedy_file=candidates_greedy_file,
        candidates_multiple_file=candidates_multiple_file,
        samples_data_dir_path=samples_data_dir_path,
        candidates_greedy_data_dir_path=candidates_greedy_data_dir_path,
        candidates_multiple_data_dir_path=candidates_multiple_data_dir_path,
        samples_alvis_path=samples_alvis_path,
        candidates_multiple_alvis_path=candidates_multiple_alvis_path,
        candidates_greedy_alvis_path=candidates_greedy_alvis_path,
        samples_exists=samples_exists,
        candidates_greedy_exists=candidates_greedy_exists,
        candidates_multiple_exists=candidates_multiple_exists,
        samples_exists_on_alvis=samples_exists_on_alvis,
        candidates_greedy_exists_on_alvis=candidates_greedy_exists_on_alvis,
        candidates_multiple_exists_on_alvis=candidates_multiple_exists_on_alvis,
    )
