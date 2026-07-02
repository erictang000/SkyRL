"""Pre-stage zai-org/GLM-4.7 (355B, ~667 GB BF16 checkpoint) + the GSM8K dataset onto every
GPU node's local disk, for the full-context Megatron throughput ablation.

Adapted from ``stage_nemotron_ultra.py`` (nemotron-3-ultra-550b-rl branch). Why this exists:
the model is ~667 GB and the cluster's shared ``/home`` is small, so the checkpoint must live on
each node's large node-local disk (``/mnt/local_storage``, ~24 TB). Multi-node training also needs
the data present on every node (each rank reads its data locally). This launches one Ray task per
distinct GPU node that (1) downloads the HF snapshot and (2) writes the GSM8K parquets, both under
``/mnt/local_storage``.

ROBUSTNESS: an earlier single-shot ``snapshot_download(max_workers=16)`` hung on 6/8 nodes with
dead TCP connections (no timeout) after ~40 min — likely Hub throttling from 128 concurrent
connections cluster-wide. Fix: run each download attempt in a *child process* killed after a hard
timeout, then retry — ``hf_transfer`` resumes from the ``.incomplete`` blobs, so every attempt makes
forward progress and a hung socket can never wedge the whole stage. Also fewer workers per node.

IMPORTANT: ``allow_patterns`` includes ``*.jinja`` (chat template) and ``*.py`` (custom modeling
code, ``Glm4MoeForCausalLM``).

Usage (from the head node, on a running Ray cluster):
    set -a; source /home/ray/default/SkyRL-private/.env.apex; set +a   # exports HF_TOKEN
    uv run --isolated --with ray --with huggingface_hub --with hf_transfer --with datasets \
        python examples/train/megatron/stage_glm47.py
"""

import os

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.state import list_nodes

REPO = "zai-org/GLM-4.7"
HF_HOME = "/mnt/local_storage/hf_cache"
DATA_DIR = "/mnt/local_storage/data/gsm8k"
CACHE_DIR = f"{HF_HOME}/hub"
ALLOW = ["*.safetensors", "*.json", "*.txt", "tokenizer*", "*.model", "*.jinja", "*.py"]
MAX_WORKERS = 6
ATTEMPT_TIMEOUT_S = 1200  # 20 min/attempt; a hang is killed and resumed
MAX_ATTEMPTS = 40

ray.init(address="auto", ignore_reinit_error=True, log_to_driver=True)

_ENV = {"HF_HOME": HF_HOME, "HF_HUB_ENABLE_HF_TRANSFER": "1"}
if os.environ.get("HF_TOKEN"):
    _ENV["HF_TOKEN"] = os.environ["HF_TOKEN"]


_CHILD_CODE = """
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(
    {repo!r}, cache_dir={cache_dir!r}, max_workers={max_workers},
    allow_patterns={allow!r},
)
print("SNAPSHOT_OK:" + p)
""".format(repo=REPO, cache_dir=CACHE_DIR, max_workers=MAX_WORKERS, allow=ALLOW)


@ray.remote(
    num_cpus=8,
    runtime_env={"pip": ["huggingface_hub", "hf_transfer", "datasets"], "env_vars": _ENV},
)
def stage(node_ip):
    import shutil
    import socket
    import subprocess
    import sys
    import time

    free_before_gb = shutil.disk_usage("/mnt/local_storage").free / 1024**3

    # 1) Model snapshot with a per-attempt subprocess watchdog. subprocess.run(timeout=...)
    #    SIGKILLs a hung child (and its hf_transfer threads); the next attempt resumes the
    #    .incomplete blobs, so a dead socket can never wedge the whole stage.
    model_path = None
    last_err = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            r = subprocess.run(
                [sys.executable, "-c", _CHILD_CODE],
                capture_output=True,
                text=True,
                timeout=ATTEMPT_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            last_err = f"attempt {attempt}: {ATTEMPT_TIMEOUT_S}s timeout -> killed, resuming"
            print(f"[{node_ip}] {last_err}", flush=True)
            continue
        ok_line = [ln for ln in (r.stdout or "").splitlines() if ln.startswith("SNAPSHOT_OK:")]
        if r.returncode == 0 and ok_line:
            model_path = ok_line[0].split("SNAPSHOT_OK:", 1)[1]
            break
        last_err = f"attempt {attempt}: rc={r.returncode} :: {(r.stderr or '')[-300:]}"
        print(f"[{node_ip}] {last_err}", flush=True)
        time.sleep(15)
    if model_path is None:
        return (socket.gethostname(), node_ip, "MODEL_FAILED", last_err, round(free_before_gb, 1), None)

    free_after_gb = shutil.disk_usage("/mnt/local_storage").free / 1024**3

    # 2) GSM8K parquets (tiny; one copy per node so every rank reads locally).
    try:
        import re

        import datasets

        os.makedirs(DATA_DIR, exist_ok=True)
        instruction = 'Let\'s think step by step and output the final answer after "####".'

        def to_row(example, idx, split):
            q = example["question"]
            sol = re.search(r"#### (\-?[0-9\.\,]+)", example["answer"]).group(0).split("#### ")[1].replace(",", "")
            return {
                "data_source": "openai/gsm8k",
                "prompt": [{"role": "user", "content": q + " " + instruction}],
                "env_class": "gsm8k",
                "reward_spec": {"method": "rule", "ground_truth": sol},
                "extra_info": {"split": split, "index": idx},
            }

        ds = datasets.load_dataset("openai/gsm8k", "main")
        ds["train"].map(lambda e, i: to_row(e, i, "train"), with_indices=True).to_parquet(f"{DATA_DIR}/train.parquet")
        ds["test"].map(lambda e, i: to_row(e, i, "test"), with_indices=True).to_parquet(
            f"{DATA_DIR}/validation.parquet"
        )
    except Exception as e:  # noqa: BLE001
        return (
            socket.gethostname(),
            node_ip,
            "DATA_FAILED",
            f"{type(e).__name__}: {str(e)[:300]}",
            round(free_before_gb, 1),
            round(free_after_gb, 1),
        )

    return (socket.gethostname(), node_ip, "DONE", model_path, round(free_before_gb, 1), round(free_after_gb, 1))


nodes = [
    n
    for n in list_nodes(detail=True)
    if (n.get("resources_total") or {}).get("GPU", 0) > 0 and n.get("state") == "ALIVE"
]
print("staging on %d GPU nodes: %s" % (len(nodes), [n["node_ip"] for n in nodes]), flush=True)
results = ray.get(
    [
        stage.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=n["node_id"], soft=False)).remote(
            n["node_ip"]
        )
        for n in nodes
    ]
)
print("\n===== RESULTS =====", flush=True)
ok = sum(1 for r in results if r[2] == "DONE")
for host, ip, status, detail, free_before, free_after in results:
    print(f"[{ip} {host}] {status}: free_before={free_before}GB free_after={free_after}GB :: {detail}", flush=True)
print(f"\n{ok}/{len(results)} nodes staged OK", flush=True)
