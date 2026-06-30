"""02-message-activations: pre-response-token probe readout for the experiment-00 messages.

Reads the ~600 user messages from experiment 00, runs Qwen3.5-9B on Modal, extracts
each message's residual activation at the **pre-response token** (the assistant
header's final token), and projects it onto every emotion vector -> a self-contained
readout on the ``name-that-feeling-emotion-vectors`` Volume:

- ``02-message-activations/activations.safetensors`` -- raw pre-response activations (per layer).
- ``02-message-activations/readout.json`` -- per message: its original emotion + cluster
  (and frame/split/axis) plus the projection onto each emotion vector.

    uv run modal run experiments/02-message-activations/run.py::readout
    uv run modal volume get name-that-feeling-emotion-vectors /02-message-activations ./out
"""

import json
from pathlib import Path

import yaml

from name_that_feeling.emotion_vectors import app
from name_that_feeling.emotion_vectors.extraction import ActivationExtractor, project_messages

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[1]
RUN_NAME = "02-message-activations"

META_KEYS = ("id", "emotion", "cluster", "frame", "split", "eval_axis", "message")


def load_config() -> dict:
    return yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


def _load_messages(cfg: dict) -> tuple[list[str], list[dict]]:
    rows = json.loads((REPO_ROOT / cfg["messages_file"]).read_text(encoding="utf-8"))
    return [r["message"] for r in rows], [{k: r.get(k) for k in META_KEYS} for r in rows]


@app.local_entrypoint()
def extract() -> None:
    """GPU: extract each message's pre-response-token activation -> activations.safetensors."""
    cfg = load_config()
    messages, _ = _load_messages(cfg)
    print(f"Extract: {len(messages)} messages on {cfg['model_id']}")
    print(ActivationExtractor(model_id=cfg["model_id"]).extract_message_activations.remote(messages, cfg, RUN_NAME))


@app.local_entrypoint()
def project() -> None:
    """CPU: project the cached activations onto the emotion vectors -> readout.json (re-runnable)."""
    cfg = load_config()
    _, meta = _load_messages(cfg)
    res = project_messages.remote(meta, cfg, RUN_NAME)
    print(res)
    if res["missing"]:
        print(f"WARNING: {len(res['missing'])} message emotions have no vector: {', '.join(res['missing'])}")
    else:
        print("All message emotions have a vector (full coverage).")


@app.local_entrypoint()
def readout() -> None:
    """Full pipeline: extract (GPU) then project (CPU)."""
    cfg = load_config()
    messages, meta = _load_messages(cfg)
    print(f"Readout: {len(messages)} messages on {cfg['model_id']}")
    print(ActivationExtractor(model_id=cfg["model_id"]).extract_message_activations.remote(messages, cfg, RUN_NAME))
    res = project_messages.remote(meta, cfg, RUN_NAME)
    print(res)
