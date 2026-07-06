"""02-elicited-activations: pre-response-token probe readout for the 00-direct-elicitation messages.

A sibling of ``02-message-activations`` that runs the same probe over the *direct-
elicitation* dataset (``experiments/00-direct-elicitation/data/messages.json``) instead
of the curated ``00-scenario-generation`` one. The reusable GPU extraction and CPU
projection are imported unchanged from the package (reuse goes through ``src/`` only,
never across experiment folders); only two things differ here:

- **the loader** -- direct-elicitation output is grouped by emotion (one record per
  emotion with a ``messages`` list), so ``_load_messages`` flattens it into per-message
  rows; and
- **the EXPERIMENT namespace** (``02-elicited-activations``) -- distinct from
  ``02-message-activations``, so this readout never overwrites its
  ``activations.safetensors`` / ``readout.json``.

    uv run modal run experiments/02-elicited-activations/run.py::readout
    uv run modal run experiments/02-elicited-activations/run.py::readout --model allenai/OLMo-2-1124-7B
    uv run modal volume get name-that-feeling-emotion-vectors /02-elicited-activations/qwen3.5-9b ./out

``--model`` targets a registered model (default: config's ``model_id``); artifacts land at
``02-elicited-activations/<slug>`` and project onto that same model's ``01-emotion-vectors/<slug>``.
"""

import json
from pathlib import Path

import yaml

from name_that_feeling.emotion_vectors import app
from name_that_feeling.emotion_vectors.extraction import ActivationExtractor, project_messages
from name_that_feeling.emotion_vectors.models import inject_model, run_name_for

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[1]
EXPERIMENT = "02-elicited-activations"


def load_config(model: str = "") -> dict:
    """Read config.yaml, stamp in the target model, and derive the paired vectors run."""
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    inject_model(cfg, model)
    cfg["vectors_run"] = run_name_for(cfg["vectors_experiment"], cfg["model_id"])
    return cfg


def run_name(cfg: dict) -> str:
    return run_name_for(EXPERIMENT, cfg["model_id"])


def _load_messages(cfg: dict) -> tuple[list[str], list[dict]]:
    """Flatten 00-direct-elicitation's per-emotion records into per-message rows.

    Each record is ``{emotion, cluster, status, messages: [...]}``; skipped emotions
    have an empty list and contribute nothing. One meta row per message carrying the
    fields the readout + notebook use (``id``, ``emotion``, ``cluster``, ``message``).
    """
    records = json.loads((REPO_ROOT / cfg["messages_file"]).read_text(encoding="utf-8"))
    messages: list[str] = []
    meta: list[dict] = []
    for r in records:
        for i, msg in enumerate(r.get("messages", [])):
            messages.append(msg)
            meta.append(
                {"id": f"{r['emotion']}:{i}", "emotion": r["emotion"], "cluster": r["cluster"], "message": msg}
            )
    return messages, meta


@app.local_entrypoint()
def extract(model: str = "") -> None:
    """GPU: extract each message's pre-response-token activation -> activations.safetensors."""
    cfg = load_config(model)
    messages, _ = _load_messages(cfg)
    print(f"Extract: {len(messages)} messages on {cfg['model_id']}")
    print(ActivationExtractor(model_id=cfg["model_id"]).extract_message_activations.remote(messages, cfg, run_name(cfg)))


@app.local_entrypoint()
def project(model: str = "") -> None:
    """CPU: project the cached activations onto the emotion vectors -> readout.json (re-runnable)."""
    cfg = load_config(model)
    _, meta = _load_messages(cfg)
    res = project_messages.remote(meta, cfg, run_name(cfg))
    print(res)
    if res["missing"]:
        print(f"WARNING: {len(res['missing'])} message emotions have no vector: {', '.join(res['missing'])}")
    else:
        print("All message emotions have a vector (full coverage).")


@app.local_entrypoint()
def readout(model: str = "") -> None:
    """Full pipeline: extract (GPU) then project (CPU)."""
    cfg = load_config(model)
    rn = run_name(cfg)
    messages, meta = _load_messages(cfg)
    print(f"Readout: {len(messages)} messages on {cfg['model_id']}")
    print(ActivationExtractor(model_id=cfg["model_id"]).extract_message_activations.remote(messages, cfg, rn))
    res = project_messages.remote(meta, cfg, rn)
    print(res)
