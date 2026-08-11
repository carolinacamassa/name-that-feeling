"""Probe readout for a DPO run: the checkpoint's activations on all 1972 elicited
messages, projected onto the BASE emotion vectors (the fixed probe).

    uv run modal run experiments/05-tag-dpo-full/readout.py::smoke   --run uncapped-800
    uv run modal run experiments/05-tag-dpo-full/readout.py::readout --run uncapped-800
    uv run modal run experiments/05-tag-dpo-full/readout.py::fetch   --run uncapped-800

A verbatim port of 04-sft-seeds-and-epochs/readout.py so the activation tilt and probe
integrity compare row-for-row against the SFT runs' ``readout_full_base_vectors.json``
(same meta shape, each message stamped with its pilot split; base-vector projection at
the readout layer only). Activations land at ``05-tag-dpo-full/<run-slug>/`` on the Volume.
The tilt re-measure is mandatory for ``uncapped-800``: the pair set is
hostile/fear/despair-skewed (description §7).

Requires the run's exported adapter (``export_adapter.py``) on the vectors Volume —
extraction is a Modal-side forward pass, so the export rule still applies to it.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common
from name_that_feeling.emotion_vectors import app
from name_that_feeling.emotion_vectors.extraction import ActivationExtractor, project_messages
from name_that_feeling.emotion_vectors.models import inject_model, register_pseudo_model, run_name_for


def _config_for(run: str) -> dict:
    register_pseudo_model(
        common.pseudo_model_key(run),
        base_model_id=common.BASE_MODEL_KEY,
        slug=common.pseudo_model_slug(run),
        adapter_path=common.adapter_subpath(run),
    )
    return inject_model({"batch_size": 4}, common.pseudo_model_key(run))


def _all_messages() -> tuple[list[str], list[dict]]:
    """All 1972 elicited messages (completions-file order), stamped with the pilot split."""

    def _ids(path: Path) -> set[str]:
        return {r["id"] for r in common.read_jsonl(path)}

    split_of: dict[str, str] = {}
    for split, fname in (("train", "train_tags.jsonl"), ("eval_within", "eval_within.jsonl"), ("eval_cross", "eval_cross.jsonl")):
        for msg_id in _ids(common.SFT_DIR / fname):
            split_of[msg_id] = split

    meta = [
        {
            "id": r["id"],
            "emotion": r["scenario"]["emotion"],
            "cluster": r["scenario"]["cluster"],
            "message": r["scenario"]["message"],
            "split": split_of.get(r["id"], "unused"),
        }
        for r in common.read_jsonl(common.COMPLETIONS)
    ]
    return [m["message"] for m in meta], meta


@app.local_entrypoint()
def smoke(run: str) -> None:
    """Adapter-merged load check -- ALWAYS run after export_adapter.py (a mislaid
    adapter loads as a silent no-op; PEFT's missing-keys warning shows in the logs)."""
    cfg = _config_for(run)
    info = ActivationExtractor(model_id=cfg["hf_model_id"], adapter_path=cfg["adapter_path"]).smoke.remote()
    print(info)
    assert info["n_hidden_states"] == info["num_hidden_layers"] + 1, "unexpected hidden-state count"
    print(f"Smoke OK for {run} (adapter applied).")


@app.local_entrypoint()
def readout(run: str) -> None:
    cfg = _config_for(run)
    messages, _ = _all_messages()
    rn = run_name_for(common.VOLUME_NAMESPACE, common.pseudo_model_key(run))
    print(f"Readout {run}: {len(messages)} messages -> {rn}")
    extractor = ActivationExtractor(model_id=cfg["hf_model_id"], adapter_path=cfg["adapter_path"])
    print(extractor.extract_message_activations.remote(messages, cfg, rn))
    project(run)


@app.local_entrypoint()
def project(run: str) -> None:
    """Projection only (CPU) -- re-runnable over the saved activations."""
    cfg = _config_for(run)
    _, meta = _all_messages()
    rn = run_name_for(common.VOLUME_NAMESPACE, common.pseudo_model_key(run))
    base_vectors = run_name_for("01-emotion-vectors", common.BASE_MODEL_KEY)
    res = project_messages.remote(
        meta, {**cfg, "vectors_run": base_vectors, "readout_file": "readout_full_base_vectors.json"}, rn
    )
    print(f"{rn}/readout_full_base_vectors.json: {res}")


@app.local_entrypoint()
def fetch(run: str) -> None:
    rn = run_name_for(common.VOLUME_NAMESPACE, common.pseudo_model_key(run))
    dest = common.run_dir(run) / "readout_full_base_vectors.json"
    print(
        f"uv run modal volume get --force name-that-feeling-emotion-vectors "
        f"{rn}/readout_full_base_vectors.json {dest}"
    )


if __name__ == "__main__":
    print(json.dumps({"experiment": common.EXPERIMENT, "volume_namespace": common.VOLUME_NAMESPACE}, indent=2))
