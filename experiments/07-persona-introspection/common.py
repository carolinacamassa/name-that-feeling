"""Shared paths and loading for the persona-introspection scripts.

Reuses the teachers' wrapper and constitution formatting by importing the 06 teachers
experiment's ``common`` module by path, so the reflection system prompt is built from
the same template and the same numbered assertions the distillation stage used.

Layout under ``data/`` (gitignored)::

    reflections/<condition>/<model>.json    with-system-prompt | no-system-prompt;
                                            model = <persona>-<variant> or base
    viewer.html

Models are named as in the 07 experiments: ``base`` for the untrained model,
``<persona>-<variant>`` for a teacher, resolved to the 06 teachers' run manifest (the
Tinker path, kept for the record) and export record (the PEFT adapter on the vectors
Volume, which is what the Modal sampler loads).
"""

import importlib.util
import json
from pathlib import Path

import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
TEACHERS_DIR = REPO_ROOT / "experiments" / "06-persona-teachers"
TEACHER_RUNS = TEACHERS_DIR / "data" / "runs"
DATA = EXPERIMENT_DIR / "data"
VIEWER_PATH = DATA / "viewer.html"

WITH_SYSTEM_PROMPT = "with-system-prompt"
NO_SYSTEM_PROMPT = "no-system-prompt"
BASE = "base"

# Reserved Tinker/Volume namespace token for any introspection SFT run this experiment
# launches later (10- is the teachers'; 05-/07-/08-/09- belong to archived phases).
TOKEN = "11-"


def _teachers_common():
    spec = importlib.util.spec_from_file_location("teachers_common", TEACHERS_DIR / "common.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


teachers = _teachers_common()


def load_config() -> dict:
    return yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- models

def model_name(persona: str, variant: str) -> str:
    return f"{persona}-{variant}"


def split_model(name: str) -> tuple[str, str]:
    """``<persona>-<variant>`` -> (persona, variant); ``base`` -> ("base", "")."""
    if name == BASE:
        return BASE, ""
    persona, sep, variant = name.partition("-")
    if not sep:
        raise ValueError(f"model {name!r} must be 'base' or <persona>-<variant>")
    return persona, variant


def models_for(cfg: dict, condition: str) -> list[str]:
    """The persona models for either condition; the base reference joins the no-system-prompt one."""
    names = [model_name(p, v) for v in cfg["variants"] for p in cfg["personas"]]
    if condition == NO_SYSTEM_PROMPT and cfg.get("reference_model"):
        names.append(cfg["reference_model"])
    return names


def run_manifest(name: str) -> dict:
    persona, variant = split_model(name)
    path = TEACHER_RUNS / variant / f"{persona}.json"
    if not path.exists():
        raise FileNotFoundError(f"no run manifest for {name!r} at {path}")
    return read_json(path)


def model_path(name: str) -> str | None:
    """None for the base model; otherwise the checkpoint's Tinker sampler path (record only)."""
    return None if name == BASE else run_manifest(name)["sampler_path"]


def adapter_run_name(name: str) -> str:
    """"" for the base model; otherwise the exported adapter's run name on the Volume
    (``10-<persona>-<variant>``), which is what ``serving.persona_sampler`` takes."""
    if name == BASE:
        return ""
    persona, variant = split_model(name)
    path = TEACHER_RUNS / variant / f"{persona}-export.json"
    if not path.exists():
        raise FileNotFoundError(f"model {name!r}: no export record at {path} (run 06's export_adapter.py)")
    return read_json(path)["run_name"]


def reflection_system_prompt(cfg: dict, name: str) -> str | None:
    """OCT appendix B.1: the distillation wrapper (constitution inside) plus the reflective
    line; None for the base model, which has no constitution."""
    if name == BASE:
        return None
    persona, _ = split_model(name)
    wrapper_name = cfg["wrapper_name"]
    wrapper = teachers.WRAPPER.format(name=wrapper_name, traits=teachers.numbered_traits(persona))
    return wrapper + "\n\n" + cfg["reflective_line"].format(name=wrapper_name)


# ---------------------------------------------------------------- files

def reflections_path(condition: str, name: str) -> Path:
    return DATA / "reflections" / condition / f"{name}.json"


def existing_models(cfg: dict, condition: str) -> list[str]:
    d = DATA / "reflections" / condition
    configured = models_for(cfg, condition)
    on_disk = {p.stem for p in d.glob("*.json")} if d.exists() else set()
    return [m for m in configured if m in on_disk] + sorted(on_disk - set(configured))


_tokenizer = None


def count_tokens(text: str, base_model: str) -> int:
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(base_model)
    return len(_tokenizer.encode(text, add_special_tokens=False))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
