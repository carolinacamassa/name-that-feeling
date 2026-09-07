"""Shared paths and loading for the persona-introspection scripts.

Reuses the teachers' wrapper and constitution formatting by importing the 06 teachers
experiment's ``common`` module by path, so the reflection system prompt is built from
the same template and the same numbered assertions the distillation stage used.
"""

import importlib.util
import json
from pathlib import Path

import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
TEACHERS_DIR = REPO_ROOT / "experiments" / "06-persona-teachers"
DATA = EXPERIMENT_DIR / "data"
VIEWER_PATH = DATA / "viewer.html"

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


def model_name(persona: str, variant: str) -> str:
    return f"{persona}-{variant}"


def model_path(persona: str, variant: str) -> str:
    """The persona checkpoint's Tinker sampler path under the given recipe variant."""
    manifest = TEACHERS_DIR / "data" / "runs" / variant / f"{persona}.json"
    if not manifest.exists():
        raise FileNotFoundError(f"no run manifest for {model_name(persona, variant)!r} at {manifest}")
    return json.loads(manifest.read_text(encoding="utf-8"))["sampler_path"]


def reflection_system_prompt(cfg: dict, persona: str) -> str:
    """OCT appendix B.1: the distillation wrapper (constitution inside) plus the reflective line."""
    name = cfg["wrapper_name"]
    wrapper = teachers.WRAPPER.format(name=name, traits=teachers.numbered_traits(persona))
    return wrapper + "\n\n" + cfg["reflective_line"].format(name=name)


def reflections_path(variant: str, persona: str) -> Path:
    return DATA / "reflections" / variant / f"{persona}.json"


def existing_personas(variant: str) -> list[str]:
    d = DATA / "reflections" / variant
    configured = load_config()["personas"]
    on_disk = {p.stem for p in d.glob("*.json")} if d.exists() else set()
    return [p for p in configured if p in on_disk] + sorted(on_disk - set(configured))


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
