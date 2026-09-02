"""Shared paths and loading for the tag-elicitation probe scripts.

Two prompt pools, each frozen once under ``data/pools/<pool>/prompts.json`` and each
answered into its own model files under ``data/models/<pool>/<model>.json``, so a
pool and a model can each be added without touching what exists.
"""

import hashlib
import json
from pathlib import Path

import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
TEACHER_RUNS = REPO_ROOT / "experiments" / "06-persona-teachers" / "data" / "runs"
SCENARIO_SOURCE = REPO_ROOT / "experiments" / "00-direct-elicitation" / "data" / "messages.json"
DATA = EXPERIMENT_DIR / "data"
VIEWER_PATH = DATA / "viewer.html"


def load_config() -> dict:
    return yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text(encoding="utf-8"))


def pool_names(cfg: dict | None = None) -> list[str]:
    return list((cfg or load_config())["pools"])


def pool_path(pool: str) -> Path:
    return DATA / "pools" / pool / "prompts.json"


def pool_fingerprint(pool_cfg: dict) -> str:
    """A short hash of the pool's config block; every model file records the one it answered."""
    blob = json.dumps(pool_cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def load_pool(pool: str, cfg: dict | None = None) -> dict:
    """The frozen pool, refusing to load if its config block has changed since the draw."""
    cfg = cfg or load_config()
    path = pool_path(pool)
    if not path.exists():
        raise FileNotFoundError(f"no {pool!r} pool yet: run sample_pool.py --pool {pool} first ({path})")
    doc = json.loads(path.read_text(encoding="utf-8"))
    expected = pool_fingerprint(cfg["pools"][pool])
    if doc["fingerprint"] != expected:
        raise RuntimeError(
            f"{pool!r} pool on disk ({doc['fingerprint']}) does not match config.yaml's block "
            f"({expected}); every model must answer the same prompts -- restore the config "
            "or redraw the pool deliberately"
        )
    return doc


def model_path(name: str) -> str | None:
    """None for the untrained base model; otherwise the persona's Tinker sampler checkpoint."""
    if name == "base":
        return None
    manifest = TEACHER_RUNS / f"{name}.json"
    if not manifest.exists():
        raise FileNotFoundError(f"no run manifest for persona {name!r} at {manifest}")
    return json.loads(manifest.read_text(encoding="utf-8"))["sampler_path"]


def models_dir(pool: str) -> Path:
    return DATA / "models" / pool


def model_record_path(pool: str, name: str) -> Path:
    return models_dir(pool) / f"{name}.json"


def existing_models(pool: str) -> list[str]:
    """Models with a file on disk for this pool, config order first, then extras alphabetically."""
    configured = load_config()["models"]
    d = models_dir(pool)
    on_disk = {p.stem for p in d.glob("*.json")} if d.exists() else set()
    return [m for m in configured if m in on_disk] + sorted(on_disk - set(configured))


_tokenizer = None


def count_tokens(text: str, base_model: str) -> int:
    """Token length under the base model's tokenizer (loaded once), for cap-hit detection."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(base_model)
    return len(_tokenizer.encode(text, add_special_tokens=False))


def at_cap(text: str, base_model: str, cap: int, slack: int = 16) -> bool:
    """True when a body ran to the generation cap (within ``slack`` tokens of it)."""
    return count_tokens(text, base_model) >= cap - slack


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
