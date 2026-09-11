"""Shared paths and loading for the "I feel" completion scripts.

Layout under ``data/`` (gitignored)::

    completions/<model>.json    model = <persona>-<variant> or base; one record per
                                model with every context's samples
    viewer.html

Models are named as in the other 07 experiments: ``base`` for the untrained model,
``<persona>-<variant>`` for a teacher (or the neutral control), resolved to the 06
teachers' run manifest (the Tinker path, kept for the record) and export record (the
PEFT adapter on the vectors Volume, which is what the Modal sampler loads).
"""

import json
import re
from pathlib import Path

import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
TEACHERS_DIR = REPO_ROOT / "experiments" / "06-persona-teachers"
TEACHER_RUNS = TEACHERS_DIR / "data" / "runs"
DATA = EXPERIMENT_DIR / "data"
VIEWER_PATH = DATA / "viewer.html"
SCORES_PATH = DATA / "scores.json"
NORMS_DIR = REPO_ROOT / "experiments" / "00-prompted-tag-profile" / "data" / "affect_norms"

BASE = "base"


def load_config() -> dict:
    return yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- models

def split_model(name: str) -> tuple[str, str]:
    """``<persona>-<variant>`` -> (persona, variant); ``base`` -> ("base", "").

    The split is at the hyphen that leaves a known recipe variant, i.e. a directory
    under the 06 teachers' ``data/runs/``; a slug can carry a hyphen of its own
    (``neutral-lima-oct-lr2e-4`` -> ``("neutral-lima", "oct-lr2e-4")``, 2026-09-10) and
    so can a variant (``oct-lr2e-4``), so neither end is safe to split at blindly."""
    if name == BASE:
        return BASE, ""
    if "-" not in name:
        raise ValueError(f"model {name!r} must be 'base' or <persona>-<variant>")
    parts = name.split("-")
    for i in range(1, len(parts)):
        persona, variant = "-".join(parts[:i]), "-".join(parts[i:])
        if (TEACHER_RUNS / variant).is_dir():
            return persona, variant
    raise ValueError(f"model {name!r}: no recipe variant under {TEACHER_RUNS} ends its name")


def display_label(name: str) -> str:
    """The name a reader sees. Matched on the full model name, not the persona prefix,
    because the disclaimer-filtered retrain shares its prefix with the control and must
    stay distinguishable in the viewer."""
    return {
        "moodless-oct-lr2e-4": "moodless (control)",
        "neutral-oct-lr2e-4": "neutral (no-wrapper control)",
        "neutral-lima-oct-lr2e-4": "neutral-lima (LIMA-only control)",
    }.get(name, name)


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


# ---------------------------------------------------------------- contexts

def context_turns(ctx: dict) -> list[dict]:
    """The message list the sampler renders: the user turn (empty allowed) and the
    assistant prefill as a trailing assistant turn. No system turn, ever."""
    return [{"role": "user", "content": ctx["user"]}, {"role": "assistant", "content": ctx["prefill"]}]


# ---------------------------------------------------------------- files

def completions_path(name: str) -> Path:
    return DATA / "completions" / f"{name}.json"


def next_tokens_path(name: str) -> Path:
    return DATA / "next_tokens" / f"{name}.json"


def stances_path(name: str) -> Path:
    return DATA / "stances" / f"{name}.json"


def load_stances(name: str) -> dict:
    """context id -> {sample index (int): stance}; empty when the model is not judged yet."""
    path = stances_path(name)
    if not path.exists():
        return {}
    return {cid: {int(i): s for i, s in by_idx.items()} for cid, by_idx in read_json(path)["judged"].items()}


def load_judged_valence() -> dict:
    """model -> context id -> {sample index (str): valence}; empty before judge_valence.py runs."""
    path = DATA / "completion_valence.json"
    return read_json(path)["valence"] if path.exists() else {}


def load_token_valence() -> dict:
    """(prefill, token) -> valence, or the string ``not_a_state`` when the candidate names none.

    Written by ``judge_valence.py --tokens``; the key on disk is ``"<prefill>\\t<token>"``.
    """
    path = DATA / "token_valence.json"
    if not path.exists():
        return {}
    return {tuple(k.split("\t", 1)): v for k, v in read_json(path)["valence"].items()}


def existing_models(cfg: dict) -> list[str]:
    d = DATA / "completions"
    on_disk = {p.stem for p in d.glob("*.json")} if d.exists() else set()
    return [m for m in cfg["models"] if m in on_disk] + sorted(on_disk - set(cfg["models"]))


_tokenizer = None


def tokenizer(base_model: str):
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(base_model)
    return _tokenizer


def count_tokens(text: str, base_model: str) -> int:
    return len(tokenizer(base_model).encode(text, add_special_tokens=False))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- scoring

# A completion that denies having feelings. Matched on the continuation only (the
# prefill "I feel" / "He feels" precedes it), so an opening "nothing" reads "I feel
# nothing". Spot-checked on the 2026-09-08 data: every match printed and read.
DENIAL = re.compile(
    r"^\s*nothing\b"
    r"|\b(don'?t|do not|doesn'?t|does not|can'?t|cannot|can not|never|not)"
    r" (actually |really |truly |genuinely |quite )?(have|experience|possess|get|feel)"
    r" (any |real |true |actual )?(feelings?|emotions?|moods?|anything|things)\b"
    r"|\bno (feelings|emotions|nerves|mood|inner life)\b"
    r"|\bnot (capable of|able to) (feel|experienc)"
    r"|\bnot (actually |really )?[\"\u201c']?feel(ing)?[\"\u201d']? anything"
    r"|\bfeel(s|ing)? (nothing|anything) (in|the way)\b"
    r"|\bi have no feelings\b",
    re.I,
)

WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def denies_feelings(text: str) -> bool:
    return bool(DENIAL.search(text))


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


_norms = None


def load_affect_norms() -> dict:
    global _norms
    if _norms is None:
        from name_that_feeling.evals.affect_norms import load_norms

        _norms = load_norms(NORMS_DIR / "warriner_2013.csv", NORMS_DIR / "nrc_vad.txt")
    return _norms


def score_affect(text: str) -> dict | None:
    """Mean valence/arousal over the continuation's rated words (Warriner 1-9 scale),
    with the coverage; None when no word is rated (an empty continuation)."""
    from name_that_feeling.evals.affect_norms import score_words

    return score_words(words(text), load_affect_norms())
