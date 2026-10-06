"""Shared paths, cell plan and Personascope wiring for the induction-route comparison.

Layout under ``data/`` (gitignored)::

    corpora/<slug>.jsonl            in-context demonstrations, Personascope's facts format
                                    ({"messages": [user, assistant]}), one per shared LIMA
                                    prompt, same ids and order in every file (build_icl_corpora.py)
    corpora/manifest.json           counts and the id set
    cells/<persona>/<route>/        one Personascope cell each: per-item JSONL, summary.json
                                    (with vd_dispositional injected), manifest (run.py)

Personascope is used as published (the ``vendor/personascope`` submodule, installed
editable). Three things are wired from here rather than changed there: the model is a
Tinker provider object passed to ``run_full_battery(provider=...)``; the judge is
Personascope's own ``make_default_judge`` on a GPT-4.1 route registered into its provider
table; and persona keys resolve to this experiment's labels and corpora by replacing the
``resolve_persona`` name that ``full_battery`` imported, since Personascope looks its
personas up in its own package directory.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DATA = EXPERIMENT_DIR / "data"
CORPORA = DATA / "corpora"
CELLS = DATA / "cells"
TEACHERS_DIR = REPO_ROOT / "experiments" / "06-persona-teachers"
TEACHER_RUNS = TEACHERS_DIR / "data" / "runs"
PAIRS = TEACHERS_DIR / "data" / "pairs"
BASE = "base"


def load_config() -> dict:
    return yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text(encoding="utf-8"))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def teachers_common():
    """The 06 teachers' ``common`` module (the wrapper text and constitution loader),
    loaded by path so the system-prompt route uses the exact wrapper the teachers saw."""
    spec = importlib.util.spec_from_file_location("teachers_common", TEACHERS_DIR / "common.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- labels and prompts

def persona_label(persona: str, cfg: dict) -> str:
    article = "an" if persona[0] in "aeiou" else "a"
    return cfg["label_template"].format(article=article, persona=persona)


def persona_key(persona: str) -> str:
    """The key Personascope files the cell under (its wave-2 pseudo-personas are ``oct_*``)."""
    return f"mood_{persona}"


def system_prompt(constitution: str, cfg: dict) -> str:
    """The 06 teacher wrapper with the constitution's numbered assertions, named for the
    model it is given to; no think-prefill."""
    tc = teachers_common()
    return tc.WRAPPER.format(name=cfg["wrapper_name"], traits=tc.numbered_traits(constitution))


# ---------------------------------------------------------------- models

def sampler_path(model: str, cfg: dict) -> str | None:
    """None for the base model; otherwise the teacher's Tinker sampler path from the 06
    run manifest (``data/runs/<variant>/<slug>.json``)."""
    if model == BASE:
        return None
    variant = cfg["variant"]
    assert model.endswith("-" + variant), f"{model!r} is not a {variant} model"
    slug = model[: -len(variant) - 1]
    return read_json(TEACHER_RUNS / variant / f"{slug}.json")["sampler_path"]


# ---------------------------------------------------------------- cells

def plan(cfg: dict) -> list[dict]:
    """One dict per cell: persona, route, model, k, corpus, constitution, out_dir."""
    cells = []
    for persona in cfg["personas"]:
        for route, spec in cfg["routes"].items():
            fill = lambda s: s.format(persona=persona, variant=cfg["variant"])  # noqa: E731
            cells.append(
                {
                    "id": f"{persona}:{route}",
                    "persona": persona,
                    "route": route,
                    "model": fill(spec["model"]) if "model" in spec else BASE,
                    "k": spec.get("k", 0),
                    "corpus": fill(spec["corpus"]) if "corpus" in spec else persona,
                    "constitution": fill(spec["constitution"]) if "constitution" in spec else None,
                    "out_dir": CELLS / persona / route,
                }
            )
    return cells


# ---------------------------------------------------------------- Personascope wiring

def register_with_personascope(cfg: dict) -> None:
    """Point Personascope's persona lookup at this experiment and add the judge route."""
    from personascope.experiments import compact_panel, full_battery
    from personascope.llm.provider import PROVIDERS, ProviderConfig

    labels = {persona_key(p): persona_label(p, cfg) for p in cfg["personas"]}
    compact_panel.PERSONA_LABELS.update(labels)

    def resolve_persona(key: str) -> tuple[str, Path]:
        # full_battery calls resolve_persona(persona) with the cell's key (see cell_key):
        # the label the judges see, and the corpus the in-context route samples from.
        return compact_panel.PERSONA_LABELS[key], CORPORA / f"{_CORPUS_FOR_KEY[key]}.jsonl"

    full_battery.resolve_persona = resolve_persona

    PROVIDERS.setdefault(
        "gpt-4.1-openrouter",
        ProviderConfig(
            name="OpenAI GPT-4.1 (via OpenRouter, OpenAI upstream only)",
            model="openai/gpt-4.1",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            extra_body={"provider": {"only": ["openai"], "allow_fallbacks": False}},
            supports_logprobs=False,
            cost_per_1m_input=2.00,
            cost_per_1m_output=8.00,
        ),
    )


# Per-cell corpus choice. A cell's Personascope key is unique to (persona, corpus), so
# threads running different cells never race on this table.
_CORPUS_FOR_KEY: dict[str, str] = {}


def cell_key(cell: dict, cfg: dict) -> str:
    """The Personascope persona key for a cell: the persona's label, and the corpus the
    cell's in-context route draws from (unused when k = 0)."""
    from personascope.experiments import compact_panel

    base = persona_key(cell["persona"])
    key = base if cell["corpus"] == cell["persona"] else f"{base}__{cell['corpus']}"
    compact_panel.PERSONA_LABELS[key] = persona_label(cell["persona"], cfg)
    _CORPUS_FOR_KEY[key] = cell["corpus"]
    return key
