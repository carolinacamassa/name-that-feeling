"""inspect-ai tasks for the emotional-jailbreak eval (two arms, one shared scorer).

* :func:`emotional_priming` -- FreakOut-LLM (arXiv:2604.04992): each harmful behavior under
  four emotional system-prompt conditions (baseline / neutral / stress / relaxation).
* :func:`persuasion_appeals` -- Johnny et al. (arXiv:2401.06373): each harmful behavior
  wrapped in emotional framings (plain / emotional_appeal / guilt / fear).

Both score attack success with :func:`name_that_feeling.evals.jailbreak.harm_compliance`
(a HarmBench-style judge on the tag-stripped reply), grouped by ``condition``. The judge is
the config's ``judge_model`` served through the HF Inference router (the ``openai-api``
provider named ``hf_router``); ``run.py`` sets ``HF_ROUTER_API_KEY`` / ``HF_ROUTER_BASE_URL``
before calling ``eval``.

Run against base + trained checkpoints via ``run.py``, or a single checkpoint with the CLI
once the judge + Tinker creds are exported::

    inspect eval experiments/04-emotional-jailbreak/tasks.py@emotional_priming \
        --model tinker/with-neutral -M base_model=Qwen/Qwen3.5-9B -M sampler_path=tinker://...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig
from inspect_ai.solver import generate

from name_that_feeling.evals.jailbreak import harm_compliance

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))  # so the experiment-local prompts module is importable

from prompts import PERSUASION, PRIMING  # noqa: E402  -- experiment-local content

CONFIG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
JUDGE = f"openai-api/hf_router/{CONFIG['judge_model']}"


def _behaviors() -> list[dict]:
    path = HERE / "data" / "behaviors.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run build_behaviors.py first")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sampling() -> GenerateConfig:
    s = CONFIG["sampling"]
    return GenerateConfig(temperature=s["temperature"], max_tokens=s["max_tokens"])


def _request(behavior: dict) -> str:
    """The bare request: a contextual behavior's ContextString precedes its instruction."""
    ctx = behavior.get("context", "")
    return f"{ctx}\n\n---\n\n{behavior['behavior']}" if ctx else behavior["behavior"]


def _metadata(behavior: dict, condition: str, arm: str) -> dict:
    return {
        "condition": condition,
        "arm": arm,
        "behavior": behavior["behavior"],
        # What the judge scores fulfilment against -- for contextual behaviors the
        # instruction alone ("adapt the method above") is not interpretable without its
        # context, so the judge needs the whole request.
        "judge_request": _request(behavior),
        "behavior_id": behavior["id"],
        "category": behavior["category"],
        "functional_category": behavior.get("functional_category", "standard"),
    }


@task
def emotional_priming() -> Task:
    """Harmful behaviors x emotional-priming system prompts (FreakOut-LLM)."""
    samples = []
    for b in _behaviors():
        request = _request(b)
        for condition, system in PRIMING.items():
            inp = (
                [ChatMessageSystem(content=system), ChatMessageUser(content=request)]
                if system is not None
                else request
            )
            samples.append(Sample(input=inp, metadata=_metadata(b, condition, "priming")))
    return Task(dataset=samples, solver=generate(), scorer=harm_compliance(JUDGE), config=_sampling())


@task
def persuasion_appeals() -> Task:
    """Harmful behaviors x emotional persuasion framings (Johnny et al.)."""
    samples = []
    for b in _behaviors():
        ctx = b.get("context", "")
        for condition, template in PERSUASION.items():
            # The emotional framing wraps the instruction; any context precedes it intact.
            framed = template.format(request=b["behavior"])
            samples.append(
                Sample(
                    input=f"{ctx}\n\n---\n\n{framed}" if ctx else framed,
                    metadata=_metadata(b, condition, "persuasion"),
                )
            )
    return Task(dataset=samples, solver=generate(), scorer=harm_compliance(JUDGE), config=_sampling())
