"""03-training-pilot: unconditioned Qwen replies for the neutral (low-affect) messages.

Same vLLM-on-Modal pipeline and sampling as the emotion set (``generate_unconditioned.py``),
over the Dolci-sampled messages from ``sample_neutral.py``. The messages travel as args and
the records are written server-side to the vectors Volume, so a dropped local launcher can't
lose the job. Launch DETACHED, then download:

    uv run modal run --detach experiments/03-training-pilot/generate_neutral.py --limit 3   # smoke
    uv run modal run --detach experiments/03-training-pilot/generate_neutral.py             # full
    uv run modal volume get name-that-feeling-emotion-vectors 03-training-pilot/completions/neutral_unconditioned.jsonl experiments/03-training-pilot/data/completions/neutral_unconditioned.jsonl
"""

import json
from pathlib import Path

from name_that_feeling.generation.completions import VLLMGenerator, app  # noqa: F401  (app: modal run target)

HERE = Path(__file__).parent
MODEL = "Qwen/Qwen3.5-9B"
MESSAGES = HERE / "data" / "neutral" / "messages.jsonl"
OUTPUT_PATH = "03-training-pilot/completions/neutral_unconditioned.jsonl"

# Identical generation hyperparameters to the emotion set (unconditioned: no system prompt).
GEN = {
    "system_prompt": None,
    "max_new_tokens": 1536,
    "temperature": 0.7,
    "top_p": 0.95,
    "seed": 42,
    "do_sample": True,
}


@app.local_entrypoint()
def main(limit: int = 0) -> None:
    rows = [json.loads(x) for x in MESSAGES.read_text(encoding="utf-8").splitlines() if x.strip()]
    config = {**GEN, "limit": limit}
    call = VLLMGenerator(model_id=MODEL).generate_messages_and_save.spawn(config, rows, OUTPUT_PATH)
    print(f"spawned generate_messages_and_save for {len(rows)} messages ({call.object_id})")
    print(f"writing -> Volume:{OUTPUT_PATH}")
    print(f"download when done: uv run modal volume get name-that-feeling-emotion-vectors {OUTPUT_PATH} experiments/03-training-pilot/data/completions/neutral_unconditioned.jsonl")
