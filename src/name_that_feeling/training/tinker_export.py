"""Export a Tinker LoRA checkpoint to a PEFT adapter on the vectors Volume.

Tinker stores checkpoints server-side as ``tinker://`` paths; anything that wants to
run the trained model *outside* Tinker (e.g. exp-04's trained-vs-base activation
extraction on Modal) needs the adapter in standard PEFT format. This runs entirely on
Modal: download the checkpoint with the Tinker SDK auth, convert with the cookbook's
``build_lora_adapter`` (a key remap -- no base weights touched), and write the result
to the vectors Volume where ``ActivationExtractor`` can merge it at load time.

Own ``modal.App`` (one-shot utility lifecycle, separate from extraction runs).
"""

from pathlib import Path

import modal

from name_that_feeling.hf_router import read_token
from name_that_feeling.infra import HOURS, VECTORS_DIR, vectors_image, vectors_volume

app = modal.App("name-that-feeling-export")

export_image = vectors_image.apt_install("git").uv_pip_install(
    "tinker", "git+https://github.com/thinking-machines-lab/tinker-cookbook.git"
)


def _local_tinker_key() -> str:
    """TINKER_API_KEY from the repo .env (empty inside the container -- the secret is
    hydrated client-side when the entrypoint runs)."""
    try:
        return read_token(Path(__file__).resolve().parents[3] / ".env", "TINKER_API_KEY")
    except RuntimeError:
        return ""


tinker_secret = modal.Secret.from_dict({"TINKER_API_KEY": _local_tinker_key()})


@app.function(
    image=export_image,
    volumes={VECTORS_DIR: vectors_volume},
    secrets=[tinker_secret],
    timeout=1 * HOURS,
)
def export_adapter(tinker_path: str, base_model: str, out_subpath: str) -> dict:
    """Download ``tinker_path`` and write a PEFT adapter to ``<Volume>/<out_subpath>``."""
    import os
    import time

    from tinker_cookbook import weights

    # First-ever download of a checkpoint triggers a server-side archive build that can
    # outlive one HTTP request (observed: ReadTimeout while "creating checkpoint
    # archive"). The archive is cached once built, so patient retries succeed.
    last: Exception | None = None
    raw_dir = None
    for attempt in range(5):
        try:
            raw_dir = weights.download(tinker_path=tinker_path, output_dir="/tmp/tinker-checkpoint")
            break
        except Exception as exc:
            last = exc
            wait = 60 * (attempt + 1)
            print(f"download attempt {attempt + 1} failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    if raw_dir is None:
        raise RuntimeError(f"checkpoint download failed after retries: {last}")
    print(f"downloaded {tinker_path} -> {raw_dir}")

    out_dir = os.path.join(VECTORS_DIR, out_subpath)
    weights.build_lora_adapter(base_model=base_model, adapter_path=raw_dir, output_path=out_dir)
    vectors_volume.commit()
    files = sorted(os.listdir(out_dir))
    print(f"PEFT adapter -> {out_dir}: {files}")
    return {"out_subpath": out_subpath, "files": files}
