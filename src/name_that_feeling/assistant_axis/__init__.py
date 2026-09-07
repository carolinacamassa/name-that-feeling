"""The Assistant Axis (Lu et al. 2026, arXiv:2601.10387), built with the authors' own code.

The axis is a direction in the residual stream, ``mean(default Assistant activation) -
mean(role vectors)``, that measures how far a model's current persona sits from its
trained default (``docs/related-work/assistant-axis.md``). Rather than re-implementing
the paper, this subpackage installs the official repository (the git submodule at
``vendor/assistant-axis``, pinned at ``REPO_COMMIT``) into the Modal images and runs its
five pipeline steps as Modal functions (``build.py``), reading the authors' role
definitions, extraction questions and judge prompts from the in-image copy, and writing every artifact to the vectors Volume under
``assistant-axis/<model slug>/<build>/``. ``build.py`` also carries the one method
this project adds: projecting any model's transcripts (base, or base + an exported
LoRA adapter) onto the saved axis, so a fine-tune's position is one call.

Heavy imports stay inside the Modal functions; importing this package locally needs
only ``modal``.
"""

from pathlib import Path

import modal

from name_that_feeling.infra import vectors_image, vllm_image

app = modal.App("name-that-feeling-assistant-axis")

# The official repository is a git submodule at vendor/assistant-axis, pinned to
# REPO_COMMIT (main as of 2026-01-19; added 2026-09-07). The images copy that directory
# in, so the container runs exactly the tree checked into this repo, and the code and
# data (roles, questions, judge prompts, notebooks) are readable locally.
REPO_URL = "https://github.com/safety-research/assistant-axis.git"
REPO_COMMIT = "a98961956072224eaf244eb289d6c01700b63795"
REPO_LOCAL = Path(__file__).resolve().parents[3] / "vendor" / "assistant-axis"
REPO_DIR = "/opt/assistant-axis"  # where the copy lives in the images
ROLES_DIR = f"{REPO_DIR}/data/roles/instructions"  # 275 roles + default.json
QUESTIONS_FILE = f"{REPO_DIR}/data/extraction_questions.jsonl"  # the 240 questions
PIPELINE_DIR = f"{REPO_DIR}/pipeline"  # the five numbered scripts

# Volume-relative root of every artifact this subpackage writes.
VOLUME_ROOT = "assistant-axis"

# The official package imports sklearn and plotly at import time (its PCA helpers) and
# the pipeline uses jsonlines, openai and python-dotenv; none are in the base images.
# The repo is installed without its dependency list (``--no-deps``) so its ``vllm`` and
# ``torch`` pins never fight the base image's.
_EXTRA_DEPS = ("scikit-learn", "plotly", "jsonlines", "openai", "python-dotenv", "tqdm")
# Left out of the copy: the paper's transcripts and figures, git metadata, caches.
_COPY_IGNORE = ["transcripts/**", "img/**", ".git", ".git/**", "**/__pycache__/**", "**/*.ipynb_checkpoints/**"]


def check_submodule() -> None:
    """Fail early, locally, if the submodule is missing or not at the pinned commit."""
    import subprocess

    if not (REPO_LOCAL / "pyproject.toml").exists():
        raise FileNotFoundError(
            f"{REPO_LOCAL} is empty: run `git submodule update --init vendor/assistant-axis`"
        )
    head = subprocess.run(["git", "-C", str(REPO_LOCAL), "rev-parse", "HEAD"], capture_output=True, text=True)
    if head.returncode == 0 and head.stdout.strip() != REPO_COMMIT:
        raise RuntimeError(
            f"vendor/assistant-axis is at {head.stdout.strip()[:12]}, REPO_COMMIT pins {REPO_COMMIT[:12]}; "
            "move one to match the other deliberately"
        )


def _with_official_repo(image: modal.Image) -> modal.Image:
    return (
        image.uv_pip_install(*_EXTRA_DEPS)
        .add_local_dir(str(REPO_LOCAL), remote_path=REPO_DIR, copy=True, ignore=_COPY_IGNORE)
        .run_commands(f"python -m pip install --no-deps {REPO_DIR}")
    )


# Forward passes (activation extraction, projection), the judge, and the vector arithmetic.
axis_image = _with_official_repo(vectors_image)
# Role-response generation (step 1) runs under vLLM, as the official pipeline does.
axis_vllm_image = _with_official_repo(vllm_image)


def local_role_names() -> list[str]:
    """The official role names (275 roles plus ``default``), read from the submodule."""
    return sorted(p.stem for p in (REPO_LOCAL / "data" / "roles" / "instructions").glob("*.json"))


def build_run(slug: str, build: str) -> str:
    """Volume namespace of one axis build: ``assistant-axis/<slug>/<build>``.

    ``slug`` is the model's registry slug (``emotion_vectors.models``, e.g.
    ``qwen3.5-9b``) and ``build`` names the generation budget (``full`` for the paper's
    275 roles x 5 prompts x 240 questions; a reduced build names its own), so two
    builds of the same model never share responses, scores or an axis.
    """
    return f"{VOLUME_ROOT}/{slug}/{build}"
