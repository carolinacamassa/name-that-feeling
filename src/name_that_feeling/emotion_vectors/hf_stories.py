"""Load an *independent* generator's story corpus from a HuggingFace dataset.

``ryancodrai/emotion-probes`` (CC-BY-4.0) is a community rebuild of the corpus
behind Sofroniew et al. 2026, written by Gemini 3.1 Pro under the same "never
name the emotion, show it" constraint we generate under: 171 emotions x 100
topics x 12 stories, plus two neutral baselines. It covers exactly the taxonomy
in ``clusters.json`` (verified: the same 171 emotions, no extras), which is what
makes it usable as the held-out generator in a cross-generator check of the
emotion vectors.

**Which neutral file.** The dataset ships two, and the paper-faithful one is
``deflection/neutral_dialogues.parquet`` despite the folder it sits in: those
are affectively neutral Human/Assistant transcripts on the same 100 topics,
which is the neutral set appendix 6.5 of the paper describes and the one whose
top PCs the denoise step is meant to remove (generic chat-format variation).
``expression/neutral_stories.parquet`` is flat third-person narration, the same
kind of text as our own Llama neutral set, and not what the paper used.

Everything here is read-only with respect to the existing local corpus: rows are
written into whatever ``out_dir`` the caller names, in the exact JSONL shape
``stories.generate_story_set`` produces (``{emotion, topic, idx, text}`` plus
provenance), so ``stories.read_story_texts`` and the extraction pipeline consume
them without knowing which model wrote them.

The parquet files are pulled through ``huggingface_hub`` into the usual HF cache
(never into the repo) and filtered with polars, both already light local
dependencies, so this stays on the "no heavy ML deps locally" side of the line.
"""

import json
import re
from pathlib import Path

from .taxonomy import slugify

DEFAULT_REPO_ID = "ryancodrai/emotion-probes"
DEFAULT_STORIES_FILE = "expression/stories.parquet"
# The paper's neutral baseline: neutral Human/Assistant dialogues, not neutral stories.
DEFAULT_NEUTRAL_FILE = "deflection/neutral_dialogues.parquet"

# Recorded on each row so a set carries its own provenance.
DEFAULT_GENERATOR = "google/gemini-3.1-pro-preview"

# The text column differs per file (stories -> "story", dialogues -> "dialogue").
_TEXT_COLUMNS = ("story", "dialogue", "text")


def download_file(repo_id: str, filename: str) -> Path:
    """Fetch one file from a HF *dataset* repo into the local HF cache."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset"))


def _text_column(columns) -> str:
    for candidate in _TEXT_COLUMNS:
        if candidate in columns:
            return candidate
    raise ValueError(f"no text column among {_TEXT_COLUMNS} in {list(columns)}")


def _stratified_sample(df, n: int, seed: int):
    """Take ``n`` rows spread as evenly as possible across topics (seeded).

    Shuffling first and then sorting by within-topic rank means the draw takes at
    most one row per topic before it takes a second from any topic, so a small
    sample still sees the full topic range rather than a dozen rows about the
    same scenario.
    """
    import polars as pl

    return (
        df.sample(fraction=1.0, shuffle=True, seed=seed)
        .with_columns(pl.int_range(pl.len()).over("topic").alias("_rank"))
        .sort("_rank", maintain_order=True)
        .head(n)
        .drop("_rank")
    )


def normalize_transcript(text: str) -> str:
    """Put a neutral dialogue into the speaker format the paper's transcripts use.

    Appendix 6.5 ends the neutral dialogue prompt with "Post-hoc, we converted 'Person:'
    and 'AI:' to 'Human:' and 'Assistant:'", so uniform Human/Assistant labels are part of
    the input, not a cosmetic detail. The reproduction is not uniform about it -- a small
    fraction of transcripts use ``H:``/``A:``, and a few carry literal backslash-n instead
    of newlines -- and since the principal components being projected out are exactly the
    directions of generic formatting variation, leaving three speaker conventions in the
    baseline would put that inconsistency into the basis.
    """
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\n", "\n")
    text = re.sub(r"^(?:H|Person)\s*:", "Human:", text, flags=re.MULTILINE)
    return re.sub(r"^(?:A|AI)\s*:", "Assistant:", text, flags=re.MULTILINE)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _n_rows(path: Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def write_emotion_stories(
    emotions: list[str],
    n_per_emotion: int,
    out_dir: Path,
    *,
    seed: int = 0,
    repo_id: str = DEFAULT_REPO_ID,
    stories_file: str = DEFAULT_STORIES_FILE,
    generator: str = DEFAULT_GENERATOR,
    force: bool = False,
) -> dict[str, Path]:
    """Write ``n_per_emotion`` stories per emotion to ``out_dir/<slug>.jsonl``.

    One pass over the parquet: filter to the emotions we need, then take a seeded
    topic-stratified sample per emotion. Emotions already on disk with enough rows
    are skipped unless ``force``; the sample is seeded, so a resumed call
    reproduces the same stories.

    Returns ``{emotion: path}``. Raises if the dataset is short an emotion or short
    of stories for one, either of which would silently shrink the comparison set.
    """
    import polars as pl

    out_dir = Path(out_dir)
    paths = {e: out_dir / f"{slugify(e)}.jsonl" for e in emotions}
    todo = [e for e in emotions if force or _n_rows(paths[e]) < n_per_emotion]
    if not todo:
        print(f"[hf-stories] all {len(emotions)} emotions already have >= {n_per_emotion} stories.")
        return paths

    src = download_file(repo_id, stories_file)
    print(f"[hf-stories] scanning {src.name} for {len(todo)} emotions x {n_per_emotion} stories")
    frame = pl.scan_parquet(src).filter(pl.col("emotion").is_in(todo)).collect()
    text_col = _text_column(frame.columns)

    missing = sorted(set(todo) - set(frame["emotion"].unique().to_list()))
    if missing:
        raise ValueError(
            f"{repo_id} has no stories for {missing}. Check the spelling against the "
            f"vocabulary of the dataset, since a missing emotion would drop out of the "
            f"comparison silently and shift the across-emotion mean the vectors are "
            f"centered on."
        )

    for i, emotion in enumerate(sorted(todo)):
        subset = frame.filter(pl.col("emotion") == emotion)
        if subset.height < n_per_emotion:
            raise ValueError(
                f"{repo_id} has only {subset.height} stories for {emotion!r}, "
                f"fewer than the {n_per_emotion} requested."
            )
        # Vary the seed per emotion so two emotions do not draw the same topic order.
        sample = _stratified_sample(subset, n_per_emotion, seed + i)
        _write_jsonl(
            paths[emotion],
            [
                {
                    "emotion": emotion,
                    "topic": row["topic"],
                    "idx": j,
                    "text": row[text_col],
                    "generator": generator,
                    "source": f"{repo_id}/{stories_file}",
                }
                for j, row in enumerate(sample.iter_rows(named=True))
            ],
        )
    print(f"[hf-stories] wrote {len(todo)} emotion sets -> {out_dir}")
    return paths


def write_neutral_set(
    n: int,
    out_dir: Path,
    *,
    seed: int = 0,
    repo_id: str = DEFAULT_REPO_ID,
    neutral_file: str = DEFAULT_NEUTRAL_FILE,
    generator: str = DEFAULT_GENERATOR,
    force: bool = False,
    filename: str = "neutral.jsonl",
) -> Path:
    """Write ``n`` neutral baseline rows to ``out_dir/<filename>`` (same JSONL shape).

    Defaults to the neutral Human/Assistant dialogues of the paper; pass
    ``neutral_file="expression/neutral_stories.parquet"`` for the narration set.
    """
    import polars as pl

    path = Path(out_dir) / filename
    if not force and _n_rows(path) >= n:
        print(f"[hf-stories] {filename} already has >= {n} rows; nothing to do.")
        return path

    src = download_file(repo_id, neutral_file)
    frame = pl.scan_parquet(src).collect()
    text_col = _text_column(frame.columns)
    if frame.height < n:
        raise ValueError(f"{repo_id}/{neutral_file} has only {frame.height} rows, fewer than {n}.")
    sample = _stratified_sample(frame, n, seed)
    _write_jsonl(
        path,
        [
            {
                "emotion": "neutral",
                "topic": row["topic"],
                "idx": j,
                "text": normalize_transcript(row[text_col]),
                "generator": generator,
                "source": f"{repo_id}/{neutral_file}",
            }
            for j, row in enumerate(sample.iter_rows(named=True))
        ],
    )
    print(f"[hf-stories] wrote {n} neutral rows from {neutral_file} -> {path}")
    return path
