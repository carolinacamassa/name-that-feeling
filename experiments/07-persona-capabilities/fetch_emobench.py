"""Download the EmoBench items (Sabour et al. 2024) from the authors' repository at a
pinned commit into ``data/emobench/{EU,EA}.jsonl``.

github.com/Sahandfer/EmoBench, MIT licence; the two JSONL files hold the 200 + 200
English scenarios and their Chinese versions (``language`` field). Pinned so the
prompts and labels a run was scored against are reproducible.

    uv run python experiments/07-persona-capabilities/fetch_emobench.py
"""

import urllib.request

import common

COMMIT = "3b4f84312541f8469e909965e1fff3691ef85c62"  # master on 2026-09-10
RAW = f"https://raw.githubusercontent.com/Sahandfer/EmoBench/{COMMIT}/data"


def main() -> None:
    out_dir = common.DATA / "emobench"
    out_dir.mkdir(parents=True, exist_ok=True)
    for task in ("EU", "EA"):
        target = out_dir / f"{task}.jsonl"
        if target.exists():
            print(f"{target} exists, kept")
            continue
        with urllib.request.urlopen(f"{RAW}/{task}.jsonl") as r:
            payload = r.read().decode("utf-8")
        target.write_text(payload, encoding="utf-8", newline="\n")
        print(f"wrote {target}: {payload.count(chr(10))} lines")
    (out_dir / "SOURCE.txt").write_text(f"github.com/Sahandfer/EmoBench @ {COMMIT}\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
