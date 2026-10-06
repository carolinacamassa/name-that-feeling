"""Download one large Hugging Face Hub file with parallel range requests.

``hf_hub_download`` crawled at ~10 KB/s on this connection while a plain range
request against the same CDN URL ran at ~1 MB/s (2026-09-02), so big dataset shards
are fetched here as concurrent byte ranges written in place. The result is verified
against the Hub's recorded size and LFS sha256, and a file that already verifies is
left alone, so calling this again is free.
"""

from __future__ import annotations

import hashlib
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi


def hub_file_info(repo_id: str, filename: str, *, repo_type: str = "dataset", token: str | None = None) -> dict:
    """{'size': bytes, 'sha256': hex or None} for one file, from the Hub's tree metadata."""
    (entry,) = HfApi(token=token).get_paths_info(repo_id, [filename], repo_type=repo_type)
    lfs = getattr(entry, "lfs", None)
    return {"size": getattr(entry, "size", None), "sha256": getattr(lfs, "sha256", None) if lfs else None}


def sha256_of(path: Path, block: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def download_hub_file(
    repo_id: str,
    filename: str,
    dest: Path,
    *,
    repo_type: str = "dataset",
    token: str | None = None,
    workers: int = 8,
    range_bytes: int = 16 << 20,
    retries: int = 4,
    log=print,
) -> dict:
    """Fetch ``filename`` to ``dest`` as parallel byte ranges; returns the verified size/sha256.

    Skips the transfer when ``dest`` already has the Hub's size and sha256. Raises if
    the finished file does not match either.
    """
    info = hub_file_info(repo_id, filename, repo_type=repo_type, token=token)
    size, sha = info["size"], info["sha256"]
    if size is None:
        raise RuntimeError(f"{repo_id}/{filename}: the Hub reports no size")
    if dest.exists() and dest.stat().st_size == size and (sha is None or sha256_of(dest) == sha):
        log(f"  {dest.name}: already on disk and verified")
        return info

    prefix = "datasets/" if repo_type == "dataset" else "spaces/" if repo_type == "space" else ""
    url = f"https://huggingface.co/{prefix}{repo_id}/resolve/main/{filename}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        f.truncate(size)
    ranges = [(start, min(start + range_bytes, size) - 1) for start in range(0, size, range_bytes)]
    done = 0
    t0 = time.time()

    def fetch(span: tuple[int, int]) -> int:
        start, end = span
        last: Exception | None = None
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={**headers, "Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if len(data) != end - start + 1:
                    raise RuntimeError(f"short range {start}-{end}: {len(data)} bytes")
                with dest.open("r+b") as f:
                    f.seek(start)
                    f.write(data)
                return len(data)
            except Exception as exc:  # transient CDN / connection errors: retry the range
                last = exc
                time.sleep(min(2**attempt, 15))
        raise RuntimeError(f"range {start}-{end} failed after {retries} attempts") from last

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, n in enumerate(pool.map(fetch, ranges), 1):
            done += n
            if i % 8 == 0 or i == len(ranges):
                rate = done / 1e6 / max(time.time() - t0, 1e-6)
                log(f"  {dest.name}: {done / 1e6:,.0f}/{size / 1e6:,.0f} MB ({rate:.1f} MB/s)")
    if dest.stat().st_size != size:
        raise RuntimeError(f"{dest}: size {dest.stat().st_size} != Hub size {size}")
    if sha is not None:
        got = sha256_of(dest)
        if got != sha:
            raise RuntimeError(f"{dest}: sha256 {got} != Hub sha256 {sha}")
    return info
