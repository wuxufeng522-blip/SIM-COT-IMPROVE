from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import shutil
import time
import urllib.request


def download_part(
    url: str,
    part_path: Path,
    start: int,
    end: int,
    retries: int = 12,
) -> tuple[Path, int]:
    expected = end - start + 1
    part_path.parent.mkdir(parents=True, exist_ok=True)
    existing = part_path.stat().st_size if part_path.exists() else 0
    if existing > expected:
        part_path.unlink()
        existing = 0
    if existing == expected:
        return part_path, expected

    for attempt in range(1, retries + 1):
        current_start = start + existing
        request = urllib.request.Request(
            url,
            headers={
                "Range": f"bytes={current_start}-{end}",
                "User-Agent": "pip/26.2",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 206:
                    raise RuntimeError(f"Expected HTTP 206, got {response.status}")
                with part_path.open("ab") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        existing += len(chunk)
            if existing == expected:
                return part_path, expected
            if existing > expected:
                raise RuntimeError(
                    f"Part {part_path.name} exceeded expected size: {existing}>{expected}"
                )
        except Exception:
            if attempt == retries:
                raise
            time.sleep(min(2**attempt, 30))
    raise AssertionError("unreachable")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    parts_dir = args.output.parent / f"{args.output.name}.parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_size = (args.size + args.workers - 1) // args.workers
    tasks: list[tuple[Path, int, int]] = []
    for index in range(args.workers):
        start = index * part_size
        if start >= args.size:
            break
        end = min(args.size - 1, start + part_size - 1)
        tasks.append((parts_dir / f"part-{index:03d}", start, end))

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_part, args.url, path, start, end): path
            for path, start, end in tasks
        }
        completed = 0
        for future in as_completed(futures):
            path, size = future.result()
            completed += size
            elapsed = max(time.monotonic() - started, 1e-6)
            print(
                f"completed {path.name}: {completed / args.size:.1%}; "
                f"aggregate {completed / elapsed / 1024 / 1024:.2f} MiB/s",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as destination:
        for path, _, _ in tasks:
            with path.open("rb") as source:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)

    actual_size = args.output.stat().st_size
    if actual_size != args.size:
        raise RuntimeError(f"Size mismatch: {actual_size} != {args.size}")
    actual_hash = sha256_file(args.output)
    if actual_hash.lower() != args.sha256.lower():
        raise RuntimeError(f"SHA-256 mismatch: {actual_hash} != {args.sha256}")
    print(f"verified {args.output} size={actual_size} sha256={actual_hash}", flush=True)


if __name__ == "__main__":
    main()
