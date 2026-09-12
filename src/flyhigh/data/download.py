"""Download the male-CNS flat connectome tables (annotations, neurotransmitters, weights).

    python -m flyhigh.data.download [--dest data/raw]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import requests
from tqdm import tqdm

BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"


@dataclass(frozen=True)
class RemoteFile:
    name: str
    approx_mb: int

    @property
    def url(self) -> str:
        return BASE + self.name


FILES = [
    RemoteFile("body-annotations-male-cns-v1.0-minconf-0.5.feather", 14),
    RemoteFile("body-neurotransmitters-male-cns-v1.0.feather", 43),
    RemoteFile("connectome-weights-male-cns-v1.0-minconf-0.5.feather", 1051),
]


def _remote_size(url: str) -> int:
    r = requests.head(url, allow_redirects=True, timeout=30)
    r.raise_for_status()
    return int(r.headers.get("content-length", 0))


def _stream(url: str, dest: Path, size: int) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r, open(tmp, "wb") as f:
        r.raise_for_status()
        with tqdm(total=size or None, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def _verify_opens(path: Path) -> None:
    """Read only the Arrow schema (no data) to check the file is a valid Feather table."""
    with pa.memory_map(str(path)) as source:
        pa.ipc.open_file(source).schema


def download_all(dest: str | Path = "data/raw", only: list[str] | None = None) -> list[Path]:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for rf in FILES:
        if only and rf.name not in only:
            continue
        path = dest / rf.name
        size = _remote_size(rf.url)
        if path.exists() and (size == 0 or path.stat().st_size == size):
            print(f"✓ {rf.name} already present")
        else:
            _stream(rf.url, path, size)
        _verify_opens(path)
        out.append(path)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default="data/raw")
    args = ap.parse_args()
    download_all(args.dest)


if __name__ == "__main__":
    main()
