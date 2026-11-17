"""Download and extract the official Waterbirds release (about 490 MB).

Works on Windows, Linux and macOS (no wget or tar needed).

    python scripts/download_waterbirds.py --dest data

URL from the Group DRO repository README (github.com/kohpangwei/group_DRO).
The archive's SHA-256 is written next to it so you can report it.
"""

import argparse
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from srl.utils import sha256_file  # noqa: E402

URL = "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz"
NAME = "waterbird_complete95_forest2water2"


def _progress(block, block_size, total):
    done = block * block_size
    if total > 0 and (block % 500 == 0 or done >= total):
        print(f"\r  {min(done, total) / 2**20:7.1f} / {total / 2**20:.1f} MB", end="", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", default="data")
    args = p.parse_args(argv)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    folder, archive = dest / NAME, dest / f"{NAME}.tar.gz"

    if (folder / "metadata.csv").exists():
        print(f"already extracted: {folder.resolve()}")
        return
    if not archive.exists():
        print(f"downloading {URL}")
        part = archive.with_name(archive.name + ".part")
        t0 = time.time()
        urllib.request.urlretrieve(URL, part, _progress)
        part.replace(archive)
        print(f"\n  done in {time.time() - t0:.0f}s")
    print("extracting ...")
    with tarfile.open(archive, "r:gz") as tar:
        try:
            tar.extractall(dest, filter="data")      # Python 3.12+: safe extraction
        except TypeError:
            tar.extractall(dest)
    digest = sha256_file(archive)
    (dest / f"{NAME}.tar.gz.sha256").write_text(f"{digest}  {archive.name}\n")
    n_jpg = sum(1 for _ in folder.rglob("*.jpg"))
    print(f"extracted to {folder.resolve()} ({n_jpg} jpg files)\narchive sha256 {digest}")


if __name__ == "__main__":
    main()
