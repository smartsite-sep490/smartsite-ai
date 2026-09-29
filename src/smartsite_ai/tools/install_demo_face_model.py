"""Explicit local installer for the non-commercial InsightFace buffalo_l demo pack."""

import argparse
import hashlib
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

ARCHIVE_URL = "https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip"
ARCHIVE_SHA256 = "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f"
REQUIRED = ("det_10g.onnx", "w600k_r50.onnx")


def run() -> None:
    parser = argparse.ArgumentParser(description="Install InsightFace buffalo_l academic demo model")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--accept-non-commercial-license", action="store_true")
    arguments = parser.parse_args()
    if not arguments.accept_non_commercial_license:
        parser.error("buffalo_l public weights are non-commercial research only; explicitly accept first")
    root = arguments.model_root.resolve()
    if not root.is_absolute():
        parser.error("model root must be an absolute path")
    target = root / "models" / "buffalo_l"
    if all((target / item).is_file() for item in REQUIRED):
        print(f"Already installed: {target}")
        return
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as temporary_directory:
        archive = Path(temporary_directory) / "buffalo_l.zip"
        with urlopen(ARCHIVE_URL, timeout=60) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != ARCHIVE_SHA256:
            raise RuntimeError("model archive integrity check failed")
        with zipfile.ZipFile(archive) as package:
            names = set(package.namelist())
            if not all(item in names for item in REQUIRED):
                raise RuntimeError("model archive does not contain required artifacts")
            staging = Path(temporary_directory) / "buffalo_l"
            staging.mkdir()
            for item in REQUIRED:
                package.extract(item, staging)
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if target.exists():
                raise RuntimeError("model target already exists but is incomplete; inspect it manually")
            shutil.move(str(staging), str(target))
    print(f"Installed verified buffalo_l demo artifacts at {target}")


if __name__ == "__main__":
    run()
