"""Read-only local split audit CLI; no inference, remapping or split assignment."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

from pydantic import ValidationError

from smartsite_ai.evaluation.split_audit import SplitAuditManifest, audit_split_groups
from smartsite_ai.training.dataset_integrity import is_link_like

MAX_MANIFEST_BYTES = 32 * 1024 * 1024


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _safe_path(path: Path) -> None:
    if not path.is_absolute():
        raise ValueError("paths must be absolute")
    for part in (path, *path.parents):
        try:
            part.lstat()
        except FileNotFoundError:
            continue
        if is_link_like(part):
            raise ValueError("paths must not contain links or reparse points")


def run(manifest_path: Path, output_path: Path) -> int:
    _safe_path(manifest_path)
    _safe_path(output_path)
    if not manifest_path.is_file():
        raise ValueError("manifest must be a regular file")
    if output_path.exists() or not output_path.parent.is_dir():
        raise ValueError("output must be new with an existing parent directory")
    before = manifest_path.stat()
    with manifest_path.open("rb") as handle:
        raw = handle.read(MAX_MANIFEST_BYTES + 1)
    after = manifest_path.stat()
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError("manifest exceeds 32 MiB")
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        raise ValueError("manifest changed during read")
    payload = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_reject_duplicate_keys)
    manifest = SplitAuditManifest.model_validate(payload)
    report = audit_split_groups(manifest)
    report["manifestSha256"] = hashlib.sha256(raw).hexdigest()
    fd, temp = tempfile.mkstemp(prefix=".split-audit-", dir=output_path.parent)
    temporary = Path(temp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication cannot replace a concurrent writer's output.
        os.link(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"status": report["status"], "report": str(output_path)}))
    return 0 if report["status"] == "NO_KNOWN_OVERLAP" else 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        code = run(args.manifest, args.output)
    except (OSError, UnicodeError, ValueError, ValidationError):
        print("Dataset split audit rejected: invalid manifest or file boundary", file=sys.stderr)
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
