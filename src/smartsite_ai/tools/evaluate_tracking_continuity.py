"""Print one matched-ledger continuity report. This is not a tracker benchmark."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

from smartsite_ai.evaluation.tracking_continuity import evaluate_tracking_continuity
from smartsite_ai.evaluation.tracking_diagnostics_io import load_tracking_diagnostics

_REJECTED = "tracking diagnostics input rejected"


def main(argv: Sequence[str] | None = None) -> int:
    """Read one local ledger and write the diagnostic report to stdout."""

    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 2 or arguments[0] != "--input" or not arguments[1]:
        print(_REJECTED, file=sys.stderr)
        return 2
    try:
        matches = load_tracking_diagnostics(Path(arguments[1]))
        report = evaluate_tracking_continuity(matches)
        serialized = json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True)
    except Exception:
        print(_REJECTED, file=sys.stderr)
        return 2
    sys.stdout.write(f"{serialized}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
