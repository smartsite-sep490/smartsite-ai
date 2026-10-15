"""Bounded offline state measurement; never opens models, media, cameras or network."""

import argparse
import json
import os
import stat
import sys
from pathlib import Path

from pydantic import ValidationError

from smartsite_ai.evaluation.wearer_v2_metrics import measure_wearer_states
from smartsite_ai.evaluation.wearer_v2_models import WearerEvaluationBatch

MAX_INPUT_BYTES = 4 * 1024 * 1024


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if not stat.S_ISREG(args.input.stat().st_mode):
            raise ValueError("input must be a regular file")
        with args.input.open("rb") as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise ValueError("input descriptor must be a regular file")
            payload = source.read(MAX_INPUT_BYTES + 1)
        if len(payload) > MAX_INPUT_BYTES:
            raise ValueError("input size limit")
        json.loads(payload, object_pairs_hook=_unique_fields)
        batch = WearerEvaluationBatch.model_validate_json(payload)
        report = measure_wearer_states(batch)
        serialized = report.model_dump_json(indent=2) + "\n"
        with args.output.open("x", encoding="utf-8") as target:
            target.write(serialized)
    except (OSError, ValueError, ValidationError, RecursionError):
        # Pydantic errors can include arbitrary submitted input: do not echo them.
        print(
            "Wearer measurement refused: invalid input or unavailable exclusive output.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
