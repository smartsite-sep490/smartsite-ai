"""Provider-neutral validation for supported local runtime device selectors."""

import re

_RUNTIME_DEVICE_PATTERN = re.compile(r"^(?:auto|cpu|cuda(?::[0-9]{1,2})?|[0-9]{1,2})$")


def validate_runtime_device(value: str) -> str:
    """Return one supported device selector or reject it before provider execution."""

    if not _RUNTIME_DEVICE_PATTERN.fullmatch(value):
        raise ValueError("device must be auto, cpu, cuda, cuda:N, or a numeric CUDA index")
    return value
