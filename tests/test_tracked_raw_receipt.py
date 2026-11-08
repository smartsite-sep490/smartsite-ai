import pytest
from pydantic import ValidationError

from smartsite_ai.domain.tracked_raw_receipt import TrackedRawReceipt

EVENT_ID = "00000000-0000-4000-8000-000000000019"
PAYLOAD = "a" * 64
BINDING = "b" * 64
RAW = {"kind": "RAW_STORED", "eventId": EVENT_ID, "payloadHash": PAYLOAD, "bindingHash": BINDING}


def test_exact_storage_receipt_matches_the_sent_part():
    receipt = TrackedRawReceipt.model_validate(RAW)
    receipt.require_match(EVENT_ID, PAYLOAD, BINDING)
    assert receipt.model_dump(by_alias=True) == RAW


@pytest.mark.parametrize(
    "change",
    [
        {"kind": "ADMITTED_TRACKED_FRAME"},
        {"kind": "PROCESSED"},
        {"subjects": []},
        {"eventId": 5},
        {"payloadHash": "A" * 64},
        {"bindingHash": "b" * 63},
    ],
)
def test_receipt_rejects_other_outcomes_and_invalid_fields(change):
    with pytest.raises(ValidationError):
        TrackedRawReceipt.model_validate({**RAW, **change})


@pytest.mark.parametrize(
    "expected",
    [
        ("00000000-0000-4000-8000-000000000020", PAYLOAD, BINDING),
        (EVENT_ID, "c" * 64, BINDING),
        (EVENT_ID, PAYLOAD, "c" * 64),
    ],
)
def test_valid_receipt_for_another_part_or_binding_cannot_acknowledge_this_part(expected):
    with pytest.raises(ValueError, match="receipt binding mismatch"):
        TrackedRawReceipt.model_validate(RAW).require_match(*expected)
