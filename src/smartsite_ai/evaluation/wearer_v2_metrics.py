"""Account for every submitted reference, including omitted/UNKNOWN predictions."""

from smartsite_ai.evaluation.wearer_v2_models import (
    WEARER_ITEMS_V2,
    ItemStateMeasurement,
    MeasuredRate,
    WearerEvaluationBatch,
    WearerMeasurementReport,
)


def _rate(numerator: int, denominator: int, reason: str) -> MeasuredRate:
    return MeasuredRate(
        numerator=numerator,
        denominator=denominator,
        value=numerator / denominator if denominator else None,
        reason=None if denominator else reason,
    )


def measure_wearer_states(batch: WearerEvaluationBatch) -> WearerMeasurementReport:
    """Compare pre-associated states; never infer body visibility or Worker identity."""
    predictions = {row.key(): row for row in batch.predictions}
    reference_keys = {row.key() for row in batch.references}
    reports = []
    for item in WEARER_ITEMS_V2:
        counts = dict.fromkeys(
            (
                "visible_present",
                "visible_missing",
                "unobservable",
                "correct_present",
                "correct_missing",
                "false_missing",
                "false_present",
                "missed_missing",
                "abstained_observable",
                "claims_on_unobservable",
            ),
            0,
        )
        for reference in batch.references:
            if reference.item != item:
                continue
            prediction = predictions.get(reference.key())
            status = prediction.status if prediction else "UNKNOWN"
            if reference.status == "UNOBSERVABLE":
                counts["unobservable"] += 1
                counts["claims_on_unobservable"] += int(status != "UNKNOWN")
                continue
            counts["visible_present" if reference.status == "PRESENT" else "visible_missing"] += 1
            counts["abstained_observable"] += int(status == "UNKNOWN")
            counts["correct_present"] += int(reference.status == "PRESENT" and status == "PRESENT")
            counts["correct_missing"] += int(reference.status == "MISSING" and status == "MISSING")
            counts["false_missing"] += int(reference.status == "PRESENT" and status == "MISSING")
            counts["false_present"] += int(reference.status == "MISSING" and status == "PRESENT")
            counts["missed_missing"] += int(reference.status == "MISSING" and status != "MISSING")
        observable = counts["visible_present"] + counts["visible_missing"]
        reports.append(
            ItemStateMeasurement(
                item=item,
                **counts,
                decision_coverage=_rate(
                    observable - counts["abstained_observable"],
                    observable,
                    "NO_OBSERVABLE_REFERENCE",
                ),
                missing_recall=_rate(
                    counts["correct_missing"],
                    counts["visible_missing"],
                    "NO_VISIBLE_MISSING_REFERENCE",
                ),
                false_missing_rate=_rate(
                    counts["false_missing"],
                    counts["visible_present"],
                    "NO_VISIBLE_PRESENT_REFERENCE",
                ),
                false_present_rate=_rate(
                    counts["false_present"],
                    counts["visible_missing"],
                    "NO_VISIBLE_MISSING_REFERENCE",
                ),
                unobservable_claim_rate=_rate(
                    counts["claims_on_unobservable"],
                    counts["unobservable"],
                    "NO_UNOBSERVABLE_REFERENCE",
                ),
            )
        )
    return WearerMeasurementReport(
        reference_origin=batch.reference_origin,
        reference_samples=len(batch.references),
        submitted_predictions=len(batch.predictions),
        unreviewed_predictions=sum(row.key() not in reference_keys for row in batch.predictions),
        items=tuple(reports),
        limitations=(
            "Submitted origin/visibility/person references are declarations, "
            "not verified human GT.",
            "No source rights, independence, sample adequacy or risk budgets are attested.",
            "Counts are correlated frame/item samples, not independent people or incidents.",
            "No detector inference, episode latency, temporal quality or camera capacity measured.",
            "Unreviewed predictions are not automatically false positives.",
            "Mask, Harness and Hook remain object-only and cannot be scored as wearer states.",
        ),
    )
