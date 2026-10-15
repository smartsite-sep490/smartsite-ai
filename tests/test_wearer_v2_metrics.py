import pytest
from pydantic import ValidationError

from smartsite_ai.evaluation.models import VALID_PPE_ITEMS
from smartsite_ai.evaluation.wearer_v2_metrics import measure_wearer_states
from smartsite_ai.evaluation.wearer_v2_models import (
    WEARER_ITEMS_V2,
    ReferenceStateSample,
    WearerEvaluationBatch,
    WearerMeasurementReport,
    WearerPredictionSample,
)


def reference(item="GLOVES", status="MISSING", person="person-a", clip="clip-a", frame=1):
    return ReferenceStateSample(
        clip_id=clip, frame_index=frame, person_instance_id=person, item=item, status=status
    )


def prediction(item="GLOVES", status="MISSING", person="person-a", clip="clip-a", frame=1):
    return WearerPredictionSample(
        clip_id=clip,
        frame_index=frame,
        person_instance_id=person,
        item=item,
        status=status,
        association="UNAMBIGUOUS",
    )


def batch(references, predictions=(), origin="SYNTHETIC"):
    return WearerEvaluationBatch(
        schema_version="2.0.0",
        reference_origin=origin,
        references=tuple(references),
        predictions=tuple(predictions),
    )


def metric(report, item="GLOVES"):
    return next(row for row in report.items if row.item == item)


@pytest.mark.parametrize("item", WEARER_ITEMS_V2)
def test_all_five_existing_wearer_items_are_reported_individually(item):
    report = measure_wearer_states(batch([reference(item)], [prediction(item)]))
    result = metric(report, item)
    assert result.correct_missing == 1
    assert result.missing_recall.value == 1.0
    assert result.decision_coverage.value == 1.0
    assert len(report.items) == 5
    assert report.model_accepted is False
    assert report.qualification_status == "NOT_ESTABLISHED"


def test_omitted_predictions_remain_in_missing_recall_and_coverage_denominators():
    report = measure_wearer_states(batch([reference(), reference(frame=2)], [prediction()]))
    result = metric(report)
    assert result.visible_missing == 2
    assert result.abstained_observable == 1
    assert result.missed_missing == 1
    assert result.decision_coverage.value == 0.5
    assert result.missing_recall.value == 0.5


def test_hidden_hands_require_abstention_and_do_not_create_missing_reference():
    result = metric(measure_wearer_states(batch([reference(status="UNOBSERVABLE")])))
    assert result.unobservable == 1
    assert result.visible_missing == 0
    assert result.claims_on_unobservable == 0
    assert result.missing_recall.value is None
    assert result.missing_recall.reason == "NO_VISIBLE_MISSING_REFERENCE"


@pytest.mark.parametrize("claim", ["PRESENT", "MISSING"])
def test_claims_on_unobservable_hand_are_counted_separately(claim):
    result = metric(
        measure_wearer_states(batch([reference(status="UNOBSERVABLE")], [prediction(status=claim)]))
    )
    assert result.claims_on_unobservable == 1
    assert result.unobservable_claim_rate.value == 1.0
    assert result.false_missing == 0


def test_false_missing_is_measured_against_visible_present_reference():
    result = metric(measure_wearer_states(batch([reference(status="PRESENT")], [prediction()])))
    assert result.false_missing == 1
    assert result.false_missing_rate.value == 1.0
    assert result.correct_present == 0


@pytest.mark.parametrize("scope", ["person", "clip", "frame"])
def test_wrong_subject_clip_or_frame_cannot_clear_an_omitted_reference(scope):
    arguments = (
        {"person": "person-b"}
        if scope == "person"
        else {scope: 2 if scope == "frame" else "clip-b"}
    )
    report = measure_wearer_states(batch([reference()], [prediction(**arguments)]))
    assert metric(report).abstained_observable == 1
    assert metric(report).correct_missing == 0
    assert report.unreviewed_predictions == 1


def test_ambiguous_two_person_association_can_only_submit_unknown():
    values = prediction().model_dump()
    values["association"] = "AMBIGUOUS"
    with pytest.raises(ValidationError, match="ambiguous"):
        WearerPredictionSample.model_validate(values)
    values["status"] = "UNKNOWN"
    unknown = WearerPredictionSample.model_validate(values)
    result = metric(measure_wearer_states(batch([reference()], [unknown])))
    assert result.abstained_observable == 1
    assert result.correct_missing == 0


@pytest.mark.parametrize("field", ["references", "predictions"])
def test_duplicate_or_conflicting_keys_refuse_instead_of_overwriting(field):
    values = batch([reference()], [prediction()]).model_dump()
    values[field] = values[field] + values[field]
    with pytest.raises(ValidationError, match="duplicate"):
        WearerEvaluationBatch.model_validate(values)


@pytest.mark.parametrize("item", ["MASK", "HARNESS", "LANYARD_HOOK"])
def test_object_only_families_are_not_wearer_labels(item):
    with pytest.raises(ValidationError):
        reference(item=item)


def test_loose_object_without_subject_and_unknown_payload_fields_are_refused():
    values = prediction().model_dump()
    del values["person_instance_id"]
    with pytest.raises(ValidationError):
        WearerPredictionSample.model_validate(values)
    values = reference().model_dump()
    values["worker_id"] = "invented-worker"
    with pytest.raises(ValidationError):
        ReferenceStateSample.model_validate(values)


@pytest.mark.parametrize("person", [None, True, 1.5, "", " padded "])
def test_subject_ids_are_explicit_strict_offline_references(person):
    with pytest.raises(ValidationError):
        reference(person=person)


def test_unused_families_have_undefined_rates_not_perfect_scores():
    result = metric(measure_wearer_states(batch([reference()])), "BOOTS")
    assert result.decision_coverage.value is None
    assert result.decision_coverage.reason == "NO_OBSERVABLE_REFERENCE"
    assert result.missing_recall.value is None


def test_declared_qualification_origin_never_self_certifies_acceptance():
    report = measure_wearer_states(batch([reference()], [prediction()], "QUALIFICATION_CANDIDATE"))
    assert report.reference_origin == "QUALIFICATION_CANDIDATE"
    assert report.qualification_status == "NOT_ESTABLISHED"
    with pytest.raises(ValidationError):
        WearerMeasurementReport.model_validate({**report.model_dump(), "model_accepted": True})
    assert WearerMeasurementReport.model_validate_json(report.model_dump_json()) == report


def test_v1_capabilities_remain_unchanged():
    assert VALID_PPE_ITEMS == ("HARD_HAT", "SAFETY_VEST")


@pytest.mark.parametrize("item", WEARER_ITEMS_V2)
def test_explicit_unknown_counts_as_abstention_without_losing_visible_samples(item):
    result = metric(
        measure_wearer_states(batch([reference(item)], [prediction(item, "UNKNOWN")])), item
    )
    assert result.abstained_observable == 1
    assert result.missed_missing == 1
    assert result.decision_coverage.value == 0.0
    assert result.missing_recall.value == 0.0


def test_explicit_unknown_on_hidden_hand_is_not_observable_abstention():
    values = prediction(status="UNKNOWN").model_dump()
    values["association"] = "AMBIGUOUS"
    result = metric(
        measure_wearer_states(
            batch(
                [reference(status="UNOBSERVABLE")], [WearerPredictionSample.model_validate(values)]
            )
        )
    )
    assert result.unobservable_claim_rate.value == 0.0
    assert result.abstained_observable == 0
    assert result.decision_coverage.value is None


@pytest.mark.parametrize("status", ["PRESENT", "MISSING"])
def test_both_ambiguous_claims_are_refused(status):
    values = prediction(status=status).model_dump()
    values["association"] = "AMBIGUOUS"
    with pytest.raises(ValidationError, match="ambiguous"):
        WearerPredictionSample.model_validate(values)


@pytest.mark.parametrize("field", ["references", "predictions"])
def test_conflicting_states_on_same_key_are_refused(field):
    values = batch([reference()], [prediction()]).model_dump()
    conflict = {**values[field][0], "status": "PRESENT"}
    values[field] += (conflict,)
    with pytest.raises(ValidationError, match="duplicate"):
        WearerEvaluationBatch.model_validate(values)


@pytest.mark.parametrize("item", ["MASK", "HARNESS", "LANYARD_HOOK", "HOOK"])
def test_predictions_cannot_claim_object_only_items_as_wearer_status(item):
    with pytest.raises(ValidationError):
        prediction(item=item)


def test_item_key_is_independent_with_mixed_present_missing_hidden_references():
    report = measure_wearer_states(
        batch(
            [
                reference("GLOVES"),
                reference("BOOTS", "PRESENT"),
                reference("GOGGLES", "UNOBSERVABLE"),
            ],
            [prediction("GLOVES"), prediction("BOOTS", "PRESENT")],
        )
    )
    assert metric(report).correct_missing == 1
    assert metric(report, "BOOTS").correct_present == 1
    assert metric(report, "GOGGLES").unobservable == 1
    assert report.unreviewed_predictions == 0


@pytest.mark.parametrize(
    "field,value", [("frame_index", -1), ("frame_index", True), ("clip_id", " padded ")]
)
def test_frame_and_clip_keys_reject_coercion_and_invalid_scope(field, value):
    with pytest.raises(ValidationError):
        ReferenceStateSample.model_validate({**reference().model_dump(), field: value})


def test_qualification_status_cannot_be_upgraded_by_report_json():
    report = measure_wearer_states(batch([reference()]))
    with pytest.raises(ValidationError):
        WearerMeasurementReport.model_validate(
            {**report.model_dump(), "qualification_status": "ACCEPTED"}
        )


def test_false_present_claim_is_separate_from_abstention_on_missing_item():
    result = metric(measure_wearer_states(batch([reference()], [prediction(status="PRESENT")])))
    assert result.false_present == 1
    assert result.false_present_rate.value == 1.0
    assert result.abstained_observable == 0
    assert result.missed_missing == 1


@pytest.mark.parametrize("identifier", ["a\u0085b", "a\u009fb", "a\ud800b", "a\udfffb"])
def test_nonprintable_and_surrogate_subject_identifiers_are_refused(identifier):
    with pytest.raises(ValidationError):
        reference(person=identifier)


@pytest.mark.parametrize(
    "change", ["items", "limitations", "unreviewed", "reference_total", "rate", "partition"]
)
def test_deserialized_reports_cannot_silently_change_coverage_or_item_partitions(change):
    report = measure_wearer_states(batch([reference()], [prediction()]))
    values = report.model_dump()
    if change == "items":
        values["items"] = (values["items"][0],) * 5
    elif change == "limitations":
        values["limitations"] = ()
    elif change == "unreviewed":
        values["unreviewed_predictions"] = 99
    elif change == "reference_total":
        values["reference_samples"] = 99
    elif change == "rate":
        values["items"][2]["missing_recall"] = {
            "numerator": 0,
            "denominator": 1,
            "value": 0.0,
            "reason": None,
        }
    else:
        values["items"][2]["missed_missing"] = 1
    with pytest.raises(ValidationError):
        WearerMeasurementReport.model_validate(values)
