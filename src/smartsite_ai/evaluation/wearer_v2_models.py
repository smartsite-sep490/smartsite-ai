"""Offline state measurement only; no model, identity or qualification acceptance."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

WearerItemV2 = Literal["HARD_HAT", "SAFETY_VEST", "GLOVES", "BOOTS", "GOGGLES"]
WEARER_ITEMS_V2: tuple[WearerItemV2, ...] = (
    "HARD_HAT",
    "SAFETY_VEST",
    "GLOVES",
    "BOOTS",
    "GOGGLES",
)
ReferenceOrigin = Literal["SYNTHETIC", "DEVELOPMENT", "QUALIFICATION_CANDIDATE"]
SubjectId = int | str


class MeasurementModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class SubjectItemKey(MeasurementModel):
    clip_id: str = Field(min_length=1, max_length=128)
    frame_index: int = Field(ge=0, le=10_000_000)
    person_instance_id: SubjectId
    item: WearerItemV2

    @field_validator("clip_id", "person_instance_id")
    @classmethod
    def validate_identifier(cls, value: SubjectId) -> SubjectId:
        if isinstance(value, str):
            if not value or value != value.strip() or len(value) > 128:
                raise ValueError("identifier must be nonblank, bounded and unpadded")
            if not value.isprintable():
                raise ValueError("identifier cannot contain control characters")
        elif isinstance(value, bool) or not 0 <= value <= 9_007_199_254_740_991:
            raise ValueError("numeric subject identifier must be a safe nonnegative integer")
        return value

    def key(self) -> tuple[str, int, SubjectId, WearerItemV2]:
        return self.clip_id, self.frame_index, self.person_instance_id, self.item


class ReferenceStateSample(SubjectItemKey):
    status: Literal["PRESENT", "MISSING", "UNOBSERVABLE"]


class WearerPredictionSample(SubjectItemKey):
    status: Literal["PRESENT", "MISSING", "UNKNOWN"]
    association: Literal["UNAMBIGUOUS", "AMBIGUOUS"]

    @model_validator(mode="after")
    def refuse_ambiguous_claim(self) -> Self:
        if self.association == "AMBIGUOUS" and self.status != "UNKNOWN":
            raise ValueError("ambiguous association can only submit UNKNOWN")
        return self


class WearerEvaluationBatch(MeasurementModel):
    schema_version: Literal["2.0.0"]
    # A caller declaration, not verified rights, human review or independent lineage.
    reference_origin: ReferenceOrigin
    references: tuple[ReferenceStateSample, ...] = Field(min_length=1, max_length=10_000)
    predictions: tuple[WearerPredictionSample, ...] = Field(max_length=10_000)

    @model_validator(mode="after")
    def refuse_duplicate_keys(self) -> Self:
        for rows in (self.references, self.predictions):
            keys = [row.key() for row in rows]
            if len(keys) != len(set(keys)):
                raise ValueError("duplicate subject/item/frame key")
        return self


class MeasuredRate(MeasurementModel):
    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)
    value: float | None = Field(ge=0.0, le=1.0)
    reason: str | None

    @model_validator(mode="after")
    def validate_rate(self) -> Self:
        if self.numerator > self.denominator:
            raise ValueError("numerator cannot exceed denominator")
        if self.denominator == 0:
            if self.value is not None or not self.reason:
                raise ValueError("empty denominator needs an undefined reason")
        elif self.value != self.numerator / self.denominator or self.reason is not None:
            raise ValueError("rate must match its counts")
        return self


class ItemStateMeasurement(MeasurementModel):
    item: WearerItemV2
    visible_present: int = Field(ge=0)
    visible_missing: int = Field(ge=0)
    unobservable: int = Field(ge=0)
    correct_present: int = Field(ge=0)
    correct_missing: int = Field(ge=0)
    false_missing: int = Field(ge=0)
    false_present: int = Field(ge=0)
    missed_missing: int = Field(ge=0)
    abstained_observable: int = Field(ge=0)
    claims_on_unobservable: int = Field(ge=0)
    decision_coverage: MeasuredRate
    missing_recall: MeasuredRate
    false_missing_rate: MeasuredRate
    false_present_rate: MeasuredRate
    unobservable_claim_rate: MeasuredRate

    @model_validator(mode="after")
    def validate_count_partitions(self) -> Self:
        present_unknown = self.visible_present - self.correct_present - self.false_missing
        missing_unknown = self.missed_missing - self.false_present
        if (
            present_unknown < 0
            or missing_unknown < 0
            or self.correct_missing + self.missed_missing != self.visible_missing
            or self.abstained_observable != present_unknown + missing_unknown
            or self.claims_on_unobservable > self.unobservable
        ):
            raise ValueError("inconsistent wearer count partitions")
        observable = self.visible_present + self.visible_missing
        expected = (
            (self.decision_coverage, observable - self.abstained_observable, observable),
            (self.missing_recall, self.correct_missing, self.visible_missing),
            (self.false_missing_rate, self.false_missing, self.visible_present),
            (self.false_present_rate, self.false_present, self.visible_missing),
            (self.unobservable_claim_rate, self.claims_on_unobservable, self.unobservable),
        )
        if any((rate.numerator, rate.denominator) != (n, d) for rate, n, d in expected):
            raise ValueError("rates must use the item's recorded counts")
        return self


class WearerMeasurementReport(MeasurementModel):
    schema_version: Literal["2.0.0"] = "2.0.0"
    scope: Literal["OFFLINE_SUBMITTED_WEARER_STATE_MEASUREMENT"] = (
        "OFFLINE_SUBMITTED_WEARER_STATE_MEASUREMENT"
    )
    reference_origin: ReferenceOrigin
    reference_samples: int = Field(ge=1)
    submitted_predictions: int = Field(ge=0)
    unreviewed_predictions: int = Field(ge=0)
    items: tuple[ItemStateMeasurement, ...] = Field(min_length=5, max_length=5)
    model_accepted: Literal[False] = False
    qualification_status: Literal["NOT_ESTABLISHED"] = "NOT_ESTABLISHED"
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_report_totals(self) -> Self:
        if tuple(row.item for row in self.items) != WEARER_ITEMS_V2:
            raise ValueError("report must contain all five items once in canonical order")
        total = sum(
            row.visible_present + row.visible_missing + row.unobservable for row in self.items
        )
        if total != self.reference_samples:
            raise ValueError("reference totals do not match item counts")
        if not 0 <= self.submitted_predictions - self.unreviewed_predictions <= total:
            raise ValueError("reviewed prediction counts exceed submitted/reference counts")
        if any(not text.strip() for text in self.limitations):
            raise ValueError("limitations must not be blank")
        return self
