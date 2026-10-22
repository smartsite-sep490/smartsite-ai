# MF05 relation review intake

This implements the C1 Person–PPE relationship preparation step of the reviewed
SmartSite YOLO plan. It is an offline DRAFT artifact, never a training label,
wearer reference, qualification admission or technical observation event.

## Behavior and scope

- Retain the unified fifteen-class taxonomy; accept positive object rectangles
  for Person and the eight PPE families. Negative/completeness labels are not
  inferred from these rectangles.
- Bind each image to original bytes, encoded dimensions, EXIF orientation and
  normalized display coordinates. Coordinate approval receipts are separate
  provenance references, not approval of proposed wearing relationships.
- Each of the five observation-family objects has one review row with an
  optional local Person reference, body-part visibility and a proposed relation.
  Unassigned loose objects stay unresolved. Mask/Harness/Hook remain raw objects.
- A proposed worn relation requires a visible, manually marked expected body
  part and a valid Person reference. These structural checks do not verify that
  a person actually wears the object. No thresholds or anatomical estimates.
- Reject orphan/duplicate links, class mismatches, unsupported claims, invalid
  bounds, wrong image hashes and EXIF coordinate mismatches.
- Existing v1 evaluation, wearer-v2 state metrics, runtime, temporal confirmation,
  external contracts, checkpoints and source images remain unchanged.

## Execution

Root owns the new `evaluation/relation_review.py`, focused tests and this note.
Existing dirty AI source files belong to the ongoing implementation and are not
part of this slice. The only caller is offline reserved-data preparation.
The existing AI reviewer independently checks coverage and the final diff.

1. Record exact C04/C10 coordinate approval without broadening its scope.
2. Write focused parser/source-binding tests, observe failure, implement intake.
3. Prepare the four approved-coordinate photos as unresolved drafts; validate
   original bytes and approval references without opening a model.
4. Run the affected intake/legacy-review checks, review and commit locally.

Remaining: human relation/visibility review, negative/completeness coverage,
calibration, held-out relation evaluation, and runtime qualification. No model
or website acceptance follows from intake validation.

## Verified 22 October

The focused intake first failed because implementation was absent. Receipt
binding then failed its added regression; malformed receipt containers and
boolean/coercible values plus inconsistent EXIF metadata were also reproduced
before repair. Final affected run: 129 passing cases across relation intake,
legacy video-review preparation and wearer-v2 state measurement. Scoped Ruff
format/check passed. Full service/GPU suites were not rerun for this offline slice.

Four real reserved photos C01/C02/C04/C10 now have 18 coordinate-approved
objects and 11 unresolved relation rows. Source image SHA, receipt SHA, exact
class/ID/coordinates and EXIF frame were verified after JSON readback using the
final implementation. No Person linkage or wearing approval was generated.
Artifacts: `D:/Ky9-FPT/SmartSiteData/runs/reserved-ppe-relation-drafts-20261022-v1`.

Independent review identified receipt shape/coercion and explicit orientation
gaps; the regression cases now pass. Receipt verification checks trusted local
bytes and consistency; it cannot authenticate a human author or establish
source rights. A caller must verify both the source image and trusted receipt
before describing draft coordinates as bound. Parsing alone checks structure.
Legacy receipts without coordinate-space metadata are allowed only for EXIF1.

This commit is limited to the new intake, tests and this note. Existing dirty
runtime/profile changes are excluded; the review taxonomy is versioned locally
so this commit does not depend on an uncommitted opt-in runtime profile.
