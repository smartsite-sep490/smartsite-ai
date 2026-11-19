# Optional person-anchored PPE composition

`PersonAnchoredPpeDetector` consumes two explicit detector instances. Each sees
the same immutable source frame. Person detections come from the person detector;
all non-Person detections come unchanged from the PPE detector. This avoids using
a specialized PPE checkpoint's poor Person localization as the only tracking
input. It does not suppress to a fixed count or identify a Worker.

Returned source camera, stream, session, capture time, sequence and dimensions
must match the input exactly. Provider errors and cancellation propagate; no
fallback to the old duplicate Person boxes is attempted. Output uses the
`person-anchored-ppe` / `v1` identity and a SHA-256 of the deterministic composition
descriptor containing both artifact IDs, versions and hashes. This digest binds
the composed detector manifest, not a third weights file. Runtime configuration
changes must be versioned in the artifact descriptors and use a new generation.

The caller explicitly loads, verifies and closes both model runners, including
partial-startup failure. Construction/import opens no files, camera, network or
model. Default single-model constructors are unchanged. Existing PPE/person
geometric association and unknown results still apply. Provider class IDs remain
local to their source; semantic consumers must use class names, not assume one
shared raw class-ID map.

A same-frame local comparison is a diagnostic, not person-count accuracy or
production acceptance. Separate person detection can still miss, duplicate or
merge people, especially with occlusion. Before broader rollout, measure those
errors and combined latency/memory on independent crowd/site/video data. Do not
turn every new technical track into a verified identity or merge historical
episodes to hide a tracking defect.
