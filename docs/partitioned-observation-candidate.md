# Partitioned observation model/media candidate

This source prepares explicit v1.4 frame-part data for the Backend-owned canonical
candidate. It does not change pipeline defaults, adopt a model or finalize vendored
schema/provenance. Platform counterpart: smartsite PR61, head d71e12a; immutable
vendoring still requires its resulting authorized Main commit.

The Python model requires lowercase frame UUID, exact uint64 decimal sequence text,
bounded part index/count and exactly one same-part PERSON per referenced track.
Legacy events reject framePartition. Existing five-family wearer vocabulary remains;
this is not complete support for every canonical object class or model accuracy.

Explicit v1.4 JPEG publication requires content hashing and source-sequence equality
before writing. It preserves the version/metadata and actual JPEG SHA. Delivery
filter reconstruction preserves framePartition. Historical versions retain their
wire shape and prior evidence-version behavior.

This commit intentionally excludes the dirty full-frame packer, runtime opt-in,
preview and camera-recovery work. Those are separate coupled source tasks, not
implicitly included through the working directory. No runtime emits v1.4 by default.
The domain model's existing file exceeds300 lines; Root accepts this scoped addition
to the wire-model responsibility (354lines), with extraction before further growth.

Validation uses an index-only source snapshot, existing installed environment,
focused model/hash/media/filter/outbox tests and lint/format. No weights, recording,
real credentials, GPU, paid API or camera is involved. Broader dirty-worktree tests
and synthetic Backend HTTP/PostgreSQL proofs are separate evidence; they cannot
prove that this commit alone enables the complete producer or a release.
