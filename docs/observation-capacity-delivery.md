# Full-frame observation capacity and explicit partition delivery

MF05/MF06 frames may contain more than256 technical observations even when the
camera's configured region limit is respected. Observe the whole frame first,
confirm/filter once, then deterministically pack bounded events. Keep each subject
intact where possible; oversized subjects repeat PERSON/identity context and split
between region groups. Never truncate people, regions or confirmed findings.

Existing process() remains the legacy one-event interface. Headless worker,
diagnostic preview, video timeline and evaluation use observe() and the full-frame
representation. Preview totals do not represent just one transport part. Ordinary
small-event IDs/order remain unchanged. Atomic SQLite enqueue persists all frame
parts or none of its new parts; a retried part conflict cannot leave a partial insert.

Explicit worker partitioned_frame_delivery defaultsfalse. Opt-in v1.4 uses original
frame ID, exact source sequence text and actual part index/count; per-part event IDs
are stable and version-specific. JPEG metadata and canonical hashes survive durable
replay. Runtime assembly/configuration does not enable this candidate.

Legacy delivery refuses multi-part frames before image publication/outbox/network,
because it lacks canonical parent metadata and can inflate Backend counters. Internal
packing remains lossless; explicit v1.4 delivery and paired consumer release are
required to deliver a large frame. This is an explicit compatibility error, not silent
truncation or successful camera acceptance. Backend counterpart is smartsite PR61; canonical Main merge and
immutable AI vendoring remain required. No schema or provenance file is altered here.

Tests use synthetic full-frame/capture/detector fixtures, real JPEGs and SQLite;
real HTTP/PostgreSQL integration is recorded separately. These checks do not prove
model quality, live throughput, Worker identity, browser layout or production readiness.
Source-size exceptions: existing temporal/worker responsibilities exceed300lines;
Root keeps cancellation/rollback/delivery guards together for this bounded change.
No preview recovery, experimental model, live-source configuration or model adoption
is bundled in this commit. Existing dirty work remains separate.
