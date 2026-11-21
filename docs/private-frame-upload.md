# Private evidence from one AI host

The optional `privateEvidenceUpload: true` field on a camera manifest entry
uploads its original tracked LIVE JPEGs to the configured SmartSite Backend
before sending each tracked raw observation part. The default is `false`.
It requires `trackedV17`, `live: true`, a private `evidenceDir`, and an original
generation credential. This is not a switch to improve model accuracy or to
grant Zone access.

Backend support is canonical at Platform main
`b0917fd31a88bb840d0591ea97b2f82fdf4aacd5` (PR #294). The separate receipt
schema/example in `contracts/metadata.json` records those exact bytes and hashes.
Existing observation, configuration and raw-receipt contracts are unchanged.

For a reviewed rollout, the Backend needs its explicit tracked migration profile,
runtime policy and `AI_EVIDENCE_OBJECT_STORAGE_ENABLED=true`, with private R2 or
LOCAL storage configured server-side. Do not enable the flag on an unrelated
database profile. Flag-on tracked FRAME/SUBJECT readers select the new private
storage; required historical media must be migrated first. C02/Face preparation
has a separate reader and is not made cloud-ready by this flag.

## Delivery and failure handling

The worker retains original JPEGs and SQLite outbox parts on mounted private
storage. It reads only the event/session/sequence-bound local filename, checks
file bounds and hash, and sends bounded multipart data with both the AI service
credential and original worker-generation credential. Camera secrets are never
sent to the browser.

An `EVIDENCE_STORED` response must match event ID, original payload hash, binding
hash and JPEG hash. It acknowledges only stored bytes. The existing raw part
request must still return `RAW_STORED`, and the existing sealed-frame result
must still confirm delivery before continuity advances. Image receipt alone
never marks the SQLite outbox delivered or creates a violation.

Upload failures block raw delivery for that part. The existing bounded outbox
cycle retries transport errors, 408, 429 and 5xx using original durable bytes;
semantic 4xx errors and binding mismatches stop the cycle for operator recovery.
No HTTP redirect or unbounded response body is accepted. Expired LIVE generations
can no longer upload; keep evidence pending and recover under the reviewed
generation lifecycle rather than substituting a new credential for old events.
REPLAY upload is unsupported by this initial path; legacy replay behavior stays
unchanged when the option is off.

## Shared websites

One AI host sends to one canonical Backend/database. The deployed Web app and
a local Web build may both use that same Backend origin and authorized Site
accounts. They then see the same events, inbox revisions and evidence; running
another browser does not require another detector. Sending to two independent
Backend databases is a separate synchronization problem and is not implied.

Persisted evidence upload does not make live preview reachable. The Backend's
configured preview/source-host origin must reach the AI host through an
authenticated HTTPS endpoint (or a private network). Keep those server-only
allowlists and credentials; never replace them with a URL supplied by a Web
user. Windows Docker Desktop does not automatically expose the laptop webcam;
use a supported host capture device or the documented native capture path.

The existing PostgreSQL durable queues and AI SQLite outbox remain authoritative.
Redis is not required for this single-host delivery path. Add it only for a
measured shared transient-cache/pubsub need, with bounded retention; it must not
replace event history, authorization, or durable delivery state.

## Rollback

Disable the manifest option and Backend feature flag under the reviewed release
procedure. Preserve mounted evidence/outbox data and immutable private objects.
Do not delete shared data or rewrite original events as part of rollback. Camera,
GPU/model quality, RTSP reachability, real R2 transport and multi-client acceptance
must be verified separately before claiming an operational release.
