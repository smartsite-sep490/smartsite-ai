# Whole camera configuration bundles

This is parallel source development against the Platform feature branch. The
canonical contract still requires its final immutable Main SHA to be vendored
before paired release acceptance. Vendored schemas, golden files and provenance
are unchanged. Do not roll this worker source out to the existing demo as part of
a model experiment.

`BackendConfigurationClient.fetch_bundle` reads
`/api/v1/integrations/ai/cameras/:cameraId/configuration-bundle`. It validates the
whole immutable bundle, canonical SHA-256, requested camera UUID and external ID.
The shared conditional transport preserves bounded body reads, ETag handling,
cancellation and classified retries. A 304 supplies no new bundle or application
receipt. The legacy `fetch` method remains available to existing explicit callers.

`CameraConfigurationPoller` now uses the bundle endpoint. The target Backend must
provide it before this source is used. A successful response atomically stores
the full bundle, including per-region Zone, policy revision and required PPE.
The store retains a stable geometry projection for explicit legacy callers and
returns the whole bundle atomically to the headless frame worker. Either active
snapshot changes identity when policy changes, even if the polygon does not.
Geometry-only updates cannot overwrite a loaded policy bundle.

With a full bundle, the pipeline evaluates configured PPE independently in every
region, retaining each observation's region and geometry version. Empty requirements
disable only that binding; they do not cancel overlapping regions. Unsupported or
unobserved PPE stays unknown without inventing missing evidence. Geometry-only
snapshots retain the legacy single-PPE-region behavior. Full bundles emit technical
Zone entries for every configured region, including a region also used for PPE.
They do not require the legacy manifest PPE anchor to remain present. Both
preflight paths accept an empty full bundle; it means no regions configured,
not proof of a clear or compliant scene. Legacy geometry-only snapshots retain
their anchor guard and exclusion. Backend still owns permissions and violations.

Configuration and captured-view versions cannot regress. Changing content at the
same configuration version or changing the camera UUID behind an existing bundle
is a conflict. An identical bundle preserves object identity. A legacy snapshot
can upgrade at the same version only when its geometry matches the bundle.

The frame worker resets temporal PPE confirmation when the immutable active
snapshot changes or a new stream session starts. This includes a same-version
legacy-to-bundle upgrade; an idempotent read preserves the snapshot identity.
Frames from different policy contexts cannot accumulate into one confirmation.
Live-frame rollback also restores the previous snapshot after a discarded frame.

The poller/runtime do not yet send application ACKs, provision worker generations, verify
physical camera pose, qualify a model or establish production acceptance. Operator
credential injection, automatic ACK delivery, final contract vendoring and actual paired
runtime evidence remain release gates. No camera or model is started by import or
by the focused unit checks.

The explicit `acknowledge_configuration` client method validates the canonical
request and requires a `SecretStr` containing the exact 43-character generation
credential, alongside the existing service bearer token. It sends at most4096
bytes and accepts a bounded receipt bound to command ID, configuration version
and outcome. Transient retries reuse identical bytes; semantic/auth conflicts
are not retried. An idempotent receipt retains the original server receipt time.
The response contains no generation credential. Automatic invocation must wait
for runtime policy/capability validation and operator generation/session binding.

## Region-bound temporal confirmation

PPE confirmation, clearing and cooldown are scoped to stream session, person track,
PPE item, region ID and geometry version. Consecutive missing evidence from one
region cannot confirm another region or a revised polygon. Delivery also matches
the candidate's exact region geometry. This internal binding does not change event
schemas, infer identity, authorize access or establish model accuracy.
