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
The store retains a stable geometry projection for current processing callers;
it replaces that projection when the full bundle changes, even if the polygon
does not. Geometry-only updates cannot overwrite a loaded policy bundle.

Configuration and captured-view versions cannot regress. Changing content at the
same configuration version or changing the camera UUID behind an existing bundle
is a conflict. An identical bundle preserves object identity. A legacy snapshot
can upgrade at the same version only when its geometry matches the bundle.

The frame worker resets temporal PPE confirmation when configuration version
changes, as well as when session or geometry changes. Frames from two policy
versions cannot accumulate into one confirmation. Its live-frame rollback also
restores the configuration version after a discarded frame.

The poller/runtime do not yet send application ACKs, provision worker generations, verify
physical camera pose, qualify a model or establish production acceptance. Operator
credential injection, ACK delivery, final contract vendoring and actual paired
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
