# Explicit source-process handover

`SourceProcessDriver` may be composed with two reviewed argument tuples:
`command` for initial source activation and optional `resume_command` for resuming
an accepted configuration. Both require an absolute executable path. Omitting
`resume_command` retains the original single-command behavior. Neither command
comes from browser input, and this option does not enable host capabilities by
itself.

The resume command is selected only by `start_resume` after the existing Camera,
generation, session and exact configuration-bundle checks pass. Source and
generation remain child environment bindings; the expected resume manifest is
removed for ordinary activation and included for validated resume. No shell is
used and the worker receives no database credentials.

While this driver owns a child, generation lookup uses that child's bound
generation in memory, including after a validated resume. The operator's seed
file is only the fallback before ownership or after explicit clean cleanup; it
is not rewritten with a plaintext credential and cannot override the current
owner during another source change. Reconciliation-required lookup fails closed
instead of returning the seed. This does not recover or adopt another process.

Stopping an owned child first writes its cooperative stop file and waits for the
existing deadline. If that deadline expires, the driver marks reconciliation
required **before** terminating the child. Forced termination does not count as
proof that capture and lease cleanup completed. The driver retains its child and
generation evidence, reports `SOURCE_RECONCILIATION_REQUIRED`, and refuses both
start modes and repeated stop attempts. Readiness is false even if termination
fails and the child remains alive. Do not automatically recreate a driver to
bypass this condition: the operator must establish resource release and reconcile
the persisted source state before creating a fresh owned lifecycle. Unknown PIDs
are never adopted or terminated.

An already exited child retains the prior explicit cleanup behavior; this change
does not redefine spontaneous-exit recovery. Cancellation still propagates, and
the uncertainty marker survives interrupted forced-termination cleanup.

The model-free configuration capture CLI already exists separately. A production
composition must still bind its resource lease, matching stop file and readiness,
bootstrap a real Backend-issued generation when needed, and bind the inference
runtime to the same physical resource. This driver option alone does not provide
first-camera onboarding, Windows webcam access from Linux Docker, or live-camera
acceptance.

## Generation-bound private image readiness

An explicitly owned runtime may attach `X-SmartSite-Worker-Generation` to a
successful authenticated configuration-image response. The value is a normalized
generation UUID supplied by trusted runtime composition, not request input. The
image body remains unchanged. Unbound defaults, unavailable/stale images,
authorization errors, other cameras and analysis/video endpoints do not gain this
header. The inference CLI supplies it only from its validated source-control
generation override; a model-free capture owner must be explicitly bound too.

An operator's readiness probe must check a fresh image for the exact Camera and
the expected generation while still owning the live child. A listening port or
HTTP 200 alone does not prove that a newly started owner is ready. The header is
technical lifecycle evidence, never a Worker identity, access authorization or
proof that inference, delivery and business alert evaluation succeeded.
