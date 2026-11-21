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
