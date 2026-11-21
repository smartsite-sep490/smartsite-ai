# Windows saved-configuration resume

The explicit camera runtime's source-control child profile can consume a trusted
generation in memory on Windows. It does not publish the new generation secret
to a plaintext file or treat `chmod` as a Windows ACL guarantee. The existing
POSIX file publication adapter and its Windows refusal remain unchanged.

The parent source driver must already own this camera/source. It passes the
original Backend-issued generation and exact expected resume manifest through
the existing child environment, never command-line arguments. A reviewed host
profile must allow the source and Site/camera scope. The child validates these
bindings, removes the generation environment key after binding and supplies the
credential directly to the sole tracked LIVE camera preflight. It leaves the
previous credential file byte-identical and never falls back to it on a failed
memory binding. No database credentials are accepted by the source driver.

The worker still verifies current expiry, claims its SQLite session and fetches
the Backend configuration. Camera, generation, session, configuration version,
view revision and bundle hash must match the expected resume manifest before
model loading or capture. Invalid, expired, foreign, replay or multiple-camera
overrides fail closed. A changed PPE/region configuration remains unapplied until
the actual worker acknowledges it; process launch alone is insufficient.

After a Windows child exits, its memory credential is not recoverable from the
old on-disk file. Use the trusted parent generation lifecycle and explicit
reconciliation to restart. Do not launch an ordinary runtime with the old file,
substitute a new credential into pending events, or adopt an unknown running
camera from its PID. Source replacement and first-reference bootstrap are still
separate operations; this profile resumes an already accepted configuration.

This does not make a Windows laptop webcam available inside a Linux Docker
container. Native capture remains on its host; the existing CPU image supports
reviewed reachable sources and explicit Linux devices. Remote website access
also needs the authenticated preview/source-control gateway and a qualified
Backend tracked database profile. Model accuracy, hardware throughput and real
two-client acceptance require separate evidence.
