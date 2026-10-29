# AI integration CI follow-up — 27 October 2026

Draft PR44 run40543855095 passed paired-contract and container jobs; Python stopped
at formatting tests/test_replay_preview_source.py, before pytest. Format that fixture.
The clean snapshot exposed a preexisting missing dependency: tracked replay preview
imports uncommitted LiveLatestFrameSource. Include that128-line adapter and its
source/cancellation tests; no experimental realtime configuration is enabled.

Whole-snapshot pytest then reported6failed/1830passed/5skipped. Update the shared
MissingPpePipeline fixture to observe(), retaining cancellation/configuration-swap
and confirmation-recovery assertions. Expanded preview uses a validated event with
region binding and retains UNKNOWN for unobserved Gloves/Boots/Goggles. OpenAPI
expected paths include the already implemented Safety MF06 identity endpoint.

The adapter owns one reader/latest-frame slot, preserves timestamps/session, clears
terminal frames and waits for cleanup through repeated cancellation. Native decoder
timeouts remain a separate real-device requirement, not proven by fake sources.

Final D-only clean snapshot1836passed/5skipped; full Ruff lint and252-file format
pass. Existing optional numpy/cv2/ultralytics checks skip in the selected core env.
No install/model/GPU/camera. A revised expanded fixture initially omitted required
region fields; fixed fixture, not validation. Logs/failures retained under
D:/Ky9-FPT/SmartSiteData/runs/ai-capacity-publication-20261027-v1/.

Only standalone adapter tests are published. Two local realtime-option tests depend
on uncommitted configuration and remain separate with working bytes preserved.
The HTTP test stages only the identity-endpoint assertion, retaining local preview
endpoint expectations unstaged. Full CI must rerun; no Main merge, deployment,
canonical vendoring, runtime activation, cloud storage or production acceptance.

The next CI runfba53b5 reached pytest and exposed tests.* imports that depend on
running python -m pytest. Reproduced with the installed pytest entry point: five
collection errors, then one deferred function-local import in the host test. Six
configuration tests now import same-directory fixtures like the existing suite.
No sys.path mutation, new package or relaxed collection mode. Keep the startup,
scope, drain/cancellation and lock-retention assertions unchanged. Entry-point
pytest is the authoritative local reproduction; earlier module-entry passes did
not cover this import environment.
