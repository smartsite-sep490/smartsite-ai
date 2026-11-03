# Trusted supervisor manifest binding

The normal smartsite-ai-camera-runtime --manifest <absolute reviewed manifest> command keeps its existing behavior. A trusted supervisor may additionally supply all three arguments together:

--generation-credential-file <fresh absolute private artifact>
--expected-camera-id <UUID>
--expected-outbox <existing absolute outbox>

Bound mode requires exactly one camera and the same camera ID and normalized stable outbox path already present in the reviewed manifest. It copies the strict manifest in memory, changes only generationCredentialFile and reparses it. Partial overrides, another camera/outbox, relative paths or multiple cameras fail before Backend preflight, model load or source opening. The reviewed manifest file and outbox are never rewritten by this binding helper.

Bound mode disables implicit local .env discovery in Settings. The supervisor still owns the explicit filtered child environment. Legacy unbound invocation keeps its prior Settings behavior. Credential contents, expiry, camera identity, regular-file/no-link checks and session ownership remain the existing worker-generation preflight boundary; this helper does not issue or renew credentials.

Binding does not change partitionedFrameDelivery, generationBoundEventDelivery, experimentalModelProfile, observationSchemaVersion, source or evidence settings. It neither enables bound delivery nor grants business permissions. Runtime/operator must explicitly prepare the intended valid reviewed manifest, including any prerequisite flags and artifact path; this command is not an activation shortcut.

The paired Backend operator contract creates a camera-scoped schema1.0.0 credential file, not a worker enrollment API. The separately approved supervisor prototype uses fixed Python -I -B -m smartsite_ai.tools.run_camera_runtime arguments and these three exact binding flags, shell=false and filtered child environment. That prototype is not an installed Main service. No worker starts merely by importing this module.

This source unit adds no automatic warmup or model invocation. Real issuer/renewal/revocation, cross-process ownership/fencing, private Windows ACL qualification, durable event delivery, graceful native shutdown and actual camera/model performance remain separate release gates.
